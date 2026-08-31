from __future__ import annotations  # keep np.ndarray hints lazy at deploy time

import gc
import io
import os
import re
import time
import threading
from datetime import datetime
import hashlib
import json
import pickle

# Modal-specific imports
import modal

# Define the Modal Stub
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "accelerate==1.6.0",
        "transformers==4.51.3",
        "diffusers",
        "tqdm",
        "numpy",
        "scipy",
        "ml-collections",
        "absl-py",
        "soundfile",
        "librosa",
        "pydub",
    )
    .run_commands(
        "mkdir -p /root/vibevoice",
        "touch /root/vibevoice/__init__.py",
        "ln -s /root/modular /root/vibevoice/modular",
        "ln -s /root/processor /root/vibevoice/processor",
        "ln -s /root/voices /root/vibevoice/voices",
        "ln -s /root/schedule /root/vibevoice/schedule"
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})  # fights fragmentation across waves
    .add_local_dir("backend_modal/modular", remote_path="/root/modular")
    .add_local_dir("backend_modal/processor", remote_path="/root/processor")
    .add_local_dir("backend_modal/voices", remote_path="/root/voices")
    .add_local_dir("text_examples", remote_path="/root/text_examples")
    .add_local_dir("backend_modal/schedule", remote_path="/root/schedule")
)

app = modal.App(
    name="vibevoice-generator",
    image=image,
)

# Heavy imports run only inside the container (deploy-time on a laptop
# doesn't have torch/numpy/librosa — and doesn't need them)
with image.imports():
    import numpy as np
    import librosa
    import soundfile as sf
    import torch

# Create a volume for caching generated audio
cache_volume = modal.Volume.from_name("vibevoice-cache", create_if_missing=True)

@app.cls(
    gpu="A100-40GB",
    scaledown_window=300,
    timeout=7200,  # was 3600: a 120-min+ record attempt hit the 1h ceiling at the
                   # finish line (2026-08-19), losing the whole render. Long-form
                   # with cloned voices runs slower than the preset-voice record
                   # pace (bigger reference prefill per chunk), so give 2h.
    volumes={"/cache": cache_volume}
)
class VibeVoiceModel:
    @modal.enter()
    def load_models(self):
        """Run once when the container starts. Loads both models to GPU."""
        self.model_paths = {
            "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
            "VibeVoice-7B": "vibevoice/VibeVoice-7B",
        }
        self.device = "cuda"
        self.inference_steps = 5
        self.cache_dir = "/cache"
        self.max_cache_size_gb = 10  # Limit cache to 10GB

        # Project-specific imports are moved here to run inside the container
        from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from processor.vibevoice_processor import VibeVoiceProcessor

        print("Entering container and loading models to GPU...")
        
        # Set compiler flags for better performance
        if torch.cuda.is_available() and hasattr(torch, '_inductor'):
            if hasattr(torch._inductor, 'config'):
                torch._inductor.config.conv_1x1_as_mm = True
                torch._inductor.config.coordinate_descent_tuning = True
                torch._inductor.config.epilogue_fusion = False
                torch._inductor.config.coordinate_descent_check_all_directions = True

        self.models = {}
        self.processors = {}
        self.current_model_name = None
        
        # Load all models directly to GPU (A100-40GB holds both; ~17 GB baseline)
        for name, path in self.model_paths.items():
            print(f" - Loading {name} from {path}")
            proc = VibeVoiceProcessor.from_pretrained(path)
            mdl = VibeVoiceForConditionalGenerationInference.from_pretrained(
                path, 
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa"
            ).to(self.device)  # Load directly to GPU
            mdl.eval()
            print(f"  {name} loaded to {self.device}")
            self.processors[name] = proc
            self.models[name] = mdl
        
        # Set default model
        self.current_model_name = "VibeVoice-1.5B"

        self.setup_voice_presets()
        self.ready_at = time.time()  # for cold-start detection in timing reports
        # VRAM baseline right after model load: requests arriving to a GPU far
        # above this are hitting a poisoned container (e.g. a cancelled run's
        # zombie generation thread) and must recycle, not proceed (2026-08-14)
        self.baseline_alloc = torch.cuda.memory_allocated()
        print(f"Model loading complete. VRAM baseline: {self.baseline_alloc/1e9:.1f} GB")

    def _place_model(self, target_name: str):
        """Both models stay on GPU. Just update the active selection."""
        self.current_model_name = target_name
        print(f"Switched to model {target_name}")

    def setup_voice_presets(self):
        self.available_voices = {}
        voices_dir = "/root/voices" # Using remote path from Mount
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            return
        wav_files = [f for f in os.listdir(voices_dir)
                     if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'))]
        for wav_file in wav_files:
            name = os.path.splitext(wav_file)[0]
            self.available_voices[name] = os.path.join(voices_dir, wav_file)
        print(f"Voices loaded: {list(self.available_voices.keys())}")

    def _emit_progress(self, stage: str, pct: float, status: str, log_text: str,
                       audio=None, done: bool = False, extra: dict = None):
        """Package a structured progress update for streaming back to Gradio."""
        payload = {
            "stage": stage,
            "pct": pct,
            "status": status,
            "log": log_text,
        }
        if audio is not None:
            payload["audio"] = audio
        if done:
            payload["done"] = True
        if extra:
            payload.update(extra)
        return payload

    def _generate_cache_key(self, script: str, model_name: str, speakers: list, cfg_scale: float) -> str:
        """Generate a unique cache key for this generation."""
        cache_data = {
            "script": script.strip().lower(),  # Normalize script
            "model": model_name,
            "speakers": list(speakers),  # Order matters: slot N's voice changes the take
            "cfg_scale": cfg_scale,
            "inference_steps": self.inference_steps,
            # Bump to invalidate all prior entries (v1 keys sorted speakers and
            # predates the restored voice assets, so old entries can replay takes
            # with wrong or corrupted voices).
            "pipeline": "chunked-parallel-v2",
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def _get_cached_audio(self, cache_key: str):
        """Check if audio is cached and return it."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    print(f"Cache hit! Loading from {cache_key}")
                    return cached_data['audio'], cached_data['sample_rate']
            except Exception as e:
                print(f"Cache read error: {e}")
        return None, None
    
    def _save_to_cache(self, cache_key: str, audio: np.ndarray, sample_rate: int):
        """Save generated audio to cache."""
        try:
            # Check cache size
            self._cleanup_cache_if_needed()
            
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            cached_data = {
                'audio': audio,
                'sample_rate': sample_rate,
                'timestamp': time.time()
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"Saved to cache: {cache_key}")
            
            # Commit the volume changes
            cache_volume.commit()
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def _cleanup_cache_if_needed(self):
        """Remove old cache files if cache is too large."""
        try:
            cache_files = []
            total_size = 0
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(self.cache_dir, filename)
                    size = os.path.getsize(filepath)
                    mtime = os.path.getmtime(filepath)
                    cache_files.append((filepath, size, mtime))
                    total_size += size
            
            # If cache is too large, remove oldest files
            max_size = self.max_cache_size_gb * 1024 * 1024 * 1024
            if total_size > max_size:
                # Sort by modification time (oldest first)
                cache_files.sort(key=lambda x: x[2])
                
                while total_size > max_size * 0.8 and cache_files:  # Keep 80% full
                    filepath, size, _ = cache_files.pop(0)
                    os.remove(filepath)
                    total_size -= size
                    print(f"Removed old cache: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"Cache cleanup error: {e}")

    # Reference-clip loudness target. Presets were mastered at different levels
    # and user clones come off hot podcast mics; VibeVoice imprints that gap
    # onto the generated speakers, so every clip is levelled to the same
    # speech RMS (~-22 dBFS) before conditioning.
    REF_TARGET_RMS = 0.08
    REF_PEAK_CEILING = 0.95
    REF_MAX_SECONDS = 60   # refs cost 7.5 prompt tokens/s in EVERY batch row —
                           # a minutes-long clone bloats VRAM and slows prefill

    @classmethod
    def _trim_reference(cls, wav: np.ndarray, sr: int) -> np.ndarray:
        """Cap a reference clip at the REF_MAX_SECONDS window with the most speech."""
        limit = int(cls.REF_MAX_SECONDS * sr)
        if len(wav) <= limit:
            return wav
        frame = max(1, int(0.05 * sr))
        n = len(wav) // frame
        rms = np.sqrt(np.mean(np.square(
            wav[:n * frame].astype(np.float32).reshape(n, frame)), axis=1))
        active = (rms > max(0.02, float(rms.max()) * 0.15)).astype(np.float32)
        win = max(1, limit // frame)
        if n <= win:
            return wav[:limit]
        score = np.convolve(active, np.ones(win), mode="valid")
        start = int(np.argmax(score)) * frame
        return wav[start:start + limit]

    @classmethod
    def _normalize_reference(cls, wav: np.ndarray, sr: int) -> np.ndarray:
        if wav.size == 0:
            return wav
        wav = wav.astype(np.float32)
        # Measure RMS over speech only: frame the clip and drop quiet frames so
        # leading/trailing silence can't inflate the gain.
        frame = max(1, int(sr * 0.05))
        n = (len(wav) // frame) * frame
        if n >= frame:
            frames = wav[:n].reshape(-1, frame)
            frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
            active = frame_rms[frame_rms > frame_rms.max() * 0.2]
            rms = float(np.sqrt(np.mean(np.square(active)))) if active.size else 0.0
        else:
            rms = float(np.sqrt(np.mean(np.square(wav))))
        if rms < 1e-6:
            return wav
        gain = cls.REF_TARGET_RMS / rms
        peak = float(np.abs(wav).max()) or 1.0
        gain = min(gain, cls.REF_PEAK_CEILING / peak)  # never clip a quiet-but-peaky clip
        return wav * gain

    def read_audio(self, audio_source, target_sr: int = 24000) -> np.ndarray:
        """audio_source is a file path (preset) or raw audio bytes (user-uploaded clone)."""
        try:
            label = "uploaded clone" if isinstance(audio_source, (bytes, bytearray)) else audio_source
            if isinstance(audio_source, (bytes, bytearray)):
                audio_source = io.BytesIO(audio_source)
            wav, sr = sf.read(audio_source)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            wav = self._trim_reference(wav, target_sr)
            return self._normalize_reference(wav, target_sr)
        except Exception as e:
            print(f"Error reading audio {label}: {e}")
            return np.array([])

    # ---- chunked-parallel generation (2026-08-14) ----------------------------
    # Rendering the whole script as ONE sequential generate call is the slow
    # path (~1.4x slower than realtime). Grouping whole turns into chunks and
    # rendering them as a BATCH in a single call is ~3x faster at batch-4 with
    # no measured quality loss (audio-level parity: WER 0.000/0.071/0.000/0.000
    # solo-vs-batched; seams inaudible with a 0.25 s crossfade). Validation
    # record: LongFlow experiments/p1_flow_head/NOTES.md, Gate Nights 4-5.
    CHUNK_TARGET_WORDS = 200          # ~80-90 s of audio per chunk
    CHUNK_MIN_SCRIPT_WORDS = 250      # was 350; ear-check 2026-08-19 found audible
                                      # rate/volume drift (N8) already at 292 words
                                      # (~98 s) rendered single-pass — drift onset
                                      # beats seam risk well below the old gate
    CROSSFADE_S = 0.25                # GN5 ear-validated
    # Ambitious caps: with OOM backoff (waves split in half and retry), an
    # overshoot costs one halving, never the job. Validated floors were 8/4.
    MAX_BATCH = {"VibeVoice-1.5B": 12, "VibeVoice-7B": 6}  # A100-40GB

    # ---- streaming fast start (2026-08-31) --------------------------------
    # The opening ~FAST_START_WORDS are split into mini-chunks and rendered as
    # their own first wave: a batch of tiny chunks finishes in the wall time
    # of ONE tiny chunk (~20-30 s), delivering ~100 s of playable audio almost
    # immediately, which bridges playback until the full-size waves land.
    # Cost: extra context resets within the opening minute (still turn-aligned,
    # same refs, same crossfade, same quality gate).
    # 350 words ≈ 140 s of audio: enough buffer to bridge the ~2 min wall time
    # of the first full-size wave (250 ran dry ~20 s early — 2026-08-31).
    FAST_START_WORDS = 350
    MINI_CHUNK_WORDS = 50

    @classmethod
    def _split_fast_start(cls, chunks: list, batch_cap: int) -> tuple:
        """Regroup the opening ~FAST_START_WORDS into turn-aligned mini-chunks.

        Returns (new_chunks, n_fast). n_fast <= 1 means fast start didn't
        apply (opening turn too long to split, or nothing to gain).
        """
        # Consume whole chunks from the front until the fast region is big
        # enough to bridge the first full wave — but always leave at least one
        # full-size chunk so the bulk keeps normal context spans.
        take, words = 0, 0
        while take < len(chunks) - 1 and words < cls.FAST_START_WORDS:
            words += len(chunks[take].split())
            take += 1
        fast_lines = [l for c in chunks[:take] for l in c.split("\n")]
        if len(fast_lines) < 2:
            return chunks, 0  # single opening turn; nothing turn-aligned to split
        total_words = sum(len(l.split()) for l in fast_lines)
        # Never build more minis than fit one batch: the whole point is that
        # the fast wave renders in a single batched call.
        target = max(cls.MINI_CHUNK_WORDS, total_words // batch_cap + 1)
        minis, cur, cur_words = [], [], 0
        for line in fast_lines:
            cur.append(line)
            cur_words += len(line.split())
            if cur_words >= target:
                minis.append("\n".join(cur))
                cur, cur_words = [], 0
        if cur:
            if minis and cur_words < target // 3:
                minis[-1] = minis[-1] + "\n" + "\n".join(cur)
            else:
                minis.append("\n".join(cur))
        if len(minis) < 2:
            return chunks, 0
        return minis + chunks[take:], len(minis)

    @classmethod
    def _split_turns_into_chunks(cls, turn_lines: list) -> list:
        """Group whole 'Speaker N:' turn lines into ~CHUNK_TARGET_WORDS chunks.

        Never splits inside a normal turn (turn boundaries are what keep pacing
        natural and identity anchored), so each chunk is itself a valid
        mini-script for the same speaker set. The one exception is an OVERSIZED
        turn — a monologue past ~1.25x CHUNK_TARGET_WORDS. Rendered whole, long
        monologues exhibit the N8 rate/volume drift (ear-confirmed at 292 words:
        speeds up and gets quieter toward the end), so they are sentence-split
        into EVEN pieces which become atomic chunks (never re-grouped). Seams
        get the same 0.25 s crossfade and per-chunk quality gate as turn seams.
        """
        import re as _re
        oversize = int(cls.CHUNK_TARGET_WORDS * 1.25)

        def split_oversized(line):
            m = _re.match(r"^(Speaker\s+\d+\s*:)\s*(.*)$", line, _re.S | _re.I)
            tag, body = (m.group(1), m.group(2)) if m else ("Speaker 1:", line)
            sentences = [s for s in _re.split(r"(?<=[.!?…])\s+", body) if s.strip()]
            if len(sentences) < 2:
                return [line]  # no sentence boundaries to split on
            words = len(body.split())
            n_pieces = max(2, round(words / cls.CHUNK_TARGET_WORDS))
            piece_target = words / n_pieces
            pieces, cur_s, cur_w = [], [], 0
            for sentence in sentences:
                cur_s.append(sentence)
                cur_w += len(sentence.split())
                if cur_w >= piece_target and len(pieces) < n_pieces - 1:
                    pieces.append(f"{tag} {' '.join(cur_s)}")
                    cur_s, cur_w = [], 0
            if cur_s:
                pieces.append(f"{tag} {' '.join(cur_s)}")
            return pieces

        chunks, cur, cur_words = [], [], 0

        def flush():
            nonlocal cur, cur_words
            if cur:
                chunks.append("\n".join(cur))
                cur, cur_words = [], 0

        for line in turn_lines:
            words = len(line.split())
            if words > oversize:
                pieces = split_oversized(line)
                if len(pieces) > 1:
                    flush()
                    chunks.extend(pieces)
                    continue
            cur.append(line)
            cur_words += words
            if cur_words >= cls.CHUNK_TARGET_WORDS:
                flush()
        if cur:
            # avoid a tiny trailing chunk: merge into the previous one
            if chunks and cur_words < cls.CHUNK_TARGET_WORDS // 3:
                chunks[-1] = chunks[-1] + "\n" + "\n".join(cur)
            else:
                chunks.append("\n".join(cur))
        return chunks

    @staticmethod
    def _turn_offsets_in_chunk(audio, sample_rate: int, word_counts: list) -> list:
        """Estimate where each turn starts inside one rendered chunk, in seconds.

        Word counts give a prior for each boundary; the audio's 20 ms energy
        envelope supplies real inter-turn dips. A small DP picks, for every
        boundary, either a genuine energy gap near its prior or the prior
        itself, keeping boundaries ordered. This runs on the raw 24 kHz audio,
        so it catches the brief dips of fast one-phrase exchanges that the
        browser's coarse waveform peaks cannot see.
        """
        import numpy as np
        n_turns = len(word_counts)
        if n_turns == 0:
            return []
        offsets = [0.0]
        if n_turns == 1:
            return offsets
        if len(audio) < sample_rate:
            dur = len(audio) / sample_rate
            total = float(sum(word_counts)) or 1.0
            acc = 0.0
            for w in word_counts[:-1]:
                acc += w
                offsets.append(round(acc / total * dur, 3))
            return offsets

        hop = int(0.02 * sample_rate)
        n_frames = len(audio) // hop
        env = np.sqrt(np.mean(np.square(
            np.asarray(audio[:n_frames * hop], dtype=np.float32).reshape(n_frames, hop)), axis=1))
        env = np.convolve(env, np.ones(3) / 3, mode="same")
        env = env / (float(env.max()) or 1.0)
        duration = n_frames * hop / sample_rate

        total_words = float(sum(word_counts)) or 1.0
        cum = np.cumsum(word_counts).astype(float)
        priors = (cum[:-1] / total_words) * duration          # N-1 boundary priors
        turn_durs = (np.asarray(word_counts, dtype=float) / total_words) * duration

        # Candidate boundaries: runs of near-silence; a boundary sits at the
        # end of a run (the next speaker's onset). Score by run length.
        floor = float(np.percentile(env, 2))
        thresh = max(0.06, min(0.15, floor + 0.05))
        quiet = env < thresh
        gaps = []  # (onset_time_sec, gap_len_sec)
        run = 0
        for f in range(n_frames):
            if quiet[f]:
                run += 1
            else:
                if run >= 2:  # >= 40 ms of quiet
                    gaps.append((f * hop / sample_rate, run * hop / sample_rate))
                run = 0

        # Assign one candidate per boundary with a small DP: each boundary may
        # take a genuine gap near its prior (cheap, cheaper still for long
        # gaps) or fall back to the prior itself, and boundaries must stay
        # ordered. DP (rather than greedy) keeps a mis-grabbed gap at one
        # boundary from cascading into the rest.
        MIN_SEP = 0.3
        PRIOR_COST = 0.8
        cand_sets = []
        for i, prior in enumerate(priors):
            dur_l, dur_r = turn_durs[i], turn_durs[i + 1]
            scale = max(0.4, 0.5 * min(dur_l, dur_r))
            cands = [(min(max(prior, 0.05), duration - 0.05), PRIOR_COST)]
            for t, glen in gaps:
                # A gap can serve this boundary only if it lies inside this
                # boundary's own turns — beyond that it belongs to a neighbour.
                if t < prior - 0.9 * dur_l or t > prior + 0.9 * dur_r:
                    continue
                dev = min(4.0, (abs(t - prior) / scale) ** 2) * 0.5
                cands.append((t, dev - 3.0 * min(glen, 0.6)))
            cand_sets.append(cands)

        INF = float("inf")
        best = [c for _, c in cand_sets[0]]
        back = [[-1] * len(cs) for cs in cand_sets]
        for i in range(1, len(cand_sets)):
            cur = [INF] * len(cand_sets[i])
            for j, (t, cost) in enumerate(cand_sets[i]):
                for k, (pt, _) in enumerate(cand_sets[i - 1]):
                    if pt <= t - MIN_SEP and best[k] + cost < cur[j]:
                        cur[j] = best[k] + cost
                        back[i][j] = k
            # If ordering left no feasible predecessor, chain from the best
            # previous state anyway — monotonicity is restored below.
            for j in range(len(cur)):
                if cur[j] == INF:
                    k = int(np.argmin(best))
                    cur[j] = best[k] + cand_sets[i][j][1] + 2.0
                    back[i][j] = k
            best = cur

        j = int(np.argmin(best))
        chosen = [0.0] * len(cand_sets)
        for i in range(len(cand_sets) - 1, -1, -1):
            chosen[i] = cand_sets[i][j][0]
            j = back[i][j] if i > 0 else 0
        # Enforce strict ordering whatever the DP produced.
        prev = 0.0
        results = []
        for c in chosen:
            prev = max(c, prev + MIN_SEP)
            results.append(min(prev, duration - 0.05))
        return offsets + results

    @classmethod
    def _chunk_starts(cls, pieces: list, sample_rate: int) -> list:
        """Where each chunk begins in the crossfade-concatenated take, in seconds.

        Mirrors _crossfade_concat: every seam overlaps the next piece by up to
        CROSSFADE_S, so piece i+1 starts that much before the naive cumulative sum.
        """
        pieces = [p for p in pieces if p is not None and len(p) > 0]
        xf = int(cls.CROSSFADE_S * sample_rate)
        starts, pos = [], 0
        for i, p in enumerate(pieces):
            starts.append(round(pos / sample_rate, 3))
            pos += len(p) - (min(xf, len(p)) if i + 1 < len(pieces) else 0)
        return starts

    @classmethod
    def _crossfade_concat(cls, pieces: list, sample_rate: int) -> np.ndarray:
        """Concatenate audio pieces with a short linear crossfade at each seam."""
        pieces = [p for p in pieces if p is not None and len(p) > 0]
        if not pieces:
            return np.array([])
        xf = int(cls.CROSSFADE_S * sample_rate)
        out = pieces[0]
        for p in pieces[1:]:
            n = min(xf, len(out), len(p))
            if n <= 0:
                out = np.concatenate([out, p])
                continue
            fade = np.linspace(1.0, 0.0, n)
            seam = out[-n:] * fade + p[:n] * (1.0 - fade)
            out = np.concatenate([out[:-n], seam, p[n:]])
        return out

    # ---- detect-and-reroll quality gate (2026-08-14) ---------------------
    # Transient-babble glitch rate is <1 per 46 min, so the gate must be
    # CONSERVATIVE: only re-roll on egregious failures, and fail OPEN (any
    # gate error = pass) so it can never block a render. Metrics per chunk:
    #   rate     — words/sec of audio; healthy renders measure ~2.2-2.5
    #   silence  — fraction of 50 ms frames near-silent (drone/stall symptom)
    #   flatness — mean spectral flatness (static/babble is noise-like, ~1.0;
    #              speech is ~0.01-0.2)
    # One reroll attempt per failing chunk on a fresh torch seed; keep the
    # reroll only if it gates better than the original.
    GATE_WPS_MIN = 1.2            # slower = audio stretched/droning
    GATE_WPS_MAX = 4.2            # faster = audio truncated/skipped words
    GATE_SILENCE_MAX = 0.5        # >50% near-silent frames
    GATE_FLATNESS_MAX = 0.5       # white noise ≈ 1.0, speech ≪ 0.5

    @classmethod
    def _chunk_quality(cls, chunk_text: str, audio, sample_rate: int):
        """Score one rendered chunk. Returns (ok, badness, reason).

        badness is a monotone severity score so a reroll can be compared
        against the original even when both fail. Fails open on any error.
        """
        import re as _re
        try:
            if audio is None or len(audio) == 0:
                return False, float("inf"), "no audio"
            words = len(_re.sub(r"(?mi)^\s*Speaker\s+\d+\s*:", "", chunk_text).split())
            dur = len(audio) / sample_rate
            if words == 0 or dur <= 0:
                return True, 0.0, "unscorable"
            wps = words / dur

            frame = int(0.05 * sample_rate)
            n = (len(audio) // frame) * frame
            frames = np.asarray(audio[:n], dtype=np.float64).reshape(-1, frame)
            rms = np.sqrt((frames ** 2).mean(axis=1))
            overall = rms.mean() or 1e-9
            silence_frac = float((rms < 0.05 * overall).mean())

            # Spectral flatness on up to 40 evenly-spaced frames (cheap):
            # geometric/arithmetic mean ratio of the power spectrum.
            idx = np.linspace(0, len(frames) - 1, min(40, len(frames))).astype(int)
            spec = np.abs(np.fft.rfft(frames[idx], axis=1)) ** 2 + 1e-12
            flatness = float(np.exp(np.log(spec).mean(axis=1)).mean()
                             / spec.mean(axis=1).mean())

            badness = 0.0
            reasons = []
            if wps < cls.GATE_WPS_MIN or wps > cls.GATE_WPS_MAX:
                badness += abs(wps - np.clip(wps, cls.GATE_WPS_MIN, cls.GATE_WPS_MAX))
                reasons.append(f"rate {wps:.2f} wps")
            if silence_frac > cls.GATE_SILENCE_MAX:
                badness += silence_frac - cls.GATE_SILENCE_MAX
                reasons.append(f"silence {silence_frac:.2f}")
            if flatness > cls.GATE_FLATNESS_MAX:
                badness += flatness - cls.GATE_FLATNESS_MAX
                reasons.append(f"flatness {flatness:.2f}")
            if reasons:
                return False, badness, ", ".join(reasons)
            return True, 0.0, f"ok (wps {wps:.2f}, sil {silence_frac:.2f}, flat {flatness:.3f})"
        except Exception as gate_err:  # fail open — never block a render
            print(f"Quality gate error (passing chunk through): {gate_err}")
            return True, 0.0, "gate error"

    @staticmethod
    def _infer_num_speakers_from_script(script: str) -> int:
        """
        Infer number of speakers by counting distinct 'Speaker X:' tags in the script.
        Robust to 0- or 1-indexed labels and repeated turns.
        Falls back to 1 if none found.
        """
        import re
        ids = re.findall(r'(?mi)^\s*Speaker\s+(\d+)\s*:', script)
        return len({int(x) for x in ids}) if ids else 1

    @modal.method()
    def get_example_scripts(self):
        examples_dir = "/root/text_examples"
        example_scripts = []
        example_scripts_natural = []
        if not os.path.exists(examples_dir):
            return [], []

        original_files = [
            "1p_ai_tedtalk.txt",
            "1p_politcal_speech.txt",
            "2p_financeipo_meeting.txt",
            "2p_telehealth_meeting.txt",
            "3p_military_meeting.txt",
            "3p_oil_meeting.txt",
            "4p_gamecreation_meeting.txt",
            "4p_product_meeting.txt"
        ]
        
        for txt_file in original_files:
            try:
                with open(os.path.join(examples_dir, txt_file), 'r', encoding='utf-8') as f:
                    script_content = f.read().strip()
                if script_content:
                    num_speakers = self._infer_num_speakers_from_script(script_content)
                    example_scripts.append([num_speakers, script_content])
                    
                natural_file = txt_file.replace('.txt', '_natural.txt')
                natural_path = os.path.join(examples_dir, natural_file)
                if os.path.exists(natural_path):
                    with open(natural_path, 'r', encoding='utf-8') as f:
                        natural_content = f.read().strip()
                    if natural_content:
                        num_speakers = self._infer_num_speakers_from_script(natural_content)
                        example_scripts_natural.append([num_speakers, natural_content])
                else:
                    example_scripts_natural.append([num_speakers, script_content])
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")
        
        return example_scripts, example_scripts_natural

    @modal.method()
    def generate_podcast(self,
                         num_speakers: int,
                         script: str,
                         model_name: str,
                         cfg_scale: float,
                         speaker_1: str = None,
                         speaker_2: str = None,
                         speaker_3: str = None,
                         speaker_4: str = None,
                         custom_audio_1: bytes = None,
                         custom_audio_2: bytes = None,
                         custom_audio_3: bytes = None,
                         custom_audio_4: bytes = None):
        """
        This is the main inference function that will be called from the frontend.
        Yields progress updates during generation. A speaker slot uses its
        custom_audio_N clip (user-uploaded voice clone) when provided, otherwise
        falls back to the named preset in speaker_N.
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")

            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            selected_custom_audio = [custom_audio_1, custom_audio_2, custom_audio_3, custom_audio_4][:num_speakers]
            has_custom_voice = any(a is not None for a in selected_custom_audio)

            speaker_labels = [
                f"custom clone ({len(audio)} bytes)" if audio is not None else (name or "unset")
                for name, audio in zip(selected_speakers, selected_custom_audio)
            ]
            log_lines = [
                f"Generating conference with {num_speakers} speakers",
                f"Model: {model_name}",
                f"Parameters: CFG Scale={cfg_scale}",
                f"Speakers: {', '.join(speaker_labels)}",
            ]
            log_text = "\n".join(log_lines)

            # Check cache first — skipped entirely for custom voice clones, since a cache
            # hit would silently return a DIFFERENT user's cloned-voice audio for what
            # looks like the same request (the clone's actual bytes aren't part of the key).
            cache_key = None
            if not has_custom_voice:
                cache_key = self._generate_cache_key(script, model_name, selected_speakers, cfg_scale)
                cached_audio, cached_sr = self._get_cached_audio(cache_key)
                if cached_audio is not None:
                    log_lines.append("Cache hit! Returning previously generated audio.")
                    log_text = "\n".join(log_lines)
                    yield self._emit_progress(
                        stage="complete", pct=100,
                        status="Loaded from cache.",
                        log_text=log_text,
                        audio=(cached_sr, cached_audio), done=True,
                    )
                    return

            yield self._emit_progress(
                stage="queued",
                pct=5,
                status="Queued GPU job and validating inputs…",
                log_text=log_text,
            )

            # Move the selected model to GPU, others to CPU
            yield self._emit_progress(
                stage="loading_model",
                pct=15,
                status=f"Loading {model_name} weights to GPU…",
                log_text=log_text,
            )
            self._place_model(model_name)

            # GPU health check: a poisoned container (zombie thread from a
            # cancelled run) shows allocated VRAM far above the post-load
            # baseline. Proceeding would OOM every wave — recycle instead.
            gc.collect()
            torch.cuda.empty_cache()
            alloc = torch.cuda.memory_allocated()
            baseline = getattr(self, "baseline_alloc", alloc)
            if alloc > baseline + 6e9:
                yield self._emit_progress(
                    stage="error", pct=0,
                    status="GPU busy from a previous cancelled job — container is restarting. Please retry in about a minute.",
                    log_text=log_text + "\nGPU poisoned (leftover allocation detected); recycling container.",
                )
                os._exit(1)

            model = self.models[model_name]
            processor = self.processors[model_name]
            model.set_ddpm_inference_steps(num_steps=self.inference_steps)

            print(f"Generating with model {model_name} on {self.device}")

            if not script.strip():
                raise ValueError("Error: Please provide a script.")

            script = script.replace("’", "'").replace("‘", "'")

            if not 1 <= num_speakers <= 4:
                raise ValueError("Error: Number of speakers must be between 1 and 4.")

            for i, (speaker_name, custom_audio) in enumerate(zip(selected_speakers, selected_custom_audio)):
                if custom_audio is not None:
                    continue
                if not speaker_name or speaker_name not in self.available_voices:
                    raise ValueError(f"Error: Please select a valid speaker for Speaker {i+1}.")

            log_lines.append("Loading voice samples…")
            log_text = "\n".join(log_lines)
            yield self._emit_progress(
                stage="loading_voices",
                pct=25,
                status="Loading reference voices…",
                log_text=log_text,
            )

            voice_samples = []
            for i, (speaker_name, custom_audio) in enumerate(zip(selected_speakers, selected_custom_audio)):
                label = speaker_labels[i]
                audio_source = custom_audio if custom_audio is not None else self.available_voices[speaker_name]
                audio_data = self.read_audio(audio_source)
                if len(audio_data) == 0:
                    raise ValueError(f"Error: Failed to load audio for Speaker {i+1} ({label}). Is the file a valid audio clip?")
                voice_samples.append(audio_data)
                voice_pct = 25 + ((i + 1) / len(selected_speakers)) * 15
                log_lines.append(f"Loaded voice {i+1}/{len(selected_speakers)}: {label}")
                log_text = "\n".join(log_lines)
                yield self._emit_progress(
                    stage="loading_voices",
                    pct=voice_pct,
                    status=f"Loaded {label}",
                    log_text=log_text,
                )

            log_lines.append(f"Loaded {len(voice_samples)} voice samples")
            log_text = "\n".join(log_lines)

            lines = script.strip().split('\n')
            formatted_script_lines = []
            for line in lines:
                line = line.strip()
                if not line: continue
                if line.startswith('Speaker ') and ':' in line:
                    formatted_script_lines.append(line)
                else:
                    speaker_id = len(formatted_script_lines) % num_speakers
                    formatted_script_lines.append(f"Speaker {speaker_id}: {line}")

            formatted_script = '\n'.join(formatted_script_lines)
            log_lines.append(f"Formatted script with {len(formatted_script_lines)} turns")
            log_text = "\n".join(log_lines)
            yield self._emit_progress(
                stage="preparing_inputs",
                pct=50,
                status="Formatting script and preparing tensors…",
                log_text=log_text,
            )

            # ---- chunked-parallel fast path (falls back to monolithic) ------
            script_words = len(formatted_script.split())
            chunks = (self._split_turns_into_chunks(formatted_script_lines)
                      if script_words >= self.CHUNK_MIN_SCRIPT_WORDS else [formatted_script])
            batch_cap = self.MAX_BATCH.get(model_name, 4)
            use_parallel = len(chunks) > 1

            n_fast = 0
            if use_parallel:
                chunks, n_fast = self._split_fast_start(chunks, batch_cap)

            if use_parallel:
                log_lines.append(
                    f"Parallel mode: {len(chunks)} chunks"
                    + (f" (fast-start {n_fast})" if n_fast else "")
                    + f", batches of up to {batch_cap}"
                )
            else:
                log_lines.append("Short script: single-pass generation")
            log_text = "\n".join(log_lines)
            yield self._emit_progress(
                stage="generating_audio",
                pct=70,
                status="Running VibeVoice diffusion…",
                log_text=log_text,
            )

            # The processor pairs voices with speakers POSITIONALLY after
            # normalizing ids, and only declares voice_samples[:n_speakers_in_
            # _chunk]. A chunk whose speaker set isn't {1..k} therefore gets
            # wrong or undeclared voices — e.g. a chunk containing only
            # Speaker 2 renders in Speaker 1's voice. Remap each chunk's ids
            # to a dense 1..k and pass exactly that chunk's voices, in order.
            _speaker_line = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE | re.DOTALL)
            id_base = 0 if any(
                _speaker_line.match(l) and int(_speaker_line.match(l).group(1)) == 0
                for l in formatted_script_lines
            ) else 1

            def _chunk_voice_view(chunk_text):
                """(remapped_text, voices_for_this_chunk) with dense 1..k ids."""
                mapping, voices, out = {}, [], []
                for line in chunk_text.split("\n"):
                    m = _speaker_line.match(line)
                    if not m:
                        out.append(line)
                        continue
                    sid = int(m.group(1))
                    if sid not in mapping:
                        mapping[sid] = len(mapping) + 1
                        voices.append(voice_samples[(sid - id_base) % len(voice_samples)])
                    out.append(f"Speaker {mapping[sid]}: {m.group(2)}")
                return "\n".join(out), (voices or [voice_samples[0]])

            def _generate_batch(text_list):
                """One batched generate call; returns list of np audio (or None)."""
                views = [_chunk_voice_view(t) for t in text_list]
                batch_inputs = processor(
                    text=[t for t, _ in views],
                    voice_samples=[v for _, v in views],
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(self.device)
                result, exc = {}, {}

                def _run():
                    try:
                        with torch.inference_mode():
                            result['outputs'] = model.generate(
                                **batch_inputs,
                                max_new_tokens=None,
                                cfg_scale=cfg_scale,
                                tokenizer=processor.tokenizer,
                                generation_config={'do_sample': False},
                                verbose=False,
                            )
                    except Exception as gen_err:
                        # store the MESSAGE, never the exception object: its
                        # traceback pins every GPU tensor of the failed attempt
                        # and makes OOM-backoff retries futile (2026-08-14 bug)
                        exc['error'] = f"{type(gen_err).__name__}: {gen_err}"

                t = threading.Thread(target=_run, daemon=True)
                t.start()
                try:
                    while t.is_alive():
                        yield None  # keep-alive tick for the caller
                        time.sleep(5)
                except GeneratorExit:
                    # Client cancelled mid-generation. Python cannot kill the
                    # worker thread, and a surviving thread becomes a zombie
                    # that eats the GPU for every later request on this warm
                    # container (observed 2026-08-14). Recycle the container.
                    print("Run cancelled mid-generation; recycling container "
                          "so no zombie thread survives.")
                    os._exit(1)
                t.join()
                if 'error' in exc:
                    result.clear()
                    del batch_inputs
                    gc.collect()
                    torch.cuda.empty_cache()
                    raise RuntimeError(exc['error'])
                outputs = result.get('outputs')
                if outputs is None or not hasattr(outputs, 'speech_outputs'):
                    raise RuntimeError("Generation produced no outputs.")
                pieces = []
                for so in outputs.speech_outputs[:len(text_list)]:
                    if so is None:
                        pieces.append(None)
                    else:
                        a = so.cpu().float().numpy()
                        pieces.append(a.squeeze() if a.ndim > 1 else a)
                # release this wave's GPU memory before the next wave starts
                del outputs
                result.clear()
                del batch_inputs
                gc.collect()
                torch.cuda.empty_cache()
                yield pieces

            def _wave_with_backoff(wave):
                """Run a wave; on CUDA OOM, split in half and retry recursively.

                Yields None keep-alive ticks, then exactly one final list of
                audio pieces (in wave order)."""
                try:
                    for tick in _generate_batch(wave):
                        yield tick
                    return
                except RuntimeError as e:
                    msg = str(e).lower()
                    if len(wave) == 1 or ("out of memory" not in msg and "cuda" not in msg):
                        raise
                    print(f"OOM at batch {len(wave)}; backing off to halves.")
                    gc.collect()  # must run BEFORE empty_cache to actually free tensors
                    torch.cuda.empty_cache()
                mid = (len(wave) + 1) // 2
                combined = []
                for sub in (wave[:mid], wave[mid:]):
                    if not sub:
                        continue
                    for tick in _wave_with_backoff(sub):
                        if tick is None:
                            yield None
                        else:
                            combined.extend(tick)
                yield combined

            start_time = time.time()
            audio = None
            ran_parallel = False
            try:
                # The fast-start minis form their own first wave so their wall
                # time is one mini render, not one full chunk render.
                if n_fast > 1:
                    rest = chunks[n_fast:]
                    waves = [chunks[:n_fast]] + [
                        rest[i:i + batch_cap] for i in range(0, len(rest), batch_cap)
                    ]
                else:
                    waves = [chunks[i:i + batch_cap] for i in range(0, len(chunks), batch_cap)]
                all_pieces = []
                rerolled_total = 0
                for wi, wave in enumerate(waves):
                    wave_pieces = []
                    for tick in _wave_with_backoff(wave):
                        if tick is None:
                            elapsed = time.time() - start_time
                            pct_hint = min(88, 70 + int(elapsed // 5))
                            yield self._emit_progress(
                                stage="generating_audio",
                                pct=pct_hint,
                                status=(f"Rendering wave {wi + 1}/{len(waves)} "
                                        f"({len(wave)} chunks in parallel)… "
                                        f"{int(elapsed)}s elapsed"),
                                log_text=log_text,
                            )
                        else:
                            wave_pieces = tick

                    # ---- quality gate: score every chunk, reroll failures --
                    if len(wave_pieces) == len(wave):
                        gates = [self._chunk_quality(wave[ci], wave_pieces[ci], 24000)
                                 for ci in range(len(wave))]
                        failed = [ci for ci, (ok, _, _) in enumerate(gates) if not ok]
                        for ci, (ok, _, reason) in enumerate(gates):
                            chunk_no = len(all_pieces) + ci + 1
                            # container-log every chunk's metrics (calibration data)
                            print(f"gate: chunk {chunk_no}: {reason}")
                            if not ok:
                                log_lines.append(
                                    f"Quality gate: chunk {chunk_no} FAILED ({reason}) — rerolling")
                        if failed:
                            log_text = "\n".join(log_lines)
                            torch.manual_seed((int(start_time * 1000) + wi * 7919) % (2 ** 31))
                            reroll_pieces = []
                            for tick in _wave_with_backoff([wave[ci] for ci in failed]):
                                if tick is None:
                                    elapsed = time.time() - start_time
                                    yield self._emit_progress(
                                        stage="generating_audio",
                                        pct=min(88, 70 + int(elapsed // 5)),
                                        status=(f"Re-rolling {len(failed)} flagged "
                                                f"chunk(s) on a new seed… "
                                                f"{int(elapsed)}s elapsed"),
                                        log_text=log_text,
                                    )
                                else:
                                    reroll_pieces = tick
                            for j, ci in enumerate(failed):
                                if j >= len(reroll_pieces):
                                    break
                                ok2, bad2, reason2 = self._chunk_quality(
                                    wave[ci], reroll_pieces[j], 24000)
                                chunk_no = len(all_pieces) + ci + 1
                                if ok2 or bad2 < gates[ci][1]:
                                    wave_pieces[ci] = reroll_pieces[j]
                                    rerolled_total += 1
                                    log_lines.append(
                                        f"Quality gate: chunk {chunk_no} reroll "
                                        f"{'passed' if ok2 else f'improved ({reason2})'} — kept")
                                else:
                                    log_lines.append(
                                        f"Quality gate: chunk {chunk_no} reroll no better "
                                        f"({reason2}) — keeping original")
                            log_text = "\n".join(log_lines)
                    # ---- stream this wave's chunks to the client ----------
                    # Emitted only AFTER the gate verdict, so streamed audio
                    # is exactly what the final take will contain.
                    base = len(all_pieces)
                    all_pieces.extend(wave_pieces)
                    for ci, piece in enumerate(wave_pieces):
                        if piece is None or len(piece) == 0:
                            continue
                        yield self._emit_progress(
                            stage="chunk_audio",
                            pct=min(88, 70 + int(18 * (base + ci + 1) / len(chunks))),
                            status=f"Chunk {base + ci + 1}/{len(chunks)} rendered",
                            log_text=log_text,
                            extra={
                                "chunk_audio": (24000, piece),
                                "chunk_index": base + ci,
                                "chunk_total": len(chunks),
                            },
                        )
                if any(p is None for p in all_pieces):
                    raise RuntimeError("A chunk produced no audio.")
                if rerolled_total == 0:
                    log_lines.append(f"Quality gate: all {len(all_pieces)} chunks passed")
                log_text = "\n".join(log_lines)
                audio = (self._crossfade_concat(all_pieces, 24000)
                         if use_parallel else all_pieces[0])
                final_pieces = all_pieces
                ran_parallel = use_parallel
            except Exception as fast_err:
                # Fall back to SEQUENTIAL PER-CHUNK, never monolithic: a long
                # monolithic render triggers the model's rate/drift defects
                # (finding N8) — turn-split chunks must be preserved even at
                # batch size 1. Only truly short scripts run monolithic.
                if not use_parallel:
                    raise
                print(f"Parallel path failed ({fast_err}); sequential per-chunk fallback.")
                log_lines.append("Parallel path failed; sequential per-chunk fallback.")
                log_text = "\n".join(log_lines)
                gc.collect()
                torch.cuda.empty_cache()
                seq_pieces = []
                for ci, chunk in enumerate(chunks):
                    for tick in _generate_batch([chunk]):
                        if tick is None:
                            elapsed = time.time() - start_time
                            yield self._emit_progress(
                                stage="generating_audio",
                                pct=min(88, 70 + int(elapsed // 5)),
                                status=(f"Sequential fallback: chunk {ci + 1}/"
                                        f"{len(chunks)}… {int(elapsed)}s elapsed"),
                                log_text=log_text,
                            )
                        else:
                            seq_pieces.extend(tick)
                    piece = seq_pieces[ci] if ci < len(seq_pieces) else None
                    if piece is not None and len(piece) > 0:
                        yield self._emit_progress(
                            stage="chunk_audio",
                            pct=min(88, 70 + int(18 * (ci + 1) / len(chunks))),
                            status=f"Chunk {ci + 1}/{len(chunks)} rendered",
                            log_text=log_text,
                            extra={
                                "chunk_audio": (24000, piece),
                                "chunk_index": ci,
                                "chunk_total": len(chunks),
                            },
                        )
                if any(p is None for p in seq_pieces) or not seq_pieces:
                    raise RuntimeError("Error: No audio was generated by the model.")
                audio = self._crossfade_concat(seq_pieces, 24000)
                final_pieces = seq_pieces

            generation_time = time.time() - start_time
            audio_s = len(audio) / 24000 if audio is not None else 0
            container_age = time.time() - getattr(self, "ready_at", time.time())
            cold = container_age < 120  # this request likely paid the container boot
            speed = (audio_s / generation_time) if generation_time > 0 else 0
            log_lines.append("—— timing ——")
            log_lines.append(
                f"Container: {'COLD START (boot + model load happened before this job)' if cold else 'warm'}"
            )
            log_lines.append(
                f"Pure generation: {generation_time:.1f}s for {audio_s:.1f}s of audio "
                f"→ {speed:.2f}x realtime"
            )
            if ran_parallel:
                n_waves = (len(chunks) + batch_cap - 1) // batch_cap
                log_lines.append(f"Mode: parallel — {len(chunks)} chunks in {n_waves} wave(s)")
            else:
                log_lines.append("Mode: sequential chunks (fallback)" if use_parallel else "Mode: single-pass")
            log_lines.append("Processing audio output…")
            log_text = "\n".join(log_lines)
            yield self._emit_progress(
                stage="processing_audio",
                pct=90,
                status="Post-processing audio output…",
                log_text=log_text,
            )

            if audio is None or len(audio) == 0:
                raise RuntimeError("Error: No audio was generated by the model.")

            sample_rate = 24000
            total_duration = len(audio) / sample_rate
            log_lines.append(f"Audio duration: {total_duration:.2f} seconds")

            if cache_key is not None:
                self._save_to_cache(cache_key, audio, sample_rate)
            log_lines.append("Complete!")
            log_text = "\n".join(log_lines)

            # Timing map: where each rendered chunk starts in the final take,
            # how many script turns it covers, and — segmented from each
            # chunk's own energy envelope — where every individual turn
            # begins. The frontend drives captions off these instead of
            # guessing from word counts. Fail-open: captions degrade, renders
            # never break.
            timing_extra = None
            try:
                if len(final_pieces) == len(chunks):
                    chunk_starts = self._chunk_starts(final_pieces, sample_rate)
                    timing_extra = {
                        "chunk_starts_sec": chunk_starts,
                        "chunk_turn_counts": [c.count("\n") + 1 for c in chunks],
                    }
                    turn_starts = []
                    for piece, chunk_text, s0 in zip(final_pieces, chunks, chunk_starts):
                        lines = chunk_text.split("\n")
                        wcs = [max(1, len(l.split(":", 1)[-1].split())) for l in lines]
                        offs = self._turn_offsets_in_chunk(piece, sample_rate, wcs)
                        turn_starts.extend(round(s0 + o, 3) for o in offs)
                    if len(turn_starts) == sum(timing_extra["chunk_turn_counts"]):
                        timing_extra["turn_starts_sec"] = turn_starts
            except Exception as timing_err:
                print(f"Turn timing map failed (non-fatal): {timing_err}")

            yield self._emit_progress(
                stage="complete",
                pct=100,
                status="Conference ready to download.",
                log_text=log_text,
                audio=(sample_rate, audio),
                done=True,
                extra=timing_extra,
            )

        except Exception as e:
            import traceback
            error_msg = f"❌ An unexpected error occurred on Modal: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            # Yield error state
            yield self._emit_progress(
                stage="error",
                pct=0,
                status="Generation failed.",
                log_text=error_msg,
            )
