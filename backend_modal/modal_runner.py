from __future__ import annotations  # keep np.ndarray hints lazy at deploy time

import gc
import io
import os
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
    timeout=3600,  # long-form renders (90-min scripts ~25 min wall) need headroom
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
                       audio=None, done: bool = False):
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

    def read_audio(self, audio_source, target_sr: int = 24000) -> np.ndarray:
        """audio_source is a file path (preset) or raw audio bytes (user-uploaded clone)."""
        try:
            if isinstance(audio_source, (bytes, bytearray)):
                audio_source = io.BytesIO(audio_source)
            wav, sr = sf.read(audio_source)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
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

            if use_parallel:
                log_lines.append(
                    f"Parallel mode: {len(chunks)} chunks, batches of up to {batch_cap}"
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

            def _generate_batch(text_list):
                """One batched generate call; returns list of np audio (or None)."""
                batch_inputs = processor(
                    text=text_list,
                    voice_samples=[voice_samples] * len(text_list),
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
                            chunk_no = wi * batch_cap + ci + 1
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
                                chunk_no = wi * batch_cap + ci + 1
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
                    all_pieces.extend(wave_pieces)
                if any(p is None for p in all_pieces):
                    raise RuntimeError("A chunk produced no audio.")
                if rerolled_total == 0:
                    log_lines.append(f"Quality gate: all {len(all_pieces)} chunks passed")
                log_text = "\n".join(log_lines)
                audio = (self._crossfade_concat(all_pieces, 24000)
                         if use_parallel else all_pieces[0])
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
                if any(p is None for p in seq_pieces) or not seq_pieces:
                    raise RuntimeError("Error: No audio was generated by the model.")
                audio = self._crossfade_concat(seq_pieces, 24000)

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

            yield self._emit_progress(
                stage="complete",
                pct=100,
                status="Conference ready to download.",
                log_text=log_text,
                audio=(sample_rate, audio),
                done=True,
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
