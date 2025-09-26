import os
import time
import threading
import numpy as np
import librosa
import soundfile as sf
import torch
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

# Create a volume for caching generated audio
cache_volume = modal.Volume.from_name("vibevoice-cache", create_if_missing=True)

@app.cls(
    gpu="A100-40GB", 
    scaledown_window=300,
    volumes={"/cache": cache_volume}
)
class VibeVoiceModel:
    def __init__(self):
        self.model_paths = {
            "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
            "VibeVoice-7B": "vibevoice/VibeVoice-7B",
        }
        self.device = "cuda"
        self.inference_steps = 5
        self.cache_dir = "/cache"
        self.max_cache_size_gb = 10  # Limit cache to 10GB

    @modal.enter()
    def load_models(self):
        """
        This method is run once when the container starts.
        With A10G (24GB), we can load both models to GPU.
        """
        # Project-specific imports are moved here to run inside the container
        from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from processor.vibevoice_processor import VibeVoiceProcessor

        print("Entering container and loading models to GPU (A10G with 24GB)...")
        
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
        
        # Load all models directly to GPU (A10G has enough memory)
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
        print("Model loading complete.")
    
    def _place_model(self, target_name: str):
        """
        With A10G, both models stay on GPU. Just update the current model.
        """
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
            "speakers": sorted(speakers),  # Sort for consistency
            "cfg_scale": cfg_scale,
            "inference_steps": self.inference_steps
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

    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])

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
                         speaker_4: str = None):
        """
        This is the main inference function that will be called from the Gradio app.
        Yields progress updates during generation.
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")

            # Initialize log scaffold
            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            log_lines = [
                f"Generating conference with {num_speakers} speakers",
                f"Model: {model_name}",
                f"Parameters: CFG Scale={cfg_scale}",
                f"Speakers: {', '.join(selected_speakers)}",
            ]
            log_text = "\n".join(log_lines)

            # Emit initial status before heavy work kicks in
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

            model = self.models[model_name]
            processor = self.processors[model_name]
            model.set_ddpm_inference_steps(num_steps=self.inference_steps)

            print(f"Generating with model {model_name} on {self.device}")

            if not script.strip():
                raise ValueError("Error: Please provide a script.")

            script = script.replace("'", "'")

            if not 1 <= num_speakers <= 4:
                raise ValueError("Error: Number of speakers must be between 1 and 4.")

            for i, speaker_name in enumerate(selected_speakers):
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
            for i, speaker_name in enumerate(selected_speakers):
                audio_path = self.available_voices[speaker_name]
                audio_data = self.read_audio(audio_path)
                if len(audio_data) == 0:
                    raise ValueError(f"Error: Failed to load audio for {speaker_name}")
                voice_samples.append(audio_data)
                voice_pct = 25 + ((i + 1) / len(selected_speakers)) * 15
                log_lines.append(f"Loaded voice {i+1}/{len(selected_speakers)}: {speaker_name}")
                log_text = "\n".join(log_lines)
                yield self._emit_progress(
                    stage="loading_voices",
                    pct=voice_pct,
                    status=f"Loaded {speaker_name}",
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

            inputs = processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            log_lines.append("Inputs prepared; starting diffusion generation…")
            log_text = "\n".join(log_lines)
            yield self._emit_progress(
                stage="generating_audio",
                pct=70,
                status="Running VibeVoice diffusion (this may take 1-2 minutes)…",
                log_text=log_text,
            )

            start_time = time.time()
            result_container = {}
            exception_container = {}

            def _run_generation():
                try:
                    with torch.inference_mode():
                        result_container['outputs'] = model.generate(
                            **inputs,
                            max_new_tokens=None,
                            cfg_scale=cfg_scale,
                            tokenizer=processor.tokenizer,
                            generation_config={'do_sample': False},
                            verbose=False,
                        )
                except Exception as gen_err:
                    exception_container['error'] = gen_err

            generation_thread = threading.Thread(target=_run_generation, daemon=True)
            generation_thread.start()

            # Emit keep-alive progress while the heavy generation is running
            while generation_thread.is_alive():
                elapsed = time.time() - start_time
                status_msg = f"Running VibeVoice diffusion… {int(elapsed)}s elapsed"
                pct_hint = min(88, 70 + int(elapsed // 5))
                yield self._emit_progress(
                    stage="generating_audio",
                    pct=pct_hint,
                    status=status_msg,
                    log_text=log_text,
                )
                time.sleep(5)

            generation_thread.join()
            if 'error' in exception_container:
                raise exception_container['error']

            outputs = result_container.get('outputs')
            if outputs is None:
                raise RuntimeError("Generation thread finished without producing outputs.")

            generation_time = time.time() - start_time

            log_lines.append(f"Generation completed in {generation_time:.2f} seconds")
            log_lines.append("Processing audio output…")
            log_text = "\n".join(log_lines)
            yield self._emit_progress(
                stage="processing_audio",
                pct=90,
                status="Post-processing audio output…",
                log_text=log_text,
            )

            if hasattr(outputs, 'speech_outputs') and outputs.speech_outputs[0] is not None:
                audio_tensor = outputs.speech_outputs[0]
                audio = audio_tensor.cpu().float().numpy()
            else:
                raise RuntimeError("Error: No audio was generated by the model.")

            if audio.ndim > 1:
                audio = audio.squeeze()

            sample_rate = 24000
            total_duration = len(audio) / sample_rate
            log_lines.append(f"Audio duration: {total_duration:.2f} seconds")
            log_lines.append("Complete!")
            log_text = "\n".join(log_lines)

            # Final yield with both audio and complete log
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
