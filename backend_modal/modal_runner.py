import os
import time
import numpy as np
import librosa
import soundfile as sf
import torch
from datetime import datetime

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
        "huggingface_hub",
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


@app.cls(gpu="T4", scaledown_window=300, secrets=[modal.Secret.from_name("hf-secret")])
class VibeVoiceModel:
    def __init__(self):
        self.model_paths = {
            "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
            "VibeVoice-7B": "vibevoice/VibeVoice-7B",
        }
        self.device = "cuda"
        self.inference_steps = 5

    @modal.enter()
    def load_models(self):
        """
        This method is run once when the container starts.
        It downloads and loads all models onto the GPU.
        """
        # Set up HuggingFace authentication
        import os
        from huggingface_hub import login
        
        # Get the HF token from Modal secrets
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            print("Logged in to HuggingFace Hub")
        else:
            print("Warning: No HF_TOKEN found in secrets")
        
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
        
        for name, path in self.model_paths.items():
            print(f" - Loading {name} from {path}")
            proc = VibeVoiceProcessor.from_pretrained(path)
            mdl = VibeVoiceForConditionalGenerationInference.from_pretrained(
                path, 
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa"
            ).to(self.device)
            mdl.eval()
            print(f"  {name} loaded to {self.device}")
            self.processors[name] = proc
            self.models[name] = mdl
        
        self.setup_voice_presets()
        print("Model loading complete.")

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
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")

            model = self.models[model_name]
            processor = self.processors[model_name]
            model.set_ddpm_inference_steps(num_steps=self.inference_steps)

            print(f"Generating with model {model_name} on {self.device}")

            if not script.strip():
                raise ValueError("Error: Please provide a script.")

            script = script.replace("’", "'")

            if not 1 <= num_speakers <= 4:
                raise ValueError("Error: Number of speakers must be between 1 and 4.")

            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            for i, speaker_name in enumerate(selected_speakers):
                if not speaker_name or speaker_name not in self.available_voices:
                    raise ValueError(f"Error: Please select a valid speaker for Speaker {i+1}.")

            log = f"Generating conference with {num_speakers} speakers\n"
            log += f"Model: {model_name}\n"
            log += f"Parameters: CFG Scale={cfg_scale}\n"
            log += f"Speakers: {', '.join(selected_speakers)}\n"

            voice_samples = []
            for speaker_name in selected_speakers:
                audio_path = self.available_voices[speaker_name]
                audio_data = self.read_audio(audio_path)
                if len(audio_data) == 0:
                    raise ValueError(f"Error: Failed to load audio for {speaker_name}")
                voice_samples.append(audio_data)

            log += f"Loaded {len(voice_samples)} voice samples\n"

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
            log += f"Formatted script with {len(formatted_script_lines)} turns\n"
            log += "Processing with VibeVoice...\n"

            inputs = processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            start_time = time.time()
            
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=processor.tokenizer,
                    generation_config={'do_sample': False},
                    verbose=False,
                )
            generation_time = time.time() - start_time

            if hasattr(outputs, 'speech_outputs') and outputs.speech_outputs[0] is not None:
                audio_tensor = outputs.speech_outputs[0]
                audio = audio_tensor.cpu().float().numpy()
            else:
                raise RuntimeError("Error: No audio was generated by the model.")

            if audio.ndim > 1:
                audio = audio.squeeze()

            sample_rate = 24000
            total_duration = len(audio) / sample_rate
            log += f"Generation completed in {generation_time:.2f} seconds\n"
            log += f"Final audio duration: {total_duration:.2f} seconds\n"

            # Return the raw audio data and sample rate, Gradio will handle the rest
            return (sample_rate, audio), log

        except Exception as e:
            import traceback
            error_msg = f"An unexpected error occurred on Modal: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            # Return a special value or raise an exception that the client can handle
            # For Gradio, returning a log message is often best.
            return None, error_msg