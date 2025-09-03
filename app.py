import os
import time
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import torch
import traceback
import threading
from spaces import GPU
from datetime import datetime

from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from processor.vibevoice_processor import VibeVoiceProcessor
from modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed

logging.set_verbosity_info()
logger = logging.get_logger(__name__)



class VibeVoiceDemo:
    def __init__(self, model_paths: dict, device: str = "cuda", inference_steps: int = 5):
        """
        model_paths: dict like {"VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
                                "VibeVoice-1.1B": "microsoft/VibeVoice-1.1B"}
        """
        self.model_paths = model_paths
        self.device = device
        self.inference_steps = inference_steps

        self.is_generating = False

        # Multi-model holders
        self.models = {}        # name -> model
        self.processors = {}    # name -> processor
        self.current_model_name = None

        self.available_voices = {}

        self.load_models()          # load all on CPU
        self.setup_voice_presets()
        self.load_example_scripts()

    def load_models(self):
        print("Loading processors and models on CPU...")
        for name, path in self.model_paths.items():
            print(f" - {name} from {path}")
            proc = VibeVoiceProcessor.from_pretrained(path)
            mdl = VibeVoiceForConditionalGenerationInference.from_pretrained(
                path, torch_dtype=torch.bfloat16
            )
            # Keep on CPU initially
            self.processors[name] = proc
            self.models[name] = mdl
        # choose default
        self.current_model_name = next(iter(self.models))
        print(f"Default model is {self.current_model_name}")

    def _place_model(self, target_name: str):
        """
        Move the selected model to CUDA and push all others back to CPU.
        """
        for name, mdl in self.models.items():
            if name == target_name:
                self.models[name] = mdl.to(self.device)
            else:
                self.models[name] = mdl.to("cpu")
        self.current_model_name = target_name
        print(f"Model {target_name} is now on {self.device}. Others moved to CPU.")

    def setup_voice_presets(self):
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
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

    @GPU(duration=120)
    def generate_podcast(self,
                         num_speakers: int,
                         script: str,
                         speaker_1: str = None,
                         speaker_2: str = None,
                         speaker_3: str = None,
                         speaker_4: str = None,
                         cfg_scale: float = 1.3,
                         model_name: str = None):
        """
        Generates a podcast as a single audio file from a script and saves it.
        Non-streaming.
        """
        try:
            # pick model
            model_name = model_name or self.current_model_name
            if model_name not in self.models:
                raise gr.Error(f"Unknown model: {model_name}")

            # place models on devices
            self._place_model(model_name)
            model = self.models[model_name]
            processor = self.processors[model_name]

            print(f"Using model {model_name} on {self.device}")

            model.eval()
            model.set_ddpm_inference_steps(num_steps=self.inference_steps)

            self.is_generating = True

            if not script.strip():
                raise gr.Error("Error: Please provide a script.")

            script = script.replace("’", "'")

            if not 1 <= num_speakers <= 4:
                raise gr.Error("Error: Number of speakers must be between 1 and 4.")

            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            for i, speaker_name in enumerate(selected_speakers):
                if not speaker_name or speaker_name not in self.available_voices:
                    raise gr.Error(f"Error: Please select a valid speaker for Speaker {i+1}.")

            log = f"Generating podcast with {num_speakers} speakers\n"
            log += f"Model: {model_name}\n"
            log += f"Parameters: CFG Scale={cfg_scale}\n"
            log += f"Speakers: {', '.join(selected_speakers)}\n"

            voice_samples = []
            for speaker_name in selected_speakers:
                audio_path = self.available_voices[speaker_name]
                audio_data = self.read_audio(audio_path)
                if len(audio_data) == 0:
                    raise gr.Error(f"Error: Failed to load audio for {speaker_name}")
                voice_samples.append(audio_data)

            log += f"Loaded {len(voice_samples)} voice samples\n"

            lines = script.strip().split('\n')
            formatted_script_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
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
            )

            start_time = time.time()
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
                raise gr.Error("Error: No audio was generated by the model. Please try again.")

            if audio.ndim > 1:
                audio = audio.squeeze()

            sample_rate = 24000

            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(output_dir, f"podcast_{timestamp}.wav")
            sf.write(file_path, audio, sample_rate)
            print(f"Podcast saved to {file_path}")

            total_duration = len(audio) / sample_rate
            log += f"Generation completed in {generation_time:.2f} seconds\n"
            log += f"Final audio duration: {total_duration:.2f} seconds\n"
            log += f"Successfully saved podcast to: {file_path}\n"

            self.is_generating = False
            return (sample_rate, audio), log

        except gr.Error as e:
            self.is_generating = False
            error_msg = f"Input Error: {str(e)}"
            print(error_msg)
            return None, error_msg

        except Exception as e:
            self.is_generating = False
            error_msg = f"An unexpected error occurred: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return None, error_msg


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

    def load_example_scripts(self):
        examples_dir = os.path.join(os.path.dirname(__file__), "text_examples")
        self.example_scripts = []
        if not os.path.exists(examples_dir):
            return

        txt_files = sorted(
            [f for f in os.listdir(examples_dir) if f.lower().endswith('.txt')]
        )
        for txt_file in txt_files:
            try:
                with open(os.path.join(examples_dir, txt_file), 'r', encoding='utf-8') as f:
                    script_content = f.read().strip()
                if script_content:
                    num_speakers = self._infer_num_speakers_from_script(script_content)
                    self.example_scripts.append([num_speakers, script_content])
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")


def convert_to_16_bit_wav(data):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = np.array(data)
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    return (data * 32767).astype(np.int16)

# Set synthwave theme
theme = gr.themes.Ocean(
    primary_hue="indigo",
    secondary_hue="fuchsia",
    neutral_hue="slate",
).set(
    button_large_radius='*radius_sm'
)

def set_working_state(*components, transcript_box=None):
    """
    Disable all interactive components and show progress in transcript/log box.
    Usage: set_working_state(generate_btn, random_example_btn, transcript_box=log_output)
    """
    updates = [gr.update(interactive=False) for _ in components]
    if transcript_box is not None:
        updates.append(gr.update(value="Generating... please wait", interactive=False))
    return tuple(updates)

def set_idle_state(*components, transcript_box=None):
    """
    Re-enable all interactive components and transcript/log box.
    Usage: set_idle_state(generate_btn, random_example_btn, transcript_box=log_output)
    """
    updates = [gr.update(interactive=True) for _ in components]
    if transcript_box is not None:
        updates.append(gr.update(interactive=True))
    return tuple(updates)
    

def create_demo_interface(demo_instance: VibeVoiceDemo):
    custom_css = """ """

    with gr.Blocks(
        title="VibeVoice - Conference Generator",
        css=custom_css,
        theme=theme,
    ) as interface:

        # Simple image
        gr.HTML("""
        <div style="width: 100%; margin-bottom: 20px;">
            <img src="https://huggingface.co/spaces/ACloudCenter/Conference-Generator-VibeVoice/resolve/main/public/images/banner.png" 
                style="width: 100%; height: auto; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.2);"
                alt="Canary-Qwen Transcriber Banner">
        </div>
        """)
        gr.Markdown("# Upload audio, use a sample track, or record yourself then ask questions about the transcript.")
        gr.Markdown('''VibeVoice is a novel framework designed for generating expressive, long-form, multi-speaker conversational audio, 
                    such as podcasts, from text. It addresses significant challenges in traditional Text-to-Speech (TTS) systems, particularly 
                    in scalability, speaker consistency, and natural turn-taking. A core innovation of VibeVoice is its use of continuous 
                    speech tokenizers (Acoustic and Semantic) operating at an ultra-low frame rate of 7.5 Hz. These tokenizers efficiently 
                    preserve audio fidelity while significantly boosting computational efficiency for processing long sequences. VibeVoice 
                    employs a next-token diffusion framework, leveraging a Large Language Model (LLM) to understand textual context and 
                    dialogue flow,and a diffusion head to generate high-fidelity acoustic details. The model can synthesize speech up to 
                    90 minutes long with up to 4 distinct speakers, surpassing the typical 1-2 speaker limits of many prior models.''')
        
        with gr.Tabs():
            with gr.Tab("Generate"):
                gr.Markdown("### 🎵 Generated Podcast")
                complete_audio_output = gr.Audio(
                    label="Complete Podcast (Download)",
                    type="numpy",
                    elem_classes="audio-output complete-audio-section",
                    autoplay=False,
                    show_download_button=True,
                    visible=True
                )
                
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="settings-card"):
                        gr.Markdown("### Podcast Settings")

                        # NEW - model dropdown
                        model_dropdown = gr.Dropdown(
                            choices=list(demo_instance.models.keys()),
                            value=demo_instance.current_model_name,
                            label="Model",
                        )

                        num_speakers = gr.Slider(
                            minimum=1, maximum=4, value=2, step=1,
                            label="Number of Speakers",
                            elem_classes="slider-container"
                        )

                        gr.Markdown("### Speaker Selection")
                        available_speaker_names = list(demo_instance.available_voices.keys())
                        default_speakers = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']

                        speaker_selections = []
                        for i in range(4):
                            default_value = default_speakers[i] if i < len(default_speakers) else None
                            speaker = gr.Dropdown(
                                choices=available_speaker_names,
                                value=default_value,
                                label=f"Speaker {i+1}",
                                visible=(i < 2),
                                elem_classes="speaker-item"
                            )
                            speaker_selections.append(speaker)

                        gr.Markdown("### Advanced Settings")
                        with gr.Accordion("Generation Parameters", open=False):
                            cfg_scale = gr.Slider(
                                minimum=1.0, maximum=2.0, value=1.3, step=0.05,
                                label="CFG Scale (Guidance Strength)",
                                elem_classes="slider-container"
                            )

                    with gr.Column(scale=2, elem_classes="generation-card"):
                        gr.Markdown("### Script Input")
                        script_input = gr.Textbox(
                            label="Conversation Script (Estimated duration will appear here)",
                            placeholder="Enter your podcast script here...",
                            lines=12,
                            max_lines=20,
                            elem_classes="script-input"
                        )

                        with gr.Row():
                            random_example_btn = gr.Button(
                                "Random Example", size="lg",
                                variant="secondary", elem_classes="random-btn", scale=1
                            )
                            generate_btn = gr.Button(
                                "🚀 Generate Podcast", size="lg",
                                variant="primary", elem_classes="generate-btn", scale=2
                            )

                        gr.Markdown("### 📚 Example Scripts")
                        
                        example_scripts_dict = {
                            "Software Product Meeting": [2, "Speaker 0: Hey team, let's discuss the new feature roadmap for Q2.\nSpeaker 1: Great! I've been analyzing user feedback from last quarter.\nSpeaker 0: What are the top requested features?\nSpeaker 1: Users want better mobile integration and real-time collaboration tools.\nSpeaker 0: That aligns with our vision. Let's prioritize those for the next sprint."],
                            "Climate Change Podcast": [2, "Speaker 0: Welcome to Climate Conversations. Today we're discussing renewable energy transitions.\nSpeaker 1: Thanks for having me. This is such a critical topic for our future.\nSpeaker 0: Can you explain the current state of renewable adoption?\nSpeaker 1: Globally, we're seeing exponential growth in solar and wind capacity.\nSpeaker 0: What are the main challenges we still face?\nSpeaker 1: Energy storage and grid infrastructure are the key bottlenecks."],
                            "Tech News Brief": [1, "Speaker 0: Here's your daily tech update. AI continues to transform industries worldwide. Major companies are investing billions in generative AI research. Meanwhile, quantum computing reaches new milestones with error correction breakthroughs. In cybersecurity news, new encryption standards are being developed to prepare for quantum threats."],
                            "Educational Tutorial": [2, "Speaker 0: Today we'll learn about machine learning basics.\nSpeaker 1: I'm excited but nervous. Is it really that complex?\nSpeaker 0: Not at all! Let's start with a simple example.\nSpeaker 1: Okay, I'm ready to learn.\nSpeaker 0: Think of it like teaching a computer to recognize patterns."],
                            "Comedy Sketch": [3, "Speaker 0: Did you hear about the AI that tried to write jokes?\nSpeaker 1: Oh no, not another AI story!\nSpeaker 2: Wait, let me guess - the punchlines were too logical?\nSpeaker 0: Exactly! It kept explaining why things were funny instead of being funny.\nSpeaker 1: That's hilarious! Or should I explain why it's hilarious?\nSpeaker 2: Please don't, you'll turn into the AI!"],
                            "Interview Format": [2, "Speaker 0: Welcome to our show. Today we have a special guest, a leading expert in biotechnology.\nSpeaker 1: Thank you for having me. It's great to be here.\nSpeaker 0: Tell us about your latest research.\nSpeaker 1: We're developing new therapies using CRISPR technology.\nSpeaker 0: How will this impact patient care?\nSpeaker 1: We expect to see personalized treatments become much more accessible."]
                        }
                        
                        with gr.Row():
                            example_btn1 = gr.Button("Software Product Meeting", size="sm", variant="secondary")
                            example_btn2 = gr.Button("Climate Change Podcast", size="sm", variant="secondary")
                            example_btn3 = gr.Button("Tech News Brief", size="sm", variant="secondary")
                        
                        with gr.Row():
                            example_btn4 = gr.Button("Educational Tutorial", size="sm", variant="secondary")
                            example_btn5 = gr.Button("Comedy Sketch", size="sm", variant="secondary")
                            example_btn6 = gr.Button("Interview Format", size="sm", variant="secondary")
                        
                        log_output = gr.Textbox(
                            label="Generation Log",
                            lines=8, max_lines=15,
                            interactive=False,
                            elem_classes="log-output"
                        )

                def update_speaker_visibility(num_speakers):
                    return [gr.update(visible=(i < num_speakers)) for i in range(4)]

                num_speakers.change(
                    fn=update_speaker_visibility,
                    inputs=[num_speakers],
                    outputs=speaker_selections
                )
                
                def update_script_label(script_text):
                    if not script_text or script_text.strip() == "":
                        return gr.update(label="Conversation Script (Estimated duration will appear here)")
                    
                    words = script_text.split()
                    word_count = len(words)
                    wpm = 150
                    estimated_minutes = word_count / wpm
                    
                    if estimated_minutes < 1:
                        duration_str = f"{int(estimated_minutes * 60)} seconds"
                    else:
                        minutes = int(estimated_minutes)
                        seconds = int((estimated_minutes - minutes) * 60)
                        if seconds > 0:
                            duration_str = f"{minutes} min {seconds} sec"
                        else:
                            duration_str = f"{minutes} min"
                    
                    warning = ""
                    if estimated_minutes * 60 > 90:
                        warning = " ⚠️ May exceed ZeroGPU timeout"
                    
                    return gr.update(label=f"Conversation Script - {word_count} words, ~{duration_str}{warning}")
                
                script_input.change(
                    fn=update_script_label,
                    inputs=[script_input],
                    outputs=[script_input]
                )

                def generate_podcast_wrapper(model_choice, num_speakers, script, *speakers_and_params):
                    try:
                        speakers = speakers_and_params[:4]
                        cfg_scale_val = speakers_and_params[4]
                        audio, log = demo_instance.generate_podcast(
                            num_speakers=int(num_speakers),
                            script=script,
                            speaker_1=speakers[0],
                            speaker_2=speakers[1],
                            speaker_3=speakers[2],
                            speaker_4=speakers[3],
                            cfg_scale=cfg_scale_val,
                            model_name=model_choice
                        )
                        return audio, log
                    except Exception as e:
                        traceback.print_exc()
                        return None, f"Error: {str(e)}"

                def on_generate_start():
                    return gr.update(interactive=False), gr.update(interactive=False), gr.update(value="🔄 Initializing generation...\n⏳ This may take up to 2 minutes depending on script length...")
                
                def on_generate_complete(audio, log):
                    return gr.update(interactive=True), gr.update(interactive=True), audio, log
                
                generate_click = generate_btn.click(
                    fn=on_generate_start,
                    inputs=[],
                    outputs=[generate_btn, random_example_btn, log_output],
                    queue=False
                ).then(
                    fn=generate_podcast_wrapper,
                    inputs=[model_dropdown, num_speakers, script_input] + speaker_selections + [cfg_scale],
                    outputs=[complete_audio_output, log_output],
                    queue=True
                ).then(
                    fn=lambda: (gr.update(interactive=True), gr.update(interactive=True)),
                    inputs=[],
                    outputs=[generate_btn, random_example_btn],
                    queue=False
                )

                def load_random_example():
                    import random
                    example_name = random.choice(list(example_scripts_dict.keys()))
                    num_speakers_value, script_value = example_scripts_dict[example_name]
                    return num_speakers_value, script_value

                random_example_btn.click(
                    fn=load_random_example,
                    inputs=[],
                    outputs=[num_speakers, script_input],
                    queue=False
                )
                
                def load_specific_example(example_name):
                    if example_name in example_scripts_dict:
                        num_speakers_value, script_value = example_scripts_dict[example_name]
                        return num_speakers_value, script_value
                    return 2, ""
                
                example_btn1.click(
                    fn=lambda: load_specific_example("Software Product Meeting"),
                    inputs=[],
                    outputs=[num_speakers, script_input],
                    queue=False
                )
                
                example_btn2.click(
                    fn=lambda: load_specific_example("Climate Change Podcast"),
                    inputs=[],
                    outputs=[num_speakers, script_input],
                    queue=False
                )
                
                example_btn3.click(
                    fn=lambda: load_specific_example("Tech News Brief"),
                    inputs=[],
                    outputs=[num_speakers, script_input],
                    queue=False
                )
                
                example_btn4.click(
                    fn=lambda: load_specific_example("Educational Tutorial"),
                    inputs=[],
                    outputs=[num_speakers, script_input],
                    queue=False
                )
                
                example_btn5.click(
                    fn=lambda: load_specific_example("Comedy Sketch"),
                    inputs=[],
                    outputs=[num_speakers, script_input],
                    queue=False
                )
                
                example_btn6.click(
                    fn=lambda: load_specific_example("Interview Format"),
                    inputs=[],
                    outputs=[num_speakers, script_input],
                    queue=False
                )
            
            with gr.Tab("Architecture"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## VibeVoice: A Frontier Open-Source Text-to-Speech Model")

                        gr.Markdown("""
                        ### Overview
                        
                        VibeVoice is a novel framework designed for generating expressive, long-form, multi-speaker conversational audio, 
                        such as podcasts, from text. It addresses significant challenges in traditional Text-to-Speech (TTS) systems, 
                        particularly in scalability, speaker consistency, and natural turn-taking.
                        
                        ### Training Architecture
                        
                        **Transformer-based Large Language Model** integrated with specialized acoustic and semantic tokenizers and a diffusion-based decoding head.
                        
                        **Core Components:**
                        - **LLM**: Qwen2.5-1.5B for this release
                        - **Acoustic Tokenizer**: Based on a σ-VAE variant with mirror-symmetric encoder-decoder structure (~340M parameters each)
                          - 7 stages of modified Transformer blocks
                          - Achieves 3200x downsampling from 24kHz input
                        - **Semantic Tokenizer**: Encoder mirrors the Acoustic Tokenizer's architecture
                          - Trained with an ASR proxy task
                        - **Diffusion Head**: Lightweight module (4 layers, ~123M parameters)
                          - Conditioned on LLM hidden states
                          - Uses DDPM process with Classifier-Free Guidance
                        
                        ### Training Details
                        
                        **Context Length**: Trained with curriculum up to 65,536 tokens
                        
                        **Training Stages:**
                        1. **Tokenizer Pre-training**: Acoustic and Semantic tokenizers trained separately
                        2. **VibeVoice Training**: Frozen tokenizers, only LLM and diffusion head trained
                           - Curriculum learning: 4k → 16K → 32K → 64K tokens
                        
                        ### Model Variants
                        
                        | Model | Context Length | Generation Length | Parameters |
                        |-------|---------------|-------------------|------------|
                        | VibeVoice-0.5B-Streaming | - | - | Coming Soon |
                        | **VibeVoice-1.5B** | 64K | ~90 min | 2.7B |
                        | VibeVoice-Large | 32K | ~45 min | Available |
                        
                        ### Technical Specifications
                        - **Frame Rate**: Ultra-low 7.5 Hz for efficiency
                        - **Sample Rate**: 24kHz audio output
                        - **Max Duration**: Up to 90 minutes
                        - **Speaker Capacity**: 1-4 distinct speakers
                        - **Languages**: English and Chinese
                        
                        ### Key Innovations
                        - Continuous speech tokenizers at ultra-low frame rate
                        - Next-token diffusion framework
                        - Curriculum learning for long-form generation
                        - Multi-speaker consistency without explicit modeling
                        """)
                    
                    with gr.Column(scale=2):
                        gr.HTML("""
                        <div style="text-align: center;">
                            <div style="margin: 20px 0;">
                                <img src="https://huggingface.co/spaces/ACloudCenter/Conference-Generator-VibeVoice/resolve/main/public/images/diagram.jpg" 
                                    style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
                                    alt="VibeVoice Architecture Diagram">
                            </div>
                            <div style="margin: 20px 0;">
                                <img src="https://huggingface.co/spaces/ACloudCenter/Conference-Generator-VibeVoice/resolve/main/public/images/chart.png" 
                                    style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
                                    alt="VibeVoice Performance Chart">
                            </div>
                        </div>
                        """)

    return interface

def run_demo(
    model_paths: dict = None,
    device: str = "cuda",
    inference_steps: int = 5,
    share: bool = True,
):
    """
    model_paths default includes two entries. Replace paths as needed.
    """
    if model_paths is None:
        model_paths = {
            "VibeVoice-Large": "microsoft/VibeVoice-Large",
            "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B"
        }

    set_seed(42)
    demo_instance = VibeVoiceDemo(model_paths, device, inference_steps)
    interface = create_demo_interface(demo_instance)
    interface.queue().launch(
        share=share,
        server_name="0.0.0.0" if share else "127.0.0.1",
        show_error=True,
        show_api=False
    )



if __name__ == "__main__":
    run_demo()
