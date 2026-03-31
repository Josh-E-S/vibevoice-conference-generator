import os
import gradio as gr
import modal
import traceback

# --- Configuration ---
# This is the name of your Modal stub.
MODAL_STUB_NAME = "vibevoice-generator"
MODAL_CLASS_NAME = "VibeVoiceModel" # Extract class name
MODAL_METHOD_NAME = "generate_podcast" # Extract method name

AVAILABLE_MODELS = ["VibeVoice-1.5B", "VibeVoice-7B"]
AVAILABLE_VOICES = ["Cherry", "Chicago", "Janus", "Mantis", "Sponge", "Starchild"]
DEFAULT_SPEAKERS = ["Cherry", "Chicago", "Janus", "Mantis"]

# Load example scripts
def load_example_scripts():
    examples_dir = "text_examples"
    example_scripts = []
    example_scripts_natural = []
    
    if not os.path.exists(examples_dir):
        return example_scripts, example_scripts_natural
    
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
        file_path = os.path.join(examples_dir, txt_file)
        natural_file = txt_file.replace(".txt", "_natural.txt")
        natural_path = os.path.join(examples_dir, natural_file)
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                example_scripts.append(f.read())
        else:
            example_scripts.append("")
            
        if os.path.exists(natural_path):
            with open(natural_path, 'r', encoding='utf-8') as f:
                example_scripts_natural.append(f.read())
        else:
            example_scripts_natural.append(example_scripts[-1] if example_scripts else "")
    
    return example_scripts, example_scripts_natural

# Number of speakers per example script
SCRIPT_SPEAKER_COUNTS = [1, 1, 2, 2, 3, 3, 4, 4]

EXAMPLE_SCRIPTS, EXAMPLE_SCRIPTS_NATURAL = load_example_scripts()

# --- Modal Connection ---
try:
    # Look up the remote class
    RemoteVibeVoiceModel = modal.Cls.from_name(MODAL_STUB_NAME, MODAL_CLASS_NAME)
    # Create an instance of the remote class
    remote_model_instance = RemoteVibeVoiceModel()
    # Get the remote method
    remote_generate_function = remote_model_instance.generate_podcast
    print("Successfully connected to Modal function.")
except modal.exception.NotFoundError:
    print("ERROR: Modal function not found.")
    print(f"Please deploy the Modal app first by running: modal deploy modal_runner.py")
    remote_generate_function = None

# --- Gradio UI Definition ---
theme = gr.themes.Ocean(
    primary_hue="indigo",
    secondary_hue="fuchsia",
    neutral_hue="slate",
).set(
    button_large_radius='*radius_sm'
)
 
AUDIO_LABEL_DEFAULT = "Complete Conference (Download)"
PRIMARY_STAGE_MESSAGES = {
    "connecting": ("🚀 Request Submitted", "Provisioning GPU resources... cold starts can take up to a minute."),
    "queued": ("🚦 Waiting For GPU", "Worker is spinning up. Cold starts may take 30-60 seconds."),
    "loading_model": ("📦 Loading Model", "Streaming VibeVoice weights to the GPU."),
    "loading_voices": ("🎙️ Loading Voices", None),
    "preparing_inputs": ("📝 Preparing Script", "Formatting the conversation for the model."),
    "generating_audio": ("🎧 Generating Audio", "Synthesizing speech — this is the longest step."),
    "processing_audio": ("✨ Finalizing Audio", "Converting tensors into a playable waveform."),
    "complete": ("✅ Ready", "Press play below or download your conference."),
    "error": ("❌ Error", "Check the log for details."),
}
AUDIO_STAGE_LABELS = {
    "connecting": "Complete Conference (requesting GPU...)",
    "queued": "Complete Conference (GPU warming up...)",
    "loading_model": "Complete Conference (loading model...)",
    "loading_voices": "Complete Conference (loading voices...)",
    "preparing_inputs": "Complete Conference (preparing inputs...)",
    "generating_audio": "Complete Conference (generating audio...)",
    "processing_audio": "Complete Conference (finalizing audio...)",
    "error": "Complete Conference (error)",
}
READY_PRIMARY_STATUS = "### Ready\nPress **Generate** to run VibeVoice."


def build_primary_status(stage: str, status_line: str) -> str:
    title, default_desc = PRIMARY_STAGE_MESSAGES.get(stage, ("⚙️ Working", "Processing..."))
    desc_parts = []
    if default_desc:
        desc_parts.append(default_desc)
    if status_line and status_line not in desc_parts:
        desc_parts.append(status_line)
    desc = "\n\n".join(desc_parts) if desc_parts else status_line
    return f"### {title}\n{desc}"


def create_demo_interface():
    with gr.Blocks(
        title="VibeVoice - Conference Generator",
        theme=theme,
    ) as interface:
        gr.HTML("""
        <div style="width: 100%; margin-bottom: 20px;">
            <img src="https://huggingface.co/spaces/ACloudCenter/Conference-Generator-VibeVoice/resolve/main/public/images/banner.png" 
                style="width: 100%; height: auto; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.2);"
                alt="VibeVoice Banner">
        </div>
        """)
        with gr.Tabs():
            with gr.Tab("Generate"):
                gr.Markdown("**Tip:** The 1.5B model is recommended — it's much faster with minimal quality difference.")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Settings")
                        model_dropdown = gr.Dropdown(
                            choices=AVAILABLE_MODELS,
                            value=AVAILABLE_MODELS[0],
                            label="Model",
                        )
                        num_speakers = gr.Slider(
                            minimum=1, maximum=4, value=2, step=1,
                            label="Number of Speakers",
                        )

                        speaker_selections = []
                        for i in range(4):
                            speaker = gr.Dropdown(
                                choices=AVAILABLE_VOICES,
                                value=DEFAULT_SPEAKERS[i] if i < len(DEFAULT_SPEAKERS) else None,
                                label=f"Speaker {i+1}",
                                visible=(i < 2),
                            )
                            speaker_selections.append(speaker)

                        with gr.Accordion("Advanced Settings", open=False):
                            cfg_scale = gr.Slider(
                                minimum=1.0, maximum=2.0, value=1.3, step=0.05,
                                label="CFG Scale (Guidance Strength)",
                            )

                    with gr.Column(scale=2):
                        script_input = gr.Textbox(
                            label="Conversation Script",
                            placeholder="Enter your conference script here...\n\nFormat:\nSpeaker 1: Hello everyone...\nSpeaker 2: Thanks for having me...",
                            lines=12,
                            max_lines=20,
                        )

                        with gr.Row():
                            use_natural = gr.Checkbox(
                                value=True,
                                label="Natural talking sounds",
                                scale=1,
                            )
                            duration_display = gr.Textbox(
                                value="",
                                label="Est. Duration",
                                interactive=False,
                                scale=1,
                            )

                        example_names = [
                            "AI TED Talk",
                            "Political Speech",
                            "Finance IPO Meeting",
                            "Telehealth Meeting",
                            "Military Meeting",
                            "Oil Meeting",
                            "Game Creation Meeting",
                            "Product Meeting",
                        ]

                        example_buttons = []
                        with gr.Row():
                            for i in range(min(4, len(example_names))):
                                btn = gr.Button(example_names[i], size="sm", variant="secondary")
                                example_buttons.append(btn)

                        with gr.Row():
                            for i in range(4, min(8, len(example_names))):
                                btn = gr.Button(example_names[i], size="sm", variant="secondary")
                                example_buttons.append(btn)

                generate_btn = gr.Button(
                    "Generate Conference", size="lg",
                    variant="primary",
                )

                primary_status = gr.Markdown(
                    value=READY_PRIMARY_STATUS,
                    elem_id="primary-status",
                )
                progress_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Progress",
                    interactive=False,
                )
                complete_audio_output = gr.Audio(
                    label=AUDIO_LABEL_DEFAULT,
                    type="numpy",
                    autoplay=False,
                    show_download_button=True,
                )
                with gr.Accordion("Generation Log", open=False):
                    log_output = gr.Textbox(
                        label="Log",
                        lines=8, max_lines=15,
                        interactive=False,
                    )

                def update_speaker_visibility(num_speakers):
                    return [gr.update(visible=(i < num_speakers)) for i in range(4)]
                
                def estimate_duration(script):
                    """Estimate duration based on word count."""
                    if not script:
                        return ""
                    words = len(script.split())
                    # Approximate 150 words per minute for natural speech
                    minutes = words / 150
                    if minutes < 1:
                        return f"~{int(minutes * 60)} seconds"
                    else:
                        return f"~{minutes:.1f} minutes"
                
                def load_specific_example(idx, natural):
                    """Load a specific example script."""
                    if idx >= len(EXAMPLE_SCRIPTS):
                        return [2, "", ""] + [None, None, None, None]

                    script = EXAMPLE_SCRIPTS_NATURAL[idx] if natural else EXAMPLE_SCRIPTS[idx]
                    num = SCRIPT_SPEAKER_COUNTS[idx] if idx < len(SCRIPT_SPEAKER_COUNTS) else 1
                    speakers = AVAILABLE_VOICES[:num]
                    duration = estimate_duration(script)

                    # Pad speakers to 4
                    while len(speakers) < 4:
                        speakers.append(None)

                    return [num, script, duration] + speakers[:4]
                
                # Connect example buttons
                for idx, btn in enumerate(example_buttons):
                    btn.click(
                        fn=lambda nat, i=idx: load_specific_example(i, nat),
                        inputs=[use_natural],
                        outputs=[num_speakers, script_input, duration_display] + speaker_selections,
                        queue=False
                    )
                
                # Update duration when script changes
                script_input.change(
                    fn=estimate_duration,
                    inputs=[script_input],
                    outputs=[duration_display],
                    queue=False
                )

                num_speakers.change(
                    fn=update_speaker_visibility,
                    inputs=[num_speakers],
                    outputs=speaker_selections
                )

                def generate_podcast_wrapper(model_choice, num_speakers_val, script, *speakers_and_params):
                    if remote_generate_function is None:
                        yield (
                            build_primary_status("error", "Modal backend is offline."),
                            gr.update(value=0),
                            gr.update(label=AUDIO_STAGE_LABELS.get("error", AUDIO_LABEL_DEFAULT)),
                            "ERROR: Modal function not deployed. Please contact the space owner.",
                        )
                        return

                    yield (
                        build_primary_status("connecting", "Provisioning GPU resources... cold starts can take up to a minute."),
                        gr.update(value=1),
                        gr.update(label=AUDIO_STAGE_LABELS.get("connecting", AUDIO_LABEL_DEFAULT)),
                        "Calling remote GPU on Modal.com...",
                    )

                    try:
                        speakers = speakers_and_params[:4]
                        cfg_scale_val = speakers_and_params[4]
                        current_log = ""
                        last_pct = 1
                        last_audio_label = AUDIO_STAGE_LABELS.get("connecting", AUDIO_LABEL_DEFAULT)
                        last_stage = "connecting"

                        for update in remote_generate_function.remote_gen(
                            num_speakers=int(num_speakers_val),
                            script=script,
                            speaker_1=speakers[0],
                            speaker_2=speakers[1],
                            speaker_3=speakers[2],
                            speaker_4=speakers[3],
                            cfg_scale=cfg_scale_val,
                            model_name=model_choice
                        ):
                            if not update:
                                continue

                            if isinstance(update, dict):
                                audio_payload = update.get("audio")
                                progress_pct = update.get("pct", last_pct)
                                stage_key = update.get("stage", last_stage) or last_stage
                                status_line = update.get("status") or "Processing..."
                                current_log = update.get("log", current_log)
                                progress_value = max(0, min(100, int(round(progress_pct))))

                                audio_label = AUDIO_STAGE_LABELS.get(stage_key)
                                if not audio_label:
                                    stage_label = stage_key.replace("_", " ").title()
                                    audio_label = f"Complete Conference ({stage_label.lower()})"
                                if stage_key == "complete":
                                    audio_label = AUDIO_LABEL_DEFAULT
                                if stage_key == "error":
                                    progress_value = 0

                                audio_update = gr.update(label=audio_label)
                                if audio_payload is not None:
                                    audio_update = gr.update(value=audio_payload, label=AUDIO_LABEL_DEFAULT)

                                yield (
                                    build_primary_status(stage_key, status_line),
                                    gr.update(value=progress_value),
                                    audio_update,
                                    current_log,
                                )

                                last_pct = progress_value
                                last_audio_label = audio_label
                                last_stage = stage_key
                            else:
                                audio_payload, log_text = update if isinstance(update, (tuple, list)) else (None, str(update))
                                if log_text:
                                    current_log = log_text

                                if audio_payload is not None:
                                    audio_update = gr.update(value=audio_payload, label=AUDIO_LABEL_DEFAULT)
                                    yield (
                                        build_primary_status("complete", "Conference ready to download."),
                                        gr.update(value=100),
                                        audio_update,
                                        current_log,
                                    )
                                else:
                                    status_line = current_log.splitlines()[-1] if current_log else "Processing..."
                                    yield (
                                        build_primary_status("generating_audio", status_line),
                                        gr.update(value=max(last_pct, 70)),
                                        gr.update(label=AUDIO_STAGE_LABELS.get("generating_audio", last_audio_label)),
                                        current_log,
                                    )
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"Error calling Modal: {e}")
                        yield (
                            build_primary_status("error", "Inference failed."),
                            gr.update(value=0),
                            gr.update(label=AUDIO_STAGE_LABELS.get("error", AUDIO_LABEL_DEFAULT)),
                            f"An error occurred: {e}\n\n{tb}",
                        )

                generate_btn.click(
                    fn=generate_podcast_wrapper,
                    inputs=[model_dropdown, num_speakers, script_input] + speaker_selections + [cfg_scale],
                    outputs=[primary_status, progress_slider, complete_audio_output, log_output],
                )
            
            with gr.Tab("Architecture"):
                gr.Markdown("## VibeVoice: A Frontier Open-Source Text-to-Speech Model")
                gr.Markdown("""VibeVoice is a novel framework designed for generating expressive, long-form, multi-speaker
                conversational audio from text. It addresses challenges in traditional TTS systems — scalability, speaker
                consistency, and natural turn-taking — using continuous speech tokenizers at an ultra-low 7.5 Hz frame rate
                and a next-token diffusion framework. It can synthesize speech up to 90 minutes long with up to 4 distinct speakers.""")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
### Key Features

- **Multi-Speaker Support**: Up to 4 distinct speakers
- **Long-Form Generation**: Up to 90 minutes of speech
- **Natural Conversation Flow**: Turn-taking and interruptions
- **Ultra-Low Frame Rate**: 7.5 Hz tokenizers for efficiency
- **High Fidelity**: Preserves acoustic details while being computationally efficient

### Technical Architecture

1. **Continuous Speech Tokenizers**: Acoustic and Semantic tokenizers at 7.5 Hz
2. **Next-Token Diffusion Framework**: Combines LLM understanding with diffusion generation
3. **Large Language Model**: Understands context and dialogue flow
4. **Diffusion Head**: Generates high-fidelity acoustic details

### Model Variants

- **VibeVoice-1.5B**: Faster inference, suitable for real-time applications
- **VibeVoice-7B**: Higher quality output, recommended for production use
                        """)

                    with gr.Column():
                        gr.Image(
                            value="public/images/diagram.jpg",
                            label="Architecture Diagram",
                            show_download_button=False,
                        )
                        gr.Image(
                            value="public/images/chart.png",
                            label="Performance Comparison",
                            show_download_button=False,
                        )
    return interface

# --- Main Execution ---
if __name__ == "__main__":
    if remote_generate_function is None:
        # If Modal isn't set up, we can't launch the full app.
        # We'll show a simplified UI with an error message.
        with gr.Blocks(theme=theme) as interface:
            gr.Markdown("# ❌ Configuration Error")
            gr.Markdown(
                "The Gradio application cannot connect to the Modal backend. "
                "The Modal app has not been deployed yet. "
                "Please run `modal deploy modal_runner.py` in your terminal and then refresh this page."
            )
        interface.launch()
    else:
        # Launch the full Gradio interface
        interface = create_demo_interface()
        interface.queue().launch(show_error=True)
