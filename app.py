import os
import gradio as gr
import modal
import traceback

# --- Configuration ---
# This is the name of your Modal stub.
MODAL_STUB_NAME = "vibevoice-generator"
MODAL_CLASS_NAME = "VibeVoiceModel" # Extract class name
MODAL_METHOD_NAME = "generate_podcast" # Extract method name

# These lists are now hardcoded because the data lives on the Modal container.
# For a more dynamic app, you could create a small Modal function to fetch these lists.
AVAILABLE_MODELS = ["VibeVoice-1.5B", "VibeVoice-7B"]
AVAILABLE_VOICES = [
    "en-Alice_woman_bgm", "en-Alice_woman", "en-Carter_man", "en-Frank_man",
    "en-Maya_woman", "en-Yasser_man", "in-Samuel_man", "zh-Anchen_man_bgm",
    "zh-Bowen_man", "zh-Xinran_woman"
]
DEFAULT_SPEAKERS = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']

# Male and female voice categories for smart speaker selection
MALE_VOICES = [
    "en-Carter_man",
    "en-Frank_man",
    "en-Yasser_man",
    "in-Samuel_man",
    "zh-Anchen_man_bgm",
    "zh-Bowen_man"
]
FEMALE_VOICES = [
    "en-Alice_woman_bgm",
    "en-Alice_woman",
    "en-Maya_woman",
    "zh-Xinran_woman"
]

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

# Gender mapping for each script's speakers
SCRIPT_SPEAKER_GENDERS = [
    ["female"],  # AI TED Talk - Rachel
    ["neutral"],  # Political Speech - generic speaker
    ["male", "female"],  # Finance IPO - James, Patricia
    ["female", "male"],  # Telehealth - Jennifer, Tom
    ["female", "male", "female"],  # Military - Sarah, David, Lisa
    ["male", "female", "male"],  # Oil - Robert, Lisa, Michael
    ["male", "female", "male", "male"],  # Game Creation - Alex, Sarah, Marcus, Emma
    ["female", "male", "female", "male"]  # Product Meeting - Sarah, Marcus, Jennifer, David
]

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
        gr.Markdown("## NOTE: The Large model takes significant generation time with limited increase in quality. I recommend trying 1.5B first.")
        
        with gr.Tabs():
            with gr.Tab("Generate"):
                gr.Markdown("### Generated Conference")
                complete_audio_output = gr.Audio(
                    label="Complete Conference (Download)",
                    type="numpy",
                    autoplay=False,
                    show_download_button=True,
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Conference Settings")
                        model_dropdown = gr.Dropdown(
                            choices=AVAILABLE_MODELS,
                            value=AVAILABLE_MODELS[0],
                            label="Model",
                        )
                        num_speakers = gr.Slider(
                            minimum=1, maximum=4, value=2, step=1,
                            label="Number of Speakers",
                        )

                        gr.Markdown("### Speaker Selection")
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
                        gr.Markdown("### Script Input")
                        script_input = gr.Textbox(
                            label="Conversation Script",
                            placeholder="Enter your conference script here...",
                            lines=12,
                            max_lines=20,
                        )
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Example Scripts")
                                with gr.Row():
                                    use_natural = gr.Checkbox(
                                        value=True,
                                        label="Natural talking sounds",
                                        scale=1
                                    )
                                    duration_display = gr.Textbox(
                                        value="",
                                        label="Est. Duration",
                                        interactive=False,
                                        scale=1
                                    )
                        
                        example_names = [
                            "AI TED Talk",
                            "Political Speech",
                            "Finance IPO Meeting",
                            "Telehealth Meeting",
                            "Military Meeting",
                            "Oil Meeting",
                            "Game Creation Meeting",
                            "Product Meeting"
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
                            "🚀 Generate Conference (on Modal)", size="lg",
                            variant="primary",
                        )
                        log_output = gr.Textbox(
                            label="Generation Log",
                            lines=8, max_lines=15,
                            interactive=False,
                        )
                        with gr.Row():
                            status_display = gr.Markdown(
                                value="Status: idle.",
                                elem_id="status-display",
                            )
                            progress_slider = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=0,
                                step=1,
                                label="Progress",
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
                
                def smart_speaker_selection(gender_list):
                    """Select speakers based on gender requirements."""
                    selected = []
                    for gender in gender_list:
                        if gender == "male" and MALE_VOICES:
                            available = [v for v in MALE_VOICES if v not in selected]
                            if available:
                                selected.append(available[0])
                            else:
                                selected.append(MALE_VOICES[0])
                        elif gender == "female" and FEMALE_VOICES:
                            available = [v for v in FEMALE_VOICES if v not in selected]
                            if available:
                                selected.append(available[0])
                            else:
                                selected.append(FEMALE_VOICES[0])
                        else:
                            # neutral or fallback
                            available = [v for v in AVAILABLE_VOICES if v not in selected]
                            if available:
                                selected.append(available[0])
                            else:
                                selected.append(AVAILABLE_VOICES[0])
                    return selected
                
                def load_specific_example(idx, natural):
                    """Load a specific example script."""
                    if idx >= len(EXAMPLE_SCRIPTS):
                        return [2, "", ""] + [None, None, None, None]
                    
                    script = EXAMPLE_SCRIPTS_NATURAL[idx] if natural else EXAMPLE_SCRIPTS[idx]
                    genders = SCRIPT_SPEAKER_GENDERS[idx] if idx < len(SCRIPT_SPEAKER_GENDERS) else ["neutral"]
                    speakers = smart_speaker_selection(genders)
                    duration = estimate_duration(script)
                    
                    # Pad speakers to 4
                    while len(speakers) < 4:
                        speakers.append(None)
                    
                    return [len(genders), script, duration] + speakers[:4]
                
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
                        error_message = "ERROR: Modal function not deployed. Please contact the space owner."
                        yield None, error_message, "Status: error.", gr.update(value=0)
                        return

                    # Show a message that we are calling the remote function
                    yield (
                        None,
                        "🔄 Calling remote GPU on Modal.com... this may take a moment to start.",
                        "**Connecting**\nRequesting GPU resources…",
                        gr.update(value=0),
                    )

                    try:
                        speakers = speakers_and_params[:4]
                        cfg_scale_val = speakers_and_params[4]
                        current_log = ""
                        last_pct = 0
                        last_status = "**Connecting**\nRequesting GPU resources…"

                        # Stream updates from the Modal function
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
                                stage_label = update.get("stage", "").replace("_", " ").title() or "Status"
                                status_line = update.get("status") or "Processing…"
                                current_log = update.get("log", current_log)

                                status_formatted = f"**{stage_label}**\n{status_line}"
                                audio_output = audio_payload if audio_payload is not None else gr.update()

                                last_pct = progress_pct
                                last_status = status_formatted

                                yield (
                                    audio_output,
                                    current_log,
                                    status_formatted,
                                    gr.update(value=progress_pct),
                                )
                            else:
                                # Backwards compatibility: older backend returns (audio, log)
                                audio_payload, log_text = update if isinstance(update, (tuple, list)) else (None, str(update))
                                if log_text:
                                    current_log = log_text
                                audio_output = audio_payload if audio_payload is not None else gr.update()
                                yield (
                                    audio_output,
                                    current_log,
                                    last_status,
                                    gr.update(value=last_pct),
                                )
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"Error calling Modal: {e}")
                        error_log = f"❌ An error occurred: {e}\n\n{tb}"
                        yield (
                            None,
                            error_log,
                            "**Error**\nInference failed.",
                            gr.update(value=0),
                        )

                generate_btn.click(
                    fn=generate_podcast_wrapper,
                    inputs=[model_dropdown, num_speakers, script_input] + speaker_selections + [cfg_scale],
                    outputs=[complete_audio_output, log_output, status_display, progress_slider]
                )
            
            with gr.Tab("Architecture"):
                with gr.Row():
                    gr.Markdown("""VibeVoice is a novel framework designed for generating expressive, long-form, multi-speaker conversational audio, 
                    such as conferences, from text. It addresses significant challenges in traditional Text-to-Speech (TTS) systems, particularly 
                    in scalability, speaker consistency, and natural turn-taking. A core innovation of VibeVoice is its use of continuous 
                    speech tokenizers (Acoustic and Semantic) operating at an ultra-low frame rate of 7.5 Hz. These tokenizers efficiently 
                    preserve audio fidelity while significantly boosting computational efficiency for processing long sequences. VibeVoice 
                    employs a next-token diffusion framework, leveraging a Large Language Model (LLM) to understand textual context and 
                    dialogue flow, and a diffusion head to generate high-fidelity acoustic details. The model can synthesize speech up to 
                    90 minutes long with up to 4 distinct speakers, surpassing the typical 1-2 speaker limits of many prior models.""")
                with gr.Row():    
                    with gr.Column():
                        gr.Markdown("## VibeVoice: A Frontier Open-Source Text-to-Speech Model")

                        gr.Markdown("""
                        ### Overview
                        
                        VibeVoice is a novel framework designed for generating expressive, long-form, multi-speaker conversational audio, 
                        such as conferences, from text. It addresses significant challenges in traditional Text-to-Speech (TTS) systems, 
                        particularly in scalability, speaker consistency, and natural turn-taking.
                        
                        ### Key Features
                        
                        - **Multi-Speaker Support**: Handles up to 4 distinct speakers
                        - **Long-Form Generation**: Synthesizes speech up to 90 minutes
                        - **Natural Conversation Flow**: Includes turn-taking and interruptions
                        - **Ultra-Low Frame Rate**: 7.5 Hz tokenizers for efficiency
                        - **High Fidelity**: Preserves acoustic details while being computationally efficient
                        
                        ### Technical Architecture
                        
                        1. **Continuous Speech Tokenizers**: Acoustic and Semantic tokenizers at 7.5 Hz
                        2. **Next-Token Diffusion Framework**: Combines LLM understanding with diffusion generation
                        3. **Large Language Model**: Understands context and dialogue flow
                        4. **Diffusion Head**: Generates high-fidelity acoustic details
                        """)
                        
                    with gr.Column():
                        gr.HTML("""
                        <div style="width: 100%; padding: 20px;">
                            <img src="https://huggingface.co/spaces/ACloudCenter/Conference-Generator-VibeVoice/resolve/main/public/images/diagram.jpg" 
                                style="width: 100%; height: auto; border-radius: 10px; box-shadow: 0 5px 20px rgba(0,0,0,0.15);"
                                alt="VibeVoice Architecture Diagram">
                        </div>
                        """)
                        
                        gr.Markdown("""
                        ### Model Variants
                        
                        **VibeVoice-1.5B**: Faster inference, suitable for real-time applications
                        **VibeVoice-7B**: Higher quality output, recommended for production use
                        
                        ### Performance Metrics
                        
                        <img src="https://huggingface.co/spaces/ACloudCenter/Conference-Generator-VibeVoice/resolve/main/public/images/chart.png" 
                            style="width: 100%; height: auto; border-radius: 10px; margin-top: 20px;"
                            alt="Performance Comparison">
                        """)
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
