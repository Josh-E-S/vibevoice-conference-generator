import os
import gradio as gr
import modal
import traceback

# --- Configuration ---
# This is the name of your Modal stub.
MODAL_STUB_NAME = "vibevoice-generator"
# This is the name of the remote class and method to call.
MODAL_FUNCTION_NAME = "VibeVoiceModel.generate_podcast"

# These lists are now hardcoded because the data lives on the Modal container.
# For a more dynamic app, you could create a small Modal function to fetch these lists.
AVAILABLE_MODELS = ["VibeVoice-1.5B", "VibeVoice-7B"]
AVAILABLE_VOICES = [
    "en-Alice_woman_bgm", "en-Alice_woman", "en-Carter_man", "en-Frank_man",
    "en-Maya_woman", "en-Yasser_man", "in-Samuel_man", "zh-Anchen_man_bgm",
    "zh-Bowen_man", "zh-Xinran_woman"
]
DEFAULT_SPEAKERS = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']

# --- Modal Connection ---
try:
    # This looks up the remote function on Modal
    # It will raise an error if the app isn't deployed (`modal deploy modal_runner.py`)
    remote_generate_function = modal.Function.lookup(MODAL_STUB_NAME, MODAL_FUNCTION_NAME)
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
        gr.Markdown("## GPU processing is now offloaded to a Modal.com backend!")
        
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
                        generate_btn = gr.Button(
                            "🚀 Generate Conference (on Modal)", size="lg",
                            variant="primary",
                        )
                        log_output = gr.Textbox(
                            label="Generation Log",
                            lines=8, max_lines=15,
                            interactive=False,
                        )

                def update_speaker_visibility(num_speakers):
                    return [gr.update(visible=(i < num_speakers)) for i in range(4)]

                num_speakers.change(
                    fn=update_speaker_visibility,
                    inputs=[num_speakers],
                    outputs=speaker_selections
                )

                def generate_podcast_wrapper(model_choice, num_speakers_val, script, *speakers_and_params):
                    if remote_generate_function is None:
                        return None, "ERROR: Modal function not deployed. Please contact the space owner."
                    
                    # Show a message that we are calling the remote function
                    yield None, "🔄 Calling remote GPU on Modal.com... this may take a moment to start."

                    try:
                        speakers = speakers_and_params[:4]
                        cfg_scale_val = speakers_and_params[4]
                        
                        # This is the call to the remote Modal function
                        result, log = remote_generate_function.remote(
                            num_speakers=int(num_speakers_val),
                            script=script,
                            speaker_1=speakers[0],
                            speaker_2=speakers[1],
                            speaker_3=speakers[2],
                            speaker_4=speakers[3],
                            cfg_scale=cfg_scale_val,
                            model_name=model_choice
                        )
                        yield result, log
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"Error calling Modal: {e}")
                        yield None, f"An error occurred in the Gradio wrapper: {e}\n\n{tb}"

                generate_btn.click(
                    fn=generate_podcast_wrapper,
                    inputs=[model_dropdown, num_speakers, script_input] + speaker_selections + [cfg_scale],
                    outputs=[complete_audio_output, log_output]
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