import os
import re
import gradio as gr
import modal
import traceback
from huggingface_hub import InferenceClient

# --- Configuration ---
MODAL_STUB_NAME = "vibevoice-generator"
MODAL_CLASS_NAME = "VibeVoiceModel"

AVAILABLE_MODELS = ["VibeVoice-1.5B", "VibeVoice-7B"]
AVAILABLE_VOICES = ["Cherry", "Chicago", "Janus", "Mantis", "Sponge", "Starchild"]
DEFAULT_SPEAKERS = ["Cherry", "Chicago", "Janus", "Mantis"]

SCRIPT_GEN_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
SCRIPT_MAX_WORDS = 1000

# --- Load example scripts ---
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
        "4p_product_meeting.txt",
    ]

    for txt_file in original_files:
        file_path = os.path.join(examples_dir, txt_file)
        natural_file = txt_file.replace(".txt", "_natural.txt")
        natural_path = os.path.join(examples_dir, natural_file)

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                example_scripts.append(f.read())
        else:
            example_scripts.append("")

        if os.path.exists(natural_path):
            with open(natural_path, "r", encoding="utf-8") as f:
                example_scripts_natural.append(f.read())
        else:
            example_scripts_natural.append(
                example_scripts[-1] if example_scripts else ""
            )

    return example_scripts, example_scripts_natural


SCRIPT_SPEAKER_COUNTS = [1, 1, 2, 2, 3, 3, 4, 4]
EXAMPLE_SCRIPTS, EXAMPLE_SCRIPTS_NATURAL = load_example_scripts()

# --- Script parsing helpers ---

def parse_script_to_turns(script_text: str) -> list[dict]:
    """Parse a 'Speaker N: text' script into a list of turn dicts."""
    turns = []
    if not script_text or not script_text.strip():
        return turns

    pattern = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.+)", re.IGNORECASE)
    current_speaker = None
    current_text = []

    for line in script_text.strip().split("\n"):
        m = pattern.match(line.strip())
        if m:
            if current_speaker is not None:
                turns.append({"speaker": current_speaker, "text": " ".join(current_text).strip()})
            current_speaker = int(m.group(1))
            current_text = [m.group(2).strip()]
        elif line.strip():
            if current_speaker is not None:
                current_text.append(line.strip())
            else:
                # Line without a speaker tag — assign to Speaker 1
                current_speaker = 1
                current_text = [line.strip()]

    if current_speaker is not None and current_text:
        turns.append({"speaker": current_speaker, "text": " ".join(current_text).strip()})

    return turns


def turns_to_script(turns: list[dict]) -> str:
    """Convert turn dicts back to 'Speaker N: text' format."""
    lines = []
    for t in turns:
        if t.get("text", "").strip():
            lines.append(f"Speaker {t['speaker']}: {t['text'].strip()}")
    return "\n\n".join(lines)


def estimate_duration(turns: list[dict]) -> str:
    """Estimate audio duration from total word count."""
    total_words = sum(len(t.get("text", "").split()) for t in turns)
    if total_words == 0:
        return ""
    minutes = total_words / 150
    if minutes < 1:
        return f"~{int(minutes * 60)} seconds"
    return f"~{minutes:.1f} minutes"


# --- AI Script Generation ---

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("WARNING: HF_TOKEN not set. Script generation will fail.")
else:
    print(f"HF_TOKEN loaded ({len(hf_token)} chars)")
llm_client = InferenceClient(model=SCRIPT_GEN_MODEL, token=hf_token)

SCRIPT_SYSTEM_PROMPT = """You are a script writer. Write a realistic, engaging conversation script.

RULES:
- Use EXACTLY this format for every line: "Speaker N: dialogue text"
- N must be a number starting from 1
- Each speaker turn is its own paragraph separated by a blank line
- Write natural, flowing dialogue — not robotic or overly formal
- Include character names and context naturally in the dialogue
- Keep the total script under {max_words} words
- Use EXACTLY {num_speakers} speakers (Speaker 1 through Speaker {num_speakers})
- Do NOT include stage directions, parentheticals, or anything other than dialogue
- Output ONLY the script, no preamble or commentary"""


def generate_script_from_prompt(prompt: str, num_speakers: int) -> list[dict]:
    """Call the HF Inference API to generate a script from a prompt."""
    system = SCRIPT_SYSTEM_PROMPT.format(
        max_words=SCRIPT_MAX_WORDS, num_speakers=num_speakers
    )
    response = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.7,
    )
    raw = response.choices[0].message.content
    turns = parse_script_to_turns(raw)
    return turns


# --- Modal Connection ---
try:
    RemoteVibeVoiceModel = modal.Cls.from_name(MODAL_STUB_NAME, MODAL_CLASS_NAME)
    remote_model_instance = RemoteVibeVoiceModel()
    remote_generate_function = remote_model_instance.generate_podcast
    print("Successfully connected to Modal function.")
except modal.exception.NotFoundError:
    print("ERROR: Modal function not found.")
    print("Please deploy the Modal app first: modal deploy backend_modal/modal_runner.py")
    remote_generate_function = None

# --- Gradio UI ---
theme = gr.themes.Ocean(
    primary_hue="indigo",
    secondary_hue="fuchsia",
    neutral_hue="slate",
).set(button_large_radius="*radius_sm")

AUDIO_LABEL_DEFAULT = "Complete Conference (Download)"
PRIMARY_STAGE_MESSAGES = {
    "connecting": ("Request Submitted", "Provisioning GPU resources... cold starts can take up to a minute."),
    "queued": ("Waiting For GPU", "Worker is spinning up. Cold starts may take 30-60 seconds."),
    "loading_model": ("Loading Model", "Streaming VibeVoice weights to the GPU."),
    "loading_voices": ("Loading Voices", None),
    "preparing_inputs": ("Preparing Script", "Formatting the conversation for the model."),
    "generating_audio": ("Generating Audio", "Synthesizing speech — this is the longest step."),
    "processing_audio": ("Finalizing Audio", "Converting tensors into a playable waveform."),
    "complete": ("Ready", "Press play below or download your conference."),
    "error": ("Error", "Check the log for details."),
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
READY_PRIMARY_STATUS = "### Ready\nPress **Generate Conference** to run VibeVoice."


def build_primary_status(stage: str, status_line: str) -> str:
    title, default_desc = PRIMARY_STAGE_MESSAGES.get(stage, ("Working", "Processing..."))
    desc_parts = []
    if default_desc:
        desc_parts.append(default_desc)
    if status_line and status_line not in desc_parts:
        desc_parts.append(status_line)
    desc = "\n\n".join(desc_parts) if desc_parts else status_line
    return f"### {title}\n{desc}"


# --- Build Interface ---

def create_demo_interface():
    with gr.Blocks(
        title="VibeVoice - Conference Generator",
        theme=theme,
    ) as interface:
        # --- Banner ---
        gr.HTML("""
        <div style="width: 100%; margin-bottom: 20px;">
            <img src="https://huggingface.co/spaces/ACloudCenter/Conference-Generator-VibeVoice/resolve/main/public/images/banner.png"
                style="width: 100%; height: auto; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.2);"
                alt="VibeVoice Banner">
        </div>
        """)

        with gr.Tabs():
            # ==================== GENERATE TAB ====================
            with gr.Tab("Generate"):
                gr.Markdown("**Tip:** The 1.5B model is recommended — much faster with minimal quality difference.")

                # --- Conversation state: list of {speaker: int, text: str} ---
                turns_state = gr.State([])

                # --- Top row: Settings (left) + Script Tools (right) ---
                with gr.Row():
                    # ---------- LEFT COLUMN: Settings ----------
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
                                label=f"Speaker {i + 1}",
                                visible=(i < 2),
                            )
                            speaker_selections.append(speaker)

                        with gr.Accordion("Advanced Settings", open=False):
                            cfg_scale = gr.Slider(
                                minimum=1.0, maximum=2.0, value=1.3, step=0.05,
                                label="CFG Scale (Guidance Strength)",
                            )

                    # ---------- RIGHT COLUMN: Script creation ----------
                    with gr.Column(scale=2):
                        # --- AI Script Generator ---
                        with gr.Accordion("Generate a Script with AI", open=True):
                            gr.Markdown("Describe the conversation you want and AI will write the script for you.")
                            script_prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="e.g. A wizard consulting an orc about battle strategy for an upcoming siege",
                                lines=2,
                                max_lines=4,
                            )
                            with gr.Row():
                                generate_script_btn = gr.Button(
                                    "Generate Script", variant="secondary",
                                )
                                script_gen_status = gr.Markdown(value="")

                        # --- Example buttons ---
                        with gr.Accordion("Example Scripts", open=False):
                            with gr.Row():
                                use_natural = gr.Checkbox(
                                    value=True,
                                    label="Natural talking sounds",
                                )
                            example_names = [
                                "AI TED Talk", "Political Speech",
                                "Finance IPO Meeting", "Telehealth Meeting",
                                "Military Meeting", "Oil Meeting",
                                "Game Creation Meeting", "Product Meeting",
                            ]
                            example_buttons = []
                            with gr.Row():
                                for i in range(4):
                                    btn = gr.Button(example_names[i], size="sm", variant="secondary")
                                    example_buttons.append(btn)
                            with gr.Row():
                                for i in range(4, 8):
                                    btn = gr.Button(example_names[i], size="sm", variant="secondary")
                                    example_buttons.append(btn)

                # --- Conversation Editor ---
                gr.Markdown("### Conversation")
                duration_display = gr.Markdown(value="")

                @gr.render(inputs=[turns_state, num_speakers])
                def render_turns(turns, n_speakers):
                    if not turns:
                        gr.Markdown("*No script yet. Generate one with AI above, load an example, or add turns manually.*")
                    else:
                        speaker_choices = [f"Speaker {i + 1}" for i in range(int(n_speakers))]
                        for idx, turn in enumerate(turns):
                            with gr.Row(key=f"turn-{idx}"):
                                spk_dd = gr.Dropdown(
                                    choices=speaker_choices,
                                    value=f"Speaker {turn['speaker']}",
                                    label="",
                                    scale=1,
                                    min_width=120,
                                    container=False,
                                    key=f"spk-{idx}",
                                )
                                txt = gr.Textbox(
                                    value=turn["text"],
                                    label="",
                                    lines=2,
                                    max_lines=6,
                                    scale=5,
                                    container=False,
                                    key=f"txt-{idx}",
                                )
                                del_btn = gr.Button("X", size="sm", variant="stop", scale=0, min_width=40, key=f"del-{idx}")

                            # Update turn text when user edits
                            def on_text_change(new_text, current_turns, i=idx):
                                if i < len(current_turns):
                                    current_turns[i]["text"] = new_text
                                return current_turns

                            txt.change(
                                fn=on_text_change,
                                inputs=[txt, turns_state],
                                outputs=[turns_state],
                                queue=False,
                            )

                            # Update speaker when user changes dropdown
                            def on_speaker_change(new_spk, current_turns, i=idx):
                                if i < len(current_turns):
                                    num = int(new_spk.replace("Speaker ", ""))
                                    current_turns[i]["speaker"] = num
                                return current_turns

                            spk_dd.change(
                                fn=on_speaker_change,
                                inputs=[spk_dd, turns_state],
                                outputs=[turns_state],
                                queue=False,
                            )

                            # Delete turn
                            def on_delete(current_turns, i=idx):
                                if i < len(current_turns):
                                    current_turns.pop(i)
                                return current_turns

                            del_btn.click(
                                fn=on_delete,
                                inputs=[turns_state],
                                outputs=[turns_state],
                            )

                with gr.Row():
                    add_turn_btn = gr.Button("+ Add Turn", size="sm", variant="secondary")

                # --- Generate Conference ---
                generate_btn = gr.Button(
                    "Generate Conference", size="lg", variant="primary",
                )

                # --- Output section ---
                primary_status = gr.Markdown(
                    value=READY_PRIMARY_STATUS,
                    elem_id="primary-status",
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

                # ==================== EVENT HANDLERS ====================

                def update_speaker_visibility(n):
                    return [gr.update(visible=(i < n)) for i in range(4)]

                num_speakers.change(
                    fn=update_speaker_visibility,
                    inputs=[num_speakers],
                    outputs=speaker_selections,
                )

                # --- Add turn ---
                def add_turn(turns, n_speakers):
                    if not turns:
                        next_speaker = 1
                    else:
                        last = turns[-1]["speaker"]
                        next_speaker = (last % int(n_speakers)) + 1
                    turns.append({"speaker": next_speaker, "text": ""})
                    return turns, estimate_duration(turns)

                add_turn_btn.click(
                    fn=add_turn,
                    inputs=[turns_state, num_speakers],
                    outputs=[turns_state, duration_display],
                )

                # --- Update duration whenever turns change ---
                def update_duration(turns):
                    return estimate_duration(turns)

                turns_state.change(
                    fn=update_duration,
                    inputs=[turns_state],
                    outputs=[duration_display],
                    queue=False,
                )

                # --- AI Script Generation ---
                def on_generate_script(prompt, n_speakers):
                    if not prompt or not prompt.strip():
                        gr.Warning("Please enter a prompt describing the conversation.")
                        yield gr.update(), gr.update(), ""
                        return

                    yield gr.update(), gr.update(), "*Generating script...*"

                    try:
                        turns = generate_script_from_prompt(prompt.strip(), int(n_speakers))
                        if not turns:
                            yield gr.update(), gr.update(), "No script returned. Try a more descriptive prompt."
                            return
                        yield turns, estimate_duration(turns), ""
                    except Exception as e:
                        print(f"Script generation error: {e}")
                        import traceback as tb
                        tb.print_exc()
                        msg = str(e)
                        if "api_key" in msg or "log in" in msg or "token" in msg.lower():
                            yield gr.update(), gr.update(), "HF_TOKEN secret not configured. Add it in Space Settings."
                        else:
                            yield gr.update(), gr.update(), f"Error: {msg}"

                generate_script_btn.click(
                    fn=on_generate_script,
                    inputs=[script_prompt, num_speakers],
                    outputs=[turns_state, duration_display, script_gen_status],
                )

                # --- Load example scripts ---
                def load_example(idx, natural):
                    if idx >= len(EXAMPLE_SCRIPTS):
                        return [], 2, "" , *[None] * 4

                    script = EXAMPLE_SCRIPTS_NATURAL[idx] if natural else EXAMPLE_SCRIPTS[idx]
                    num = SCRIPT_SPEAKER_COUNTS[idx] if idx < len(SCRIPT_SPEAKER_COUNTS) else 1
                    turns = parse_script_to_turns(script)

                    speakers = list(AVAILABLE_VOICES[:num])
                    while len(speakers) < 4:
                        speakers.append(None)

                    return turns, num, estimate_duration(turns), *speakers[:4]

                for idx, btn in enumerate(example_buttons):
                    btn.click(
                        fn=lambda nat, i=idx: load_example(i, nat),
                        inputs=[use_natural],
                        outputs=[turns_state, num_speakers, duration_display] + speaker_selections,
                        queue=False,
                    )

                # --- Generate Conference (audio) ---
                def generate_podcast_wrapper(
                    model_choice, num_speakers_val, turns, *speakers_and_params
                ):
                    if remote_generate_function is None:
                        yield (
                            build_primary_status("error", "Modal backend is offline."),
                            gr.update(label=AUDIO_STAGE_LABELS.get("error", AUDIO_LABEL_DEFAULT)),
                            "ERROR: Modal function not deployed. Please contact the space owner.",
                        )
                        return

                    # Assemble turns into script text
                    script = turns_to_script(turns)
                    if not script.strip():
                        yield (
                            build_primary_status("error", "No script to generate."),
                            gr.update(label=AUDIO_STAGE_LABELS.get("error", AUDIO_LABEL_DEFAULT)),
                            "Please add some dialogue before generating.",
                        )
                        return

                    yield (
                        build_primary_status("connecting", "Provisioning GPU resources... cold starts can take up to a minute."),
                        gr.update(label=AUDIO_STAGE_LABELS.get("connecting", AUDIO_LABEL_DEFAULT)),
                        "Calling remote GPU on Modal.com...",
                    )

                    try:
                        speakers = speakers_and_params[:4]
                        cfg_scale_val = speakers_and_params[4]
                        current_log = ""
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
                            model_name=model_choice,
                        ):
                            if not update:
                                continue

                            if isinstance(update, dict):
                                audio_payload = update.get("audio")
                                stage_key = update.get("stage", last_stage) or last_stage
                                status_line = update.get("status") or "Processing..."
                                current_log = update.get("log", current_log)

                                audio_label = AUDIO_STAGE_LABELS.get(stage_key)
                                if not audio_label:
                                    stage_label = stage_key.replace("_", " ").title()
                                    audio_label = f"Complete Conference ({stage_label.lower()})"
                                if stage_key == "complete":
                                    audio_label = AUDIO_LABEL_DEFAULT

                                audio_update = gr.update(label=audio_label)
                                if audio_payload is not None:
                                    audio_update = gr.update(value=audio_payload, label=AUDIO_LABEL_DEFAULT)

                                yield (
                                    build_primary_status(stage_key, status_line),
                                    audio_update,
                                    current_log,
                                )

                                last_audio_label = audio_label
                                last_stage = stage_key
                            else:
                                audio_payload, log_text = (
                                    update if isinstance(update, (tuple, list)) else (None, str(update))
                                )
                                if log_text:
                                    current_log = log_text

                                if audio_payload is not None:
                                    yield (
                                        build_primary_status("complete", "Conference ready to download."),
                                        gr.update(value=audio_payload, label=AUDIO_LABEL_DEFAULT),
                                        current_log,
                                    )
                                else:
                                    status_line = current_log.splitlines()[-1] if current_log else "Processing..."
                                    yield (
                                        build_primary_status("generating_audio", status_line),
                                        gr.update(label=AUDIO_STAGE_LABELS.get("generating_audio", last_audio_label)),
                                        current_log,
                                    )
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"Error calling Modal: {e}")
                        yield (
                            build_primary_status("error", "Inference failed."),
                            gr.update(label=AUDIO_STAGE_LABELS.get("error", AUDIO_LABEL_DEFAULT)),
                            f"An error occurred: {e}\n\n{tb}",
                        )

                generate_btn.click(
                    fn=generate_podcast_wrapper,
                    inputs=[model_dropdown, num_speakers, turns_state] + speaker_selections + [cfg_scale],
                    outputs=[primary_status, complete_audio_output, log_output],
                )

            # ==================== ARCHITECTURE TAB ====================
            with gr.Tab("Architecture"):
                gr.Markdown("## VibeVoice: A Frontier Open-Source Text-to-Speech Model")
                gr.Markdown(
                    """VibeVoice is a novel framework designed for generating expressive, long-form, multi-speaker
                conversational audio from text. It addresses challenges in traditional TTS systems — scalability, speaker
                consistency, and natural turn-taking — using continuous speech tokenizers at an ultra-low 7.5 Hz frame rate
                and a next-token diffusion framework. It can synthesize speech up to 90 minutes long with up to 4 distinct speakers."""
                )

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


# --- Main ---
if __name__ == "__main__":
    if remote_generate_function is None:
        with gr.Blocks(theme=theme) as interface:
            gr.Markdown("# Configuration Error")
            gr.Markdown(
                "The Gradio application cannot connect to the Modal backend. "
                "Please run `modal deploy backend_modal/modal_runner.py` and refresh."
            )
        interface.launch()
    else:
        interface = create_demo_interface()
        interface.queue().launch(show_error=True)
