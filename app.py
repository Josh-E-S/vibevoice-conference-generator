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
SCRIPT_MAX_WORDS = 1000           # AI generation cap
MAX_SCRIPT_WORDS = 1500           # Hard limit for audio generation (~10 min)
MAX_TURNS = 50                    # Max conversation turns


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
        return f"~{int(minutes * 60)}s"
    return f"~{minutes:.1f}m"


# --- AI Script Generation ---

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("WARNING: HF_TOKEN not set. Script generation will fail.")
else:
    print(f"HF_TOKEN loaded ({len(hf_token)} chars)")
llm_client = InferenceClient(model=SCRIPT_GEN_MODEL, token=hf_token)

SCRIPT_SYSTEM_PROMPT = """You are an expert script writer for spoken audio. Write a conversation that sounds like real people talking.

STYLE:
- Each speaker should talk for a FULL PARAGRAPH per turn — 3 to 8 sentences minimum
- Speakers share complete thoughts, explain their reasoning, give examples, and build arguments before the other person responds
- This is NOT a rapid-fire back-and-forth. It should feel like a real meeting, interview, or deep conversation where people take time to make their point
- Use natural speech patterns: filler words (um, uh, well, you know), false starts, self-corrections, and thinking pauses
- Speakers should reference what the other person said, react naturally, and build on previous points
- Include personality — people joke, digress slightly, use analogies, get passionate about topics

FORMAT RULES:
- Use EXACTLY this format: "Speaker N: dialogue text" where N starts at 1
- Each turn is separated by a blank line
- Choose the right number of speakers for the scenario (1 to 4 max)
- Keep the total script under {max_words} words
- Output ONLY the script — no stage directions, no commentary, no preamble"""


def generate_script_from_prompt(prompt: str) -> tuple[list[dict], int]:
    """Call the HF Inference API to generate a script. Returns (turns, num_speakers)."""
    system = SCRIPT_SYSTEM_PROMPT.format(max_words=SCRIPT_MAX_WORDS)
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
    # Enforce limits on AI output
    turns = turns[:MAX_TURNS]
    total_words = sum(len(t["text"].split()) for t in turns)
    while total_words > MAX_SCRIPT_WORDS and turns:
        turns.pop()
        total_words = sum(len(t["text"].split()) for t in turns)
    speaker_ids = {t["speaker"] for t in turns}
    num_speakers = max(min(len(speaker_ids), 4), 1) if speaker_ids else 1
    return turns, num_speakers


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


# --- Theme & CSS ---
theme = gr.themes.Ocean(
    primary_hue="indigo",
    secondary_hue="fuchsia",
    neutral_hue="slate",
).set(button_large_radius="*radius_sm")

SPEAKER_COLORS = ["#6366f1", "#ec4899", "#22c55e", "#f59e0b"]
SPEAKER_LABELS = ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4"]

CUSTOM_CSS = """
/* ---- Conversation scroll container ---- */
.conversation-scroll {
    max-height: 480px;
    overflow-y: auto;
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    padding: 10px;
    background: var(--background-fill-secondary);
}
.conversation-scroll::-webkit-scrollbar { width: 6px; }
.conversation-scroll::-webkit-scrollbar-thumb {
    background: var(--border-color-primary);
    border-radius: 3px;
}

/* ---- Speaker color bars ---- */
""" + "\n".join(f"""
.speaker-{i+1} {{
    border-left: 4px solid {c} !important;
    padding-left: 10px !important;
    margin-bottom: 6px !important;
    border-radius: 8px !important;
    background: {c}08 !important;
}}""" for i, c in enumerate(SPEAKER_COLORS)) + """

/* ---- Prompt bar ---- */
.prompt-bar {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 16px;
    background: var(--background-fill-secondary);
}

/* ---- Voice chips row ---- */
.voice-row {
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    padding: 12px 16px;
    background: var(--background-fill-secondary);
}

/* ---- Example pill buttons ---- */
.example-btn button {
    border-radius: 20px !important;
    font-size: 0.85em !important;
}

/* ---- CTA buttons — consistent styling ---- */
.script-cta button, .generate-cta button {
    border-radius: 10px !important;
    letter-spacing: 0.02em;
    background: var(--button-primary-background-fill) !important;
    color: var(--button-primary-text-color) !important;
    border: none !important;
}
.script-cta button {
    font-size: 1.05em !important;
    padding: 12px 24px !important;
}
.generate-cta button {
    font-size: 1.15em !important;
    padding: 16px !important;
}

/* ---- Sticky empty state ---- */
.empty-state {
    text-align: center;
    padding: 48px 20px !important;
    opacity: 0.6;
}
"""


# --- Status helpers ---
AUDIO_LABEL_DEFAULT = "Generated Audio"
PRIMARY_STAGE_MESSAGES = {
    "connecting": ("Submitted", "Provisioning GPU resources... cold starts can take up to a minute."),
    "queued": ("Queued", "Worker is spinning up. Cold starts may take 30-60 seconds."),
    "loading_model": ("Loading Model", "Streaming VibeVoice weights to the GPU."),
    "loading_voices": ("Loading Voices", None),
    "preparing_inputs": ("Preparing", "Formatting the conversation for the model."),
    "generating_audio": ("Generating", "Synthesizing speech — this is the longest step."),
    "processing_audio": ("Finalizing", "Converting tensors into a playable waveform."),
    "complete": ("Complete", "Press play below or download your audio."),
    "error": ("Error", "Check the log for details."),
}
AUDIO_STAGE_LABELS = {
    "connecting": "Audio (requesting GPU...)",
    "queued": "Audio (GPU warming up...)",
    "loading_model": "Audio (loading model...)",
    "loading_voices": "Audio (loading voices...)",
    "preparing_inputs": "Audio (preparing...)",
    "generating_audio": "Audio (generating...)",
    "processing_audio": "Audio (finalizing...)",
    "error": "Audio (error)",
}


def build_primary_status(stage: str, status_line: str) -> str:
    title, default_desc = PRIMARY_STAGE_MESSAGES.get(stage, ("Working", "Processing..."))
    desc_parts = []
    if default_desc:
        desc_parts.append(default_desc)
    if status_line and status_line not in desc_parts:
        desc_parts.append(status_line)
    desc = "\n\n".join(desc_parts) if desc_parts else status_line
    return f"### {title}\n{desc}"


# ========================================================
# BUILD INTERFACE
# ========================================================

def create_demo_interface():
    with gr.Blocks(
        title="VibeVoice - Conference Generator",
        theme=theme,
        css=CUSTOM_CSS,
    ) as interface:

        # --- State ---
        turns_state = gr.State([])

        # ---- BANNER ----
        gr.HTML("""
        <div style="width:100%; margin-bottom:12px;">
            <img src="https://huggingface.co/spaces/ACloudCenter/Conference-Generator-VibeVoice/resolve/main/public/images/banner.png"
                 style="width:100%; height:auto; border-radius:14px; box-shadow:0 8px 32px rgba(0,0,0,0.25);"
                 alt="VibeVoice Banner">
        </div>
        """)

        with gr.Tabs():
            # ==================== GENERATE TAB ====================
            with gr.Tab("Generate"):

                # ---- PROMPT BAR ----
                gr.HTML("""
                <p style="margin:0 0 8px 0; opacity:0.7; font-size:0.95em;">
                    Describe any scenario — two people, a panel, a debate — and AI will write the full script.
                    Then review, edit, assign voices, and generate audio.
                </p>
                """)
                script_prompt = gr.Textbox(
                    label="Describe your conversation",
                    placeholder="A wizard and an orc debating battle strategy before a siege...",
                    lines=2,
                    max_lines=3,
                )
                with gr.Row():
                    generate_script_btn = gr.Button(
                        "Write Script with AI", variant="primary",
                        size="lg",
                        elem_classes="script-cta",
                    )
                    script_gen_status = gr.Textbox(
                        value="", label="", container=False,
                        interactive=False, scale=3,
                    )

                # ---- EXAMPLE PILLS ----
                example_names = [
                    "AI TED Talk", "Political Speech",
                    "Finance IPO", "Telehealth",
                    "Military Briefing", "Oil & Energy",
                    "Game Dev Meeting", "Product Review",
                ]
                with gr.Row():
                    gr.HTML("<span style='font-weight:600; white-space:nowrap; padding-top:8px;'>Examples:</span>")
                    example_buttons = []
                    for name in example_names:
                        btn = gr.Button(name, size="sm", variant="secondary",
                                        elem_classes="example-btn", min_width=80)
                        example_buttons.append(btn)

                # ---- CONVERSATION EDITOR ----
                with gr.Row():
                    gr.Markdown("### Script")
                    duration_display = gr.Markdown(value="")

                with gr.Column(elem_classes="conversation-scroll"):
                    @gr.render(inputs=[turns_state, gr.State(4)])
                    def render_turns(turns, _max):
                        if not turns:
                            gr.Markdown(
                                "Your conversation will appear here.\n\n"
                                "Type a prompt above and click **Write Script with AI**, "
                                "or pick an example to get started.",
                                elem_classes="empty-state",
                            )
                            return

                        # Detect how many speakers are in the current script
                        n_speakers = max(t["speaker"] for t in turns) if turns else 1
                        speaker_choices = [f"Speaker {i+1}" for i in range(max(n_speakers, 1))]

                        for idx, turn in enumerate(turns):
                            spk_num = turn["speaker"]
                            color_class = f"speaker-{spk_num}" if 1 <= spk_num <= 4 else "speaker-1"

                            with gr.Row(key=f"turn-{idx}", elem_classes=color_class):
                                spk_dd = gr.Dropdown(
                                    choices=speaker_choices,
                                    value=f"Speaker {spk_num}",
                                    label="",
                                    scale=1, min_width=115,
                                    container=False,
                                    key=f"spk-{idx}",
                                )
                                txt = gr.Textbox(
                                    value=turn["text"],
                                    label="",
                                    lines=2, max_lines=8,
                                    scale=6,
                                    container=False,
                                    key=f"txt-{idx}",
                                )
                                del_btn = gr.Button(
                                    "X", size="sm", variant="stop",
                                    scale=0, min_width=36, key=f"del-{idx}",
                                )

                            def on_text_change(new_text, current_turns, i=idx):
                                if i < len(current_turns):
                                    current_turns[i]["text"] = new_text
                                return current_turns

                            txt.change(fn=on_text_change, inputs=[txt, turns_state],
                                       outputs=[turns_state], queue=False)

                            def on_speaker_change(new_spk, current_turns, i=idx):
                                if i < len(current_turns):
                                    current_turns[i]["speaker"] = int(new_spk.replace("Speaker ", ""))
                                return current_turns

                            spk_dd.change(fn=on_speaker_change, inputs=[spk_dd, turns_state],
                                          outputs=[turns_state], queue=False)

                            def on_delete(current_turns, i=idx):
                                if i < len(current_turns):
                                    current_turns.pop(i)
                                return current_turns

                            del_btn.click(fn=on_delete, inputs=[turns_state],
                                          outputs=[turns_state])

                with gr.Row():
                    gr.HTML("<div></div>")  # spacer
                    add_turn_btn = gr.Button("+ Add Turn", size="sm", variant="secondary",
                                             scale=0, min_width=120)
                    gr.HTML("<div></div>")  # spacer

                # ---- VOICE & MODEL SETTINGS ----
                # Hidden slider that gets auto-set — not shown to user
                num_speakers = gr.Slider(
                    minimum=1, maximum=4, value=2, step=1,
                    visible=False,
                )

                with gr.Group(elem_classes="voice-row"):
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=AVAILABLE_MODELS,
                            value=AVAILABLE_MODELS[0],
                            label="Model",
                            scale=1,
                        )
                        speaker_selections = []
                        for i in range(4):
                            s = gr.Dropdown(
                                choices=AVAILABLE_VOICES,
                                value=DEFAULT_SPEAKERS[i] if i < len(DEFAULT_SPEAKERS) else None,
                                label=f"Voice {i+1}",
                                visible=(i < 2),
                                scale=1,
                            )
                            speaker_selections.append(s)
                    with gr.Row():
                        cfg_scale = gr.Slider(
                            minimum=1.0, maximum=2.0, value=1.3, step=0.05,
                            label="CFG Scale",
                            scale=3,
                        )

                # ---- GENERATE BUTTON ----
                generate_btn = gr.Button(
                    "Generate Conference Audio", size="lg", variant="primary",
                    elem_classes="generate-cta",
                )

                # ---- OUTPUT ----
                primary_status = gr.Markdown(value="", elem_id="primary-status")
                complete_audio_output = gr.Audio(
                    label=AUDIO_LABEL_DEFAULT,
                    type="numpy",
                    autoplay=False,
                    show_download_button=True,
                )
                with gr.Accordion("Generation Log", open=False):
                    log_output = gr.Textbox(
                        label="Log", lines=8, max_lines=15, interactive=False,
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
                def add_turn(turns):
                    if len(turns) >= MAX_TURNS:
                        gr.Warning(f"Maximum {MAX_TURNS} turns reached.")
                        return turns, estimate_duration(turns)
                    if not turns:
                        next_speaker = 1
                    else:
                        max_spk = max(t["speaker"] for t in turns)
                        last = turns[-1]["speaker"]
                        next_speaker = (last % max_spk) + 1
                    turns.append({"speaker": next_speaker, "text": ""})
                    return turns, estimate_duration(turns)

                add_turn_btn.click(
                    fn=add_turn,
                    inputs=[turns_state],
                    outputs=[turns_state, duration_display],
                )

                turns_state.change(
                    fn=lambda turns: estimate_duration(turns),
                    inputs=[turns_state],
                    outputs=[duration_display],
                    queue=False,
                )

                # --- AI Script Generation ---
                def _no_change(status_text):
                    return (gr.update(), gr.update(), status_text,
                            gr.update(), *[gr.update()] * 4)

                def on_generate_script(prompt):
                    if not prompt or not prompt.strip():
                        gr.Warning("Please enter a prompt.")
                        yield _no_change("")
                        return

                    yield _no_change("*Writing script...*")

                    try:
                        turns, detected = generate_script_from_prompt(prompt.strip())
                        if not turns:
                            yield _no_change("Empty result — try a more descriptive prompt.")
                            return

                        voices = list(AVAILABLE_VOICES[:detected])
                        while len(voices) < 4:
                            voices.append(None)

                        yield (turns, estimate_duration(turns), "",
                               detected, *voices[:4])
                    except Exception as e:
                        print(f"Script generation error: {e}")
                        traceback.print_exc()
                        msg = str(e)
                        if "api_key" in msg or "log in" in msg or "token" in msg.lower():
                            yield _no_change("HF_TOKEN not configured. Add it in Space Settings → Secrets.")
                        else:
                            yield _no_change(f"Error: {msg}")

                generate_script_btn.click(
                    fn=on_generate_script,
                    inputs=[script_prompt],
                    outputs=[turns_state, duration_display, script_gen_status,
                             num_speakers] + speaker_selections,
                )

                # --- Load examples (always use natural versions) ---
                def load_example(idx):
                    if idx >= len(EXAMPLE_SCRIPTS):
                        return [], 2, "", *[None] * 4

                    script = EXAMPLE_SCRIPTS_NATURAL[idx]
                    num = SCRIPT_SPEAKER_COUNTS[idx] if idx < len(SCRIPT_SPEAKER_COUNTS) else 1
                    turns = parse_script_to_turns(script)

                    voices = list(AVAILABLE_VOICES[:num])
                    while len(voices) < 4:
                        voices.append(None)

                    return turns, num, estimate_duration(turns), *voices[:4]

                for idx, btn in enumerate(example_buttons):
                    btn.click(
                        fn=lambda i=idx: load_example(i),
                        inputs=[],
                        outputs=[turns_state, num_speakers, duration_display] + speaker_selections,
                        queue=False,
                    )

                # --- Generate audio ---
                def generate_podcast_wrapper(
                    model_choice, num_speakers_val, turns, *speakers_and_params
                ):
                    if remote_generate_function is None:
                        yield (
                            build_primary_status("error", "Modal backend is offline."),
                            gr.update(label=AUDIO_STAGE_LABELS.get("error", AUDIO_LABEL_DEFAULT)),
                            "ERROR: Modal function not deployed.",
                        )
                        return

                    script = turns_to_script(turns)
                    if not script.strip():
                        yield (
                            build_primary_status("error", "No script to generate."),
                            gr.update(label=AUDIO_STAGE_LABELS.get("error", AUDIO_LABEL_DEFAULT)),
                            "Add dialogue before generating.",
                        )
                        return

                    word_count = len(script.split())
                    if word_count > MAX_SCRIPT_WORDS:
                        yield (
                            build_primary_status("error",
                                f"Script too long: {word_count} words (max {MAX_SCRIPT_WORDS}). "
                                "Shorten some turns to keep generation costs reasonable."),
                            gr.update(label=AUDIO_STAGE_LABELS.get("error", AUDIO_LABEL_DEFAULT)),
                            f"Script has {word_count} words, max is {MAX_SCRIPT_WORDS}.",
                        )
                        return

                    yield (
                        build_primary_status("connecting", "Provisioning GPU..."),
                        gr.update(label=AUDIO_STAGE_LABELS.get("connecting", AUDIO_LABEL_DEFAULT)),
                        "Requesting GPU on Modal.com...",
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
                                    audio_label = f"Audio ({stage_key.replace('_',' ')})"
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
                                        build_primary_status("complete", "Ready."),
                                        gr.update(value=audio_payload, label=AUDIO_LABEL_DEFAULT),
                                        current_log,
                                    )
                                else:
                                    yield (
                                        build_primary_status("generating_audio",
                                            current_log.splitlines()[-1] if current_log else "Processing..."),
                                        gr.update(label=AUDIO_STAGE_LABELS.get("generating_audio", last_audio_label)),
                                        current_log,
                                    )
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"Error calling Modal: {e}")
                        yield (
                            build_primary_status("error", "Inference failed."),
                            gr.update(label=AUDIO_STAGE_LABELS.get("error", AUDIO_LABEL_DEFAULT)),
                            f"Error: {e}\n\n{tb}",
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
                    """VibeVoice generates expressive, long-form, multi-speaker conversational audio from text.
                It uses continuous speech tokenizers at an ultra-low 7.5 Hz frame rate and a next-token diffusion
                framework to synthesize up to 90 minutes of speech with up to 4 distinct speakers."""
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
### Key Features
- **Multi-Speaker**: Up to 4 distinct voices
- **Long-Form**: Up to 90 minutes of audio
- **Natural Flow**: Turn-taking, interruptions, filler words
- **Efficient**: 7.5 Hz tokenizers for low compute cost

### Architecture
1. **Continuous Speech Tokenizers** — Acoustic + Semantic at 7.5 Hz
2. **Next-Token Diffusion** — LLM context + diffusion generation
3. **Diffusion Head** — High-fidelity acoustic output

### Models
- **1.5B** — Fast inference, great for iteration
- **7B** — Higher fidelity, longer generation time
                        """)
                    with gr.Column():
                        gr.Image(value="public/images/diagram.jpg",
                                 label="Architecture", show_download_button=False)
                        gr.Image(value="public/images/chart.png",
                                 label="Performance", show_download_button=False)

    return interface


# --- Main ---
if __name__ == "__main__":
    if remote_generate_function is None:
        with gr.Blocks(theme=theme) as interface:
            gr.Markdown("# Configuration Error")
            gr.Markdown(
                "Cannot connect to Modal backend. "
                "Run `modal deploy backend_modal/modal_runner.py` and refresh."
            )
        interface.launch()
    else:
        interface = create_demo_interface()
        interface.queue().launch(show_error=True)
