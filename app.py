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
VOICE_GENDERS = {
    "Cherry": "F", "Chicago": "M", "Janus": "M",
    "Mantis": "F", "Sponge": "M", "Starchild": "F",
}
AVAILABLE_VOICES = list(VOICE_GENDERS.keys())
VOICE_DISPLAY = [f"{name} ({g})" for name, g in VOICE_GENDERS.items()]
DEFAULT_SPEAKERS_DISPLAY = ["Cherry (F)", "Chicago (M)", "Janus (M)", "Mantis (F)"]


def voice_display_to_name(display: str) -> str:
    """Strip gender tag: 'Cherry (F)' -> 'Cherry'"""
    if display and " (" in display:
        return display.rsplit(" (", 1)[0]
    return display

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

# Matches "Speaker 3:" or a named character tag like "Mom:", "Dr. Smith:", "Wizard:"
# at the start of a line OR inline mid-paragraph. Captures the label and the text after it.
_SPEAKER_TAG = re.compile(
    r"(?:^|(?<=[\s\"'.!?,—–\-]))"                    # boundary: start or after whitespace/punct
    r"(Speaker\s+\d+|[A-Z][A-Za-z.'\- ]{0,24}?)"     # label: "Speaker N" OR capitalized name
    r"\s*:\s+"                                         # the colon separator
    r"(?=[A-Z\"'])",                                  # followed by capital letter / quote (real dialogue)
    re.MULTILINE,
)

# Labels we should NEVER treat as speaker tags (common false positives)
_LABEL_BLOCKLIST = {
    "title", "note", "scene", "setting", "fade in", "fade out", "cut to",
    "interior", "exterior", "int", "ext", "cont", "continued", "act",
}


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip().lower()


def parse_script_to_turns(script_text: str) -> list[dict]:
    """Parse dialogue into turns, handling both 'Speaker N:' and named-character tags.

    Robust to the LLM slipping in mid-paragraph speaker changes like:
        Speaker 1: ...We need magic. Mom: Hey kids! ...
    which get split into separate turns, with 'Mom' mapped to its own Speaker number.
    """
    turns: list[dict] = []
    if not script_text or not script_text.strip():
        return turns

    text = script_text.strip()

    # 1. Find every speaker tag occurrence in the entire text (line-start OR mid-line).
    tags: list[tuple[int, int, str]] = []  # (start, end, label)
    for m in _SPEAKER_TAG.finditer(text):
        label = m.group(1).strip()
        norm = _normalize_label(label)
        if norm in _LABEL_BLOCKLIST:
            continue
        # Reject labels that are just common sentence-starters that happen to precede a colon
        if norm in {"well", "so", "okay", "yes", "no", "right", "look", "listen"}:
            continue
        tags.append((m.start(), m.end(), label))

    if not tags:
        # No tags at all — treat entire text as Speaker 1
        return [{"speaker": 1, "text": text}]

    # 2. Assign each unique label to a speaker number.
    # First, reserve slots for all explicit "Speaker N" numbers in the script,
    # so inline named characters (Mom, Wizard) don't steal those numbers.
    label_to_speaker: dict[str, int] = {}
    reserved_numbers: set[int] = set()
    for _, _, lbl in tags:
        m = re.match(r"speaker\s+(\d+)", lbl, re.IGNORECASE)
        if m:
            reserved_numbers.add(int(m.group(1)))

    def speaker_for(label: str) -> int:
        # "Speaker N" preserves its number; named labels get auto-assigned.
        m = re.match(r"speaker\s+(\d+)", label, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            label_to_speaker.setdefault(label.lower(), n)
            return n
        key = label.lower()
        if key in label_to_speaker:
            return label_to_speaker[key]
        # Find next available speaker number (1..4), skipping reserved & already-used.
        used = set(label_to_speaker.values()) | reserved_numbers
        for n in range(1, 5):
            if n not in used:
                label_to_speaker[key] = n
                return n
        # Overflow: reuse highest available named slot, cap at 4
        label_to_speaker[key] = 4
        return 4

    # 3. Walk tags and slice out each turn's text (from end-of-tag to start-of-next-tag).
    # Any leading text before the first tag is ignored (usually empty / title residue).
    for i, (start, end, label) in enumerate(tags):
        next_start = tags[i + 1][0] if i + 1 < len(tags) else len(text)
        body = text[end:next_start].strip()
        body = re.sub(r"\s+", " ", body)
        if not body:
            continue
        turns.append({"speaker": speaker_for(label), "text": body})

    return turns


def turns_to_script(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        if t.get("text", "").strip():
            lines.append(f"Speaker {t['speaker']}: {t['text'].strip()}")
    return "\n\n".join(lines)


def estimate_duration(turns: list[dict]) -> str:
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

CASTING (IMPORTANT):
- Before writing, identify EVERY character in the scenario — including any who enter, interrupt, or arrive later (parents, bosses, narrators, bystanders, etc.)
- If the prompt mentions someone at all, they get their own Speaker number (up to 4 max)
- Example: "Two kids argue until their mom walks in" = 3 speakers, not 2
- Example: "A detective interviews a suspect while a lawyer objects" = 3 speakers
- Assign Speaker numbers in order of first appearance

STRICT NO-NO's (VibeVoice reads these LITERALLY as spoken words — never use them):
- NO bracketed stage directions: [whispering], [sighs], [laughs], [door slams], [pause], [music], etc.
- NO parenthetical emotion cues: (softly), (angrily), (laughing), (sarcastically), etc.
- NO asterisk actions: *laughs*, *sighs*, *door opens*, etc.
- NO scene headings, sound effects, or narration lines
- Convey emotion through WORD CHOICE and natural speech only (e.g., actually type "hahaha" or "ugh" or "whoa" as part of the dialogue itself)

FORMAT RULES:
- Start with a title on the FIRST LINE in this format: "Title: Your Script Title Here"
- Then a blank line, then the dialogue
- Use EXACTLY this format for dialogue: "Speaker N: dialogue text" where N starts at 1
- Each turn is separated by a blank line
- Choose the right number of speakers for the scenario (1 to 4 max)
- Keep the total script under {max_words} words
- Output ONLY the title and script — no stage directions, no commentary, no preamble

CRITICAL — ONE SPEAKER PER TURN:
- NEVER embed another character's dialogue inside someone else's turn
- WRONG: "Speaker 1: We need magic. Mom: Hey kids, what's going on?"
- RIGHT: Every time the speaker changes, END the current turn, add a BLANK LINE, then start a NEW turn with "Speaker N:" on its own line
- Do NOT use character names as inline labels like "Mom:" or "Wizard:" mid-paragraph — always use "Speaker N:" on a fresh line"""


# Strip bracketed stage directions, parenthetical cues, and asterisk actions.
# VibeVoice reads these literally, so we defensively remove them even if the LLM sneaks them in.
_STAGE_DIRECTION_PATTERNS = [
    re.compile(r"\[[^\]]*\]"),           # [whispering], [sighs], [door slams]
    re.compile(r"\*[^*\n]+\*"),          # *laughs*, *sighs*
]
# Common parenthetical emotion/action cues — only strip short ones that look like directions,
# not legitimate asides like "(which, by the way, is huge)".
_PAREN_CUE_WORDS = {
    "softly", "angrily", "laughing", "laughs", "sighs", "sighing", "whispers", "whispering",
    "shouts", "shouting", "sarcastically", "sarcastic", "nervously", "excitedly",
    "quietly", "loudly", "pauses", "pause", "crying", "sobbing", "giggling", "chuckling",
    "sternly", "coldly", "warmly", "mockingly", "sadly", "happily", "angry", "sad",
    "clears throat", "beat", "aside", "muttering", "mutters", "groans", "groaning",
}
_PAREN_PATTERN = re.compile(r"\(([^)\n]{1,40})\)")


def sanitize_dialogue(text: str) -> str:
    """Remove stage directions VibeVoice would read as literal words."""
    for pat in _STAGE_DIRECTION_PATTERNS:
        text = pat.sub("", text)

    def _paren_filter(m):
        inside = m.group(1).strip().lower().rstrip(".!?")
        if inside in _PAREN_CUE_WORDS:
            return ""
        # Also strip single-word parentheticals ending in -ly (adverbs)
        if " " not in inside and inside.endswith("ly"):
            return ""
        return m.group(0)  # keep legitimate asides

    text = _PAREN_PATTERN.sub(_paren_filter, text)
    # Collapse whitespace the stripping may have introduced
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def generate_script_from_prompt(prompt: str) -> tuple[list[dict], int, str]:
    """Returns (turns, num_speakers, title)."""
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

    # Extract title from first line if present
    title = ""
    lines = raw.strip().split("\n")
    if lines and lines[0].lower().startswith("title:"):
        title = lines[0].split(":", 1)[1].strip()
        raw = "\n".join(lines[1:])

    turns = parse_script_to_turns(raw)
    # Scrub stage directions from each turn, drop any turn that becomes empty
    turns = [
        {"speaker": t["speaker"], "text": sanitize_dialogue(t["text"])}
        for t in turns
    ]
    turns = [t for t in turns if t["text"].strip()]
    turns = turns[:MAX_TURNS]
    total_words = sum(len(t["text"].split()) for t in turns)
    while total_words > MAX_SCRIPT_WORDS and turns:
        turns.pop()
        total_words = sum(len(t["text"].split()) for t in turns)
    speaker_ids = {t["speaker"] for t in turns}
    num_speakers = max(min(len(speaker_ids), 4), 1) if speaker_ids else 1
    return turns, num_speakers, title


PARODY_SYSTEM_PROMPT = """You are a comedian narrator. The user will give you a scenario. Write a SHORT, funny behind-the-scenes narration of what's "really" happening while their audio is being generated. Be absurd, self-aware, and poke fun at AI.

RULES:
- Write 15-25 short sentences, one per line
- Each line should be its own complete funny thought or observation
- Reference the user's scenario but make it ridiculous
- Break the fourth wall — you know you're an AI generating audio
- Mix in jokes about GPUs, neural networks, robots, etc.
- Keep it PG and lighthearted
- Output ONLY the lines, no numbering, no quotes"""


def generate_parody_story(prompt: str) -> list[str]:
    """Generate a funny behind-the-scenes narration for the loading screen."""
    try:
        response = llm_client.chat_completion(
            messages=[
                {"role": "system", "content": PARODY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.9,
        )
        raw = response.choices[0].message.content
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        return lines if lines else ["Generating your audio... hang tight!"]
    except Exception as e:
        print(f"Parody generation failed (non-critical): {e}")
        return ["Generating your audio... hang tight!"]


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

/* ---- Voice settings row ---- */
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

/* ---- CTA buttons ---- */
.cta-btn button {
    border-radius: 10px !important;
    font-size: 1.1em !important;
    padding: 14px 24px !important;
    letter-spacing: 0.02em;
}

/* ---- Status text (inline, no chrome) ---- */
.status-text {
    font-style: italic;
    opacity: 0.8;
    padding: 8px 0;
    min-height: 0 !important;
}

/* ---- Empty state ---- */
.empty-state {
    text-align: center;
    padding: 48px 20px !important;
    opacity: 0.6;
}

/* ---- Generation status banner ---- */
.gen-status {
    border-radius: 12px;
    padding: 18px 24px;
    margin-top: 10px;
    text-align: center;
    font-size: 1.05em;
    min-height: 0;
}
.gen-status-active {
    border: 2px solid #6366f1;
    background: linear-gradient(
        90deg,
        rgba(99, 102, 241, 0.05) 0%,
        rgba(99, 102, 241, 0.18) 30%,
        rgba(236, 72, 153, 0.14) 50%,
        rgba(99, 102, 241, 0.18) 70%,
        rgba(99, 102, 241, 0.05) 100%
    );
    background-size: 300% 100%;
    animation: gradient-sweep 2.5s ease-in-out infinite, border-glow 2s ease-in-out infinite;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.15), 0 0 30px rgba(99, 102, 241, 0.05);
}
@keyframes gradient-sweep {
    0% { background-position: 100% 0; }
    50% { background-position: 0% 0; }
    100% { background-position: 100% 0; }
}
@keyframes border-glow {
    0%, 100% {
        border-color: #6366f1;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.15), 0 0 30px rgba(99, 102, 241, 0.05);
    }
    33% {
        border-color: #ec4899;
        box-shadow: 0 0 15px rgba(236, 72, 153, 0.2), 0 0 30px rgba(236, 72, 153, 0.08);
    }
    66% {
        border-color: #22c55e;
        box-shadow: 0 0 15px rgba(34, 197, 94, 0.2), 0 0 30px rgba(34, 197, 94, 0.08);
    }
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


def build_status_html(stage: str, status_line: str) -> str:
    """Build an HTML status banner for the generation progress."""
    title, default_desc = PRIMARY_STAGE_MESSAGES.get(stage, ("Working", "Processing..."))
    desc = status_line or default_desc or ""

    if stage == "complete":
        icon = '<span style="color:#22c55e; font-size:1.4em; vertical-align:middle;">&#10003;</span>'
        cls = "gen-status"
    elif stage == "error":
        icon = '<span style="color:#ef4444; font-size:1.4em; vertical-align:middle;">&#10007;</span>'
        cls = "gen-status"
    else:
        # Pulsing animated dot
        icon = (
            '<span style="display:inline-block; width:12px; height:12px; '
            'border-radius:50%; background:#6366f1; vertical-align:middle; '
            'animation: dot-pulse 1.2s ease-in-out infinite;"></span>'
        )
        cls = "gen-status gen-status-active"

    return (
        f'<style>@keyframes dot-pulse {{ 0%,100% {{ opacity:1; transform:scale(1); }} 50% {{ opacity:0.4; transform:scale(0.7); }} }}</style>'
        f'<div class="{cls}">'
        f'{icon} <strong style="font-size:1.1em;">{title}</strong>'
        f'<br><span style="opacity:0.7; font-size:0.9em;">{desc}</span>'
        f'</div>'
    )


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
        script_title_state = gr.State("")
        parody_lines_state = gr.State([])  # funny loading story for audio generation

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

                # ---- STEP 1: DESCRIBE ----
                gr.HTML("""
                <p style="margin:0 0 4px 0; opacity:0.65; font-size:0.9em;">
                    Describe any scenario and AI writes the script. Then edit, assign voices, and generate audio.
                </p>
                """)
                script_prompt = gr.Textbox(
                    label="Describe your conversation",
                    placeholder="A wizard and an orc debating battle strategy before a siege...",
                    lines=2,
                    max_lines=3,
                )
                generate_script_btn = gr.Button(
                    "Write Script with AI", variant="primary",
                    size="lg", elem_classes="cta-btn",
                )
                script_gen_status = gr.HTML(value="", elem_classes="status-text")

                # ---- EXAMPLES ----
                example_names = [
                    "AI TED Talk", "Political Speech",
                    "Finance IPO", "Telehealth",
                    "Military Briefing", "Oil & Energy",
                    "Game Dev Meeting", "Product Review",
                ]
                with gr.Row():
                    example_buttons = []
                    for name in example_names:
                        btn = gr.Button(name, size="sm", variant="secondary",
                                        elem_classes="example-btn", min_width=80)
                        example_buttons.append(btn)

                # ---- STEP 2: SCRIPT EDITOR ----
                with gr.Row():
                    script_title_display = gr.HTML(value="<h3 style='margin:0'>Script</h3>")
                    duration_display = gr.HTML(value="")

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

                        # Show all 4 speakers with voice name + gender
                        speaker_choices = []
                        for i in range(4):
                            voice_name = AVAILABLE_VOICES[i] if i < len(AVAILABLE_VOICES) else "?"
                            gender = VOICE_GENDERS.get(voice_name, "")
                            speaker_choices.append(f"Speaker {i+1} - {voice_name} ({gender})")

                        for idx, turn in enumerate(turns):
                            spk_num = turn["speaker"]
                            color_class = f"speaker-{spk_num}" if 1 <= spk_num <= 4 else "speaker-1"
                            spk_value = speaker_choices[spk_num - 1] if 1 <= spk_num <= 4 else speaker_choices[0]

                            with gr.Row(key=f"turn-{idx}", elem_classes=color_class):
                                spk_dd = gr.Dropdown(
                                    choices=speaker_choices,
                                    value=spk_value,
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
                                    # Parse "Speaker 2 - Chicago (M)" -> 2
                                    m = re.match(r"Speaker (\d+)", new_spk)
                                    if m:
                                        current_turns[i]["speaker"] = int(m.group(1))
                                return current_turns

                            spk_dd.change(fn=on_speaker_change, inputs=[spk_dd, turns_state],
                                          outputs=[turns_state], queue=False)

                            def on_delete(current_turns, i=idx):
                                if i < len(current_turns):
                                    current_turns.pop(i)
                                return current_turns

                            del_btn.click(fn=on_delete, inputs=[turns_state],
                                          outputs=[turns_state])

                add_turn_btn = gr.Button("+ Add Turn", size="sm", variant="secondary")

                # ---- STEP 3: VOICE & MODEL ----
                num_speakers = gr.Slider(
                    minimum=1, maximum=4, value=2, step=1, visible=False,
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
                                choices=VOICE_DISPLAY,
                                value=DEFAULT_SPEAKERS_DISPLAY[i] if i < len(DEFAULT_SPEAKERS_DISPLAY) else None,
                                label=f"Voice {i+1}",
                                visible=(i < 2),
                                scale=1,
                            )
                            speaker_selections.append(s)
                    with gr.Row():
                        cfg_scale = gr.Slider(
                            minimum=1.0, maximum=2.0, value=2.0, step=0.05,
                            label="CFG Scale",
                        )

                # ---- STEP 4: GENERATE ----
                generate_btn = gr.Button(
                    "Generate Conference Audio", size="lg", variant="primary",
                    elem_classes="cta-btn",
                )
                primary_status = gr.HTML(value="", elem_classes="gen-status")

                # ---- OUTPUT ----
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
                import time as _time
                import threading as _threading

                SCRIPT_GEN_MESSAGES = [
                    "Writing script...",
                    "Still generating...",
                    "Making magic happen...",
                    "Bossing around robot writers...",
                    "Entering the matrix...",
                    "Crafting dialogue...",
                    "Teaching AI to be dramatic...",
                    "Consulting the creative robots...",
                    "Spilling digital ink...",
                    "Herding AI cats into a script...",
                    "Negotiating with the muse...",
                    "Downloading inspiration...",
                    "Warming up the plot engine...",
                    "Shaking the idea tree...",
                    "Feeding the word machine...",
                    "Polishing virtual microphones...",
                    "Rehearsing in the AI green room...",
                    "Bribing the creativity daemon...",
                    "Untangling narrative spaghetti...",
                    "Summoning fictional characters...",
                    "Tuning the dialogue generator...",
                    "Spinning up the story factory...",
                    "Convincing electrons to be eloquent...",
                    "Wrangling syllables into sentences...",
                    "Asking the AI to use its inside voice...",
                    "Loading dramatic tension...",
                    "Calibrating the sass levels...",
                    "Assembling words in the right order...",
                    "Generating witty banter...",
                    "Overthinking your prompt (in a good way)...",
                    "Running it by the robot editor...",
                    "Adding a pinch of personality...",
                    "Almost done, probably...",
                    "Spell-checking the AI's homework...",
                    "Giving characters their motivation...",
                    "Practicing dramatic pauses...",
                    "Reticulating splines (just kidding)...",
                    "The AI is in the zone...",
                    "Finalizing the masterpiece...",
                    "One more revision, we promise...",
                ]

                # outputs: turns, duration, status, title, audio, script_btn, gen_btn, parody, num_speakers, *4 voices
                def _script_no_change(status_html):
                    return (gr.update(), gr.update(), status_html,
                            gr.update(), gr.update(),
                            gr.update(), gr.update(),
                            gr.update(),
                            gr.update(), *[gr.update()] * 4)

                def _script_buttons_busy(status_html):
                    return (gr.update(), gr.update(), status_html,
                            gr.update(), gr.update(),
                            gr.update(interactive=False, value="Writing..."),
                            gr.update(interactive=False),
                            gr.update(),
                            gr.update(), *[gr.update()] * 4)

                def _script_buttons_ready(status_html=""):
                    return (gr.update(), gr.update(), status_html,
                            gr.update(), gr.update(),
                            gr.update(interactive=True, value="Write Script with AI"),
                            gr.update(interactive=True),
                            gr.update(),
                            gr.update(), *[gr.update()] * 4)

                def _make_title_html(title):
                    if title:
                        return f"<h3 style='margin:0'>{title}</h3>"
                    return "<h3 style='margin:0'>Script</h3>"

                def on_generate_script(prompt):
                    if not prompt or not prompt.strip():
                        gr.Warning("Please enter a prompt.")
                        yield _script_no_change("")
                        return

                    # Disable both buttons
                    yield _script_buttons_busy(f"<em>{SCRIPT_GEN_MESSAGES[0]}</em>")

                    # Run script + parody generation in threads
                    result = {}
                    error = {}
                    parody_result = {"lines": []}

                    def _run():
                        try:
                            result["data"] = generate_script_from_prompt(prompt.strip())
                        except Exception as e:
                            error["err"] = e

                    def _run_parody():
                        parody_result["lines"] = generate_parody_story(prompt.strip())

                    thread = _threading.Thread(target=_run, daemon=True)
                    parody_thread = _threading.Thread(target=_run_parody, daemon=True)
                    thread.start()
                    parody_thread.start()

                    msg_idx = 1
                    while thread.is_alive():
                        _time.sleep(3)
                        msg = SCRIPT_GEN_MESSAGES[msg_idx % len(SCRIPT_GEN_MESSAGES)]
                        msg_idx += 1
                        yield _script_buttons_busy(f"<em>{msg}</em>")

                    thread.join()
                    parody_thread.join()

                    if "err" in error:
                        e = error["err"]
                        print(f"Script generation error: {e}")
                        traceback.print_exc()
                        msg = str(e)
                        if "api_key" in msg or "log in" in msg or "token" in msg.lower():
                            yield _script_buttons_ready("<em>HF_TOKEN not configured. Add it in Space Settings.</em>")
                        else:
                            yield _script_buttons_ready(f"<em>Error: {msg[:200]}</em>")
                        return

                    turns, detected, title = result["data"]
                    if not turns:
                        yield _script_buttons_ready("<em>Empty result — try a more descriptive prompt.</em>")
                        return

                    voices = list(VOICE_DISPLAY[:detected])
                    while len(voices) < 4:
                        voices.append(None)

                    audio_label = title if title else AUDIO_LABEL_DEFAULT
                    yield (turns, estimate_duration(turns), "",
                           _make_title_html(title),
                           gr.update(label=audio_label),
                           gr.update(interactive=True, value="Write Script with AI"),
                           gr.update(interactive=True),
                           parody_result["lines"],
                           detected, *voices[:4])

                generate_script_btn.click(
                    fn=on_generate_script,
                    inputs=[script_prompt],
                    outputs=[turns_state, duration_display, script_gen_status,
                             script_title_display, complete_audio_output,
                             generate_script_btn, generate_btn,
                             parody_lines_state,
                             num_speakers] + speaker_selections,
                )

                # --- Load examples ---
                def load_example(idx):
                    if idx >= len(EXAMPLE_SCRIPTS):
                        return [], 2, "", "<h3 style='margin:0'>Script</h3>", gr.update(), *[None] * 4

                    title = example_names[idx]
                    script = EXAMPLE_SCRIPTS_NATURAL[idx]
                    num = SCRIPT_SPEAKER_COUNTS[idx] if idx < len(SCRIPT_SPEAKER_COUNTS) else 1
                    turns = parse_script_to_turns(script)

                    voices = list(VOICE_DISPLAY[:num])
                    while len(voices) < 4:
                        voices.append(None)

                    return (turns, num, estimate_duration(turns),
                            f"<h3 style='margin:0'>{title}</h3>",
                            gr.update(label=title),
                            *voices[:4])

                for idx, btn in enumerate(example_buttons):
                    btn.click(
                        fn=lambda i=idx: load_example(i),
                        inputs=[],
                        outputs=[turns_state, num_speakers, duration_display,
                                 script_title_display, complete_audio_output] + speaker_selections,
                        queue=False,
                    )

                # --- Generate audio ---
                def _gen_yield(status_html, btn_label, btn_interactive, audio_update, log_text):
                    return (
                        status_html,
                        gr.update(value=btn_label, interactive=btn_interactive),
                        gr.update(interactive=btn_interactive),
                        audio_update,
                        log_text,
                    )

                def generate_podcast_wrapper(
                    model_choice, num_speakers_val, turns, parody_lines, *speakers_and_params
                ):
                    BTN_BUSY = "Generating..."
                    BTN_READY = "Generate Conference Audio"

                    # Set up parody line cycling
                    parody_idx = [0]  # mutable counter
                    def _next_parody():
                        if not parody_lines:
                            return None
                        line = parody_lines[parody_idx[0] % len(parody_lines)]
                        parody_idx[0] += 1
                        return line

                    if remote_generate_function is None:
                        yield _gen_yield(
                            build_status_html("error", "Modal backend is offline."),
                            BTN_READY, True,
                            gr.update(), "ERROR: Modal function not deployed.",
                        )
                        return

                    script = turns_to_script(turns)
                    if not script.strip():
                        yield _gen_yield(
                            build_status_html("error", "No script to generate."),
                            BTN_READY, True,
                            gr.update(), "Add dialogue before generating.",
                        )
                        return

                    word_count = len(script.split())
                    if word_count > MAX_SCRIPT_WORDS:
                        yield _gen_yield(
                            build_status_html("error",
                                f"Script too long: {word_count} words (max {MAX_SCRIPT_WORDS}). "
                                "Shorten some turns."),
                            BTN_READY, True,
                            gr.update(), f"Script has {word_count} words, max is {MAX_SCRIPT_WORDS}.",
                        )
                        return

                    # Disable buttons, show connecting status
                    first_line = _next_parody() or "Provisioning GPU resources..."
                    yield _gen_yield(
                        build_status_html("connecting", first_line),
                        BTN_BUSY, False,
                        gr.update(label=AUDIO_STAGE_LABELS.get("connecting", AUDIO_LABEL_DEFAULT)),
                        "Requesting GPU on Modal.com...",
                    )

                    try:
                        speakers = [voice_display_to_name(s) for s in speakers_and_params[:4]]
                        cfg_scale_val = speakers_and_params[4]
                        current_log = ""
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

                                audio_label = AUDIO_STAGE_LABELS.get(stage_key,
                                    f"Audio ({stage_key.replace('_',' ')})")
                                is_done = stage_key in ("complete", "error")
                                if stage_key == "complete":
                                    audio_label = AUDIO_LABEL_DEFAULT

                                audio_update = gr.update(label=audio_label)
                                if audio_payload is not None:
                                    audio_update = gr.update(value=audio_payload, label=AUDIO_LABEL_DEFAULT)

                                # Use parody line for active stages, real status for done
                                if is_done:
                                    display_line = status_line
                                else:
                                    display_line = _next_parody() or status_line

                                yield _gen_yield(
                                    build_status_html(stage_key, display_line),
                                    BTN_READY if is_done else BTN_BUSY,
                                    is_done,
                                    audio_update,
                                    current_log,
                                )
                                last_stage = stage_key
                            else:
                                audio_payload, log_text = (
                                    update if isinstance(update, (tuple, list)) else (None, str(update))
                                )
                                if log_text:
                                    current_log = log_text
                                if audio_payload is not None:
                                    yield _gen_yield(
                                        build_status_html("complete", "Ready."),
                                        BTN_READY, True,
                                        gr.update(value=audio_payload, label=AUDIO_LABEL_DEFAULT),
                                        current_log,
                                    )
                                else:
                                    display_line = _next_parody() or (
                                        current_log.splitlines()[-1] if current_log else "Processing...")
                                    yield _gen_yield(
                                        build_status_html("generating_audio", display_line),
                                        BTN_BUSY, False,
                                        gr.update(), current_log,
                                    )
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"Error calling Modal: {e}")
                        yield _gen_yield(
                            build_status_html("error", "Inference failed."),
                            BTN_READY, True,
                            gr.update(), f"Error: {e}\n\n{tb}",
                        )

                generate_btn.click(
                    fn=generate_podcast_wrapper,
                    inputs=[model_dropdown, num_speakers, turns_state, parody_lines_state] + speaker_selections + [cfg_scale],
                    outputs=[primary_status, generate_btn, generate_script_btn, complete_audio_output, log_output],
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
