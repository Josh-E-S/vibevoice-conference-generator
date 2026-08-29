from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import re
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Annotated

import modal
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import InferenceClient
from pydantic import BaseModel
from scipy.io import wavfile

# --- Configuration ---
ROOT = Path(__file__).resolve().parent
MODAL_STUB_NAME = "vibevoice-generator"
MODAL_CLASS_NAME = "VibeVoiceModel"

AVAILABLE_MODELS = ["VibeVoice-1.5B", "VibeVoice-7B"]
VOICE_INFO = {
    "Cherry": {"gender": "F", "tags": ["Warm", "Storyteller"], "color": "#E2582A"},
    "Chicago": {"gender": "M", "tags": ["Deep", "Narrator"], "color": "#2F6F63"},
    "Janus": {"gender": "M", "tags": ["Bright", "Conversational"], "color": "#CC8A2E"},
    "Mantis": {"gender": "F", "tags": ["Crisp", "Energetic"], "color": "#7B4B94"},
    "Sponge": {"gender": "M", "tags": ["Playful", "Animated"], "color": "#3A7CA5"},
    "Starchild": {"gender": "F", "tags": ["Airy", "Dreamy"], "color": "#B6558C"},
}
VOICE_GENDERS = {name: info["gender"] for name, info in VOICE_INFO.items()}
AVAILABLE_VOICES = list(VOICE_GENDERS.keys())
DEFAULT_SPEAKERS = ["Cherry", "Chicago", "Janus", "Mantis"]

SCRIPT_GEN_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
WORDS_PER_MINUTE = 150             # Matches the pace assumed by the client's duration estimate
DURATION_OPTIONS_MINUTES = [1, 2, 5, 10, 15, 20, 30, 45, 60]
MAX_COMPLETION_TOKENS = 8192       # Good-faith ceiling for a single chat_completion call; the
                                    # underlying provider may cap lower, in which case the longest
                                    # duration options may come back shorter than requested
MAX_SCRIPT_WORDS = 100000         # Effectively uncapped (2026-08-14, Josh) — backend chunking has no
                                   # real ceiling; the old 20,000 cap was a leftover UI guess that
                                   # blocked genuine long-form renders below the backend's actual limit
MAX_TURNS = 250                    # Hard ceiling regardless of target length (safety valve)
MAX_GEN_ROUNDS = 24                # Hard cap on LLM calls per script regardless of target length
MIN_TARGET_FRACTION = 0.9          # Stop extending once the script reaches 90% of the word target
MIN_ROUND_YIELD_WORDS = 40         # A continuation shorter than this means the model is done talking
AUDIO_TTL_SECONDS = 900
MAX_CUSTOM_AUDIO_BYTES = 15 * 1024 * 1024  # cap per uploaded voice-clone clip


# --- Load example scripts ---
def load_example_scripts():
    examples_dir = ROOT / "text_examples"
    example_scripts = []
    example_scripts_natural = []

    if not examples_dir.exists():
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
        file_path = examples_dir / txt_file
        natural_path = examples_dir / txt_file.replace(".txt", "_natural.txt")

        if file_path.exists():
            example_scripts.append(file_path.read_text(encoding="utf-8"))
        else:
            example_scripts.append("")

        if natural_path.exists():
            example_scripts_natural.append(natural_path.read_text(encoding="utf-8"))
        else:
            example_scripts_natural.append(example_scripts[-1] if example_scripts else "")

    return example_scripts, example_scripts_natural


EXAMPLE_NAMES = [
    "AI TED Talk", "Political Speech",
    "Finance IPO", "Telehealth",
    "Military Briefing", "Oil & Energy",
    "Game Dev Meeting", "Product Review",
]
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
- LENGTH TARGET: write approximately {target_words} words total — enough dialogue to fill
  roughly {target_minutes} minute(s) of natural spoken audio. This is a target, not just a
  ceiling: keep the conversation developing — new angles, follow-up questions, examples,
  pushback — rather than wrapping up early. Do not stop far short of the target.
- Output ONLY the title and script — no stage directions, no commentary, no preamble

CRITICAL — ONE SPEAKER PER TURN:
- NEVER embed another character's dialogue inside someone else's turn
- WRONG: "Speaker 1: We need magic. Mom: Hey kids, what's going on?"
- RIGHT: Every time the speaker changes, END the current turn, add a BLANK LINE, then start a NEW turn with "Speaker N:" on its own line
- Do NOT use character names as inline labels like "Mom:" or "Wizard:" mid-paragraph — always use "Speaker N:" on a fresh line

AFTER THE DIALOGUE — Character roster (REQUIRED):
- After the final dialogue turn, add a blank line, then a single line in this EXACT format:
  Character Genders: Speaker 1: <F or M>, Speaker 2: <F or M>, Speaker 3: <F or M>, Speaker 4: <F or M>
- Only list speakers you actually used. Use "F" for feminine-presenting voices (women, girls, moms, queens, witches, female narrators) and "M" for masculine-presenting (men, boys, dads, kings, wizards-as-male, male narrators).
- For gender-ambiguous roles (robots, narrators, dragons), pick whichever fits the tone. Never use "N" or "?"
- Example: "Character Genders: Speaker 1: M, Speaker 2: M, Speaker 3: F" """

# Sent when the first response comes back short of the word target — models chronically
# undershoot large word counts, so the script is built up over multiple calls.
CONTINUATION_PROMPT = """The script so far is about {so_far} words, but the target is roughly {target_words} words (~{target_minutes} minutes of spoken audio). Continue the SAME conversation from exactly where it left off.

- Do NOT restart, re-title, recap, or wrap up — keep the conversation developing with new angles, follow-up questions, examples, digressions, and pushback
- Write approximately {chunk_words} more words of dialogue
- Begin your reply with "Speaker N:" on its own line (whichever speaker would naturally talk next) and keep the exact same "Speaker N:" format and numbering as before
- Same style rules apply: full-paragraph turns, no stage directions, no commentary
- Do NOT repeat the "Character Genders:" line"""


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


_GENDER_LINE = re.compile(
    r"character\s+genders\s*:\s*(.+?)$",
    re.IGNORECASE | re.MULTILINE,
)
_GENDER_PAIR = re.compile(r"speaker\s+(\d+)\s*:\s*([FM])", re.IGNORECASE)


def _extract_genders(raw: str) -> tuple[str, dict[int, str]]:
    """Find and remove the 'Character Genders: ...' line. Returns (cleaned_text, genders_dict)."""
    genders: dict[int, str] = {}
    m = _GENDER_LINE.search(raw)
    if not m:
        return raw, genders
    for pair in _GENDER_PAIR.finditer(m.group(1)):
        try:
            n = int(pair.group(1))
            g = pair.group(2).upper()
            if 1 <= n <= 4 and g in ("F", "M"):
                genders[n] = g
        except ValueError:
            pass
    cleaned = raw[: m.start()].rstrip() + "\n" + raw[m.end():].lstrip()
    return cleaned, genders


def assign_voices_by_gender(genders: dict[int, str], num_speakers: int) -> list[str | None]:
    """Return a list of 4 plain voice names, picking matching-gender voices without duplicates.

    Falls back to DEFAULT_SPEAKERS if no gender info for a slot.
    """
    female_pool = [v for v in AVAILABLE_VOICES if VOICE_GENDERS.get(v) == "F"]
    male_pool = [v for v in AVAILABLE_VOICES if VOICE_GENDERS.get(v) == "M"]
    used: set[str] = set()
    chosen: list[str | None] = []

    for i in range(4):
        slot = i + 1
        if slot <= num_speakers:
            g = genders.get(slot)
            pool = female_pool if g == "F" else (male_pool if g == "M" else AVAILABLE_VOICES)
            pick = next((v for v in pool if v not in used), None)
            if pick is None:
                pick = next((v for v in AVAILABLE_VOICES if v not in used), AVAILABLE_VOICES[0])
            used.add(pick)
            chosen.append(pick)
        else:
            chosen.append(DEFAULT_SPEAKERS[i] if i < len(DEFAULT_SPEAKERS) else None)
    return chosen


def generate_script_from_prompt(
    prompt: str, target_minutes: int = 2
) -> tuple[list[dict], int, str, list[str | None]]:
    """Returns (turns, num_speakers, title, voice_selections)."""
    target_minutes = target_minutes if target_minutes in DURATION_OPTIONS_MINUTES else 2
    target_words = target_minutes * WORDS_PER_MINUTE
    min_words = int(target_words * MIN_TARGET_FRACTION)

    system = SCRIPT_SYSTEM_PROMPT.format(target_words=target_words, target_minutes=target_minutes)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    # Models chronically undershoot large word targets, so build the script over multiple
    # calls: generate, count actual words, and keep asking for continuations until the
    # script is close enough to the target (or the model stops producing new material).
    title = ""
    genders: dict[int, str] = {}
    turns: list[dict] = []
    total_words = 0
    # Real continuations yield ~350-800 words each; budget enough rounds to reach the
    # target even at the low end, within the hard cap.
    rounds_budget = max(3, min(MAX_GEN_ROUNDS, -(-target_words // 400) + 1))
    for round_i in range(rounds_budget):
        remaining_words = target_words - total_words
        completion_tokens = min(MAX_COMPLETION_TOKENS, int(remaining_words * 1.6) + 400)
        response = llm_client.chat_completion(
            messages=messages,
            max_tokens=completion_tokens,
            temperature=0.7,
        )
        raw = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": raw})

        # Extract title from first line if present (first round only)
        lines = raw.strip().split("\n")
        if round_i == 0 and lines and lines[0].lower().startswith("title:"):
            title = lines[0].split(":", 1)[1].strip()
            raw = "\n".join(lines[1:])

        # Extract and strip the "Character Genders:" line before parsing turns
        raw, chunk_genders = _extract_genders(raw)
        for n, g in chunk_genders.items():
            genders.setdefault(n, g)

        # Scrub stage directions from each turn, drop any turn that becomes empty
        chunk_turns = [
            {"speaker": t["speaker"], "text": sanitize_dialogue(t["text"])}
            for t in parse_script_to_turns(raw)
        ]
        chunk_turns = [t for t in chunk_turns if t["text"].strip()]
        chunk_words = sum(len(t["text"].split()) for t in chunk_turns)
        turns.extend(chunk_turns)
        total_words += chunk_words

        if total_words >= min_words or len(turns) >= MAX_TURNS:
            break
        if round_i > 0 and chunk_words < MIN_ROUND_YIELD_WORDS:
            break  # the model has run out of things to say; don't spin on empty rounds
        messages.append({
            "role": "user",
            "content": CONTINUATION_PROMPT.format(
                so_far=total_words,
                target_words=target_words,
                target_minutes=target_minutes,
                chunk_words=target_words - total_words,
            ),
        })

    turns = turns[:MAX_TURNS]
    # Allow some overshoot past the target before trimming — the model runs long sometimes.
    overshoot_ceiling = int(target_words * 1.3) + 100
    total_words = sum(len(t["text"].split()) for t in turns)
    while total_words > overshoot_ceiling and turns:
        turns.pop()
        total_words = sum(len(t["text"].split()) for t in turns)
    speaker_ids = {t["speaker"] for t in turns}
    num_speakers = max(min(len(speaker_ids), 4), 1) if speaker_ids else 1

    voice_selections = assign_voices_by_gender(genders, num_speakers)
    return turns, num_speakers, title, voice_selections


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


# --- Precomputed example index (parsed once at startup) ---
def _build_examples_index() -> list[dict]:
    index = []
    for i, name in enumerate(EXAMPLE_NAMES):
        script = EXAMPLE_SCRIPTS_NATURAL[i] if i < len(EXAMPLE_SCRIPTS_NATURAL) else ""
        turns = parse_script_to_turns(script)
        num = SCRIPT_SPEAKER_COUNTS[i] if i < len(SCRIPT_SPEAKER_COUNTS) else 1
        voices = (AVAILABLE_VOICES[:num] + [None, None, None, None])[:4]
        index.append({
            "id": i, "title": name, "turns": turns,
            "num_speakers": num, "voices": voices,
        })
    return index


# Examples retired 2026-08-22 (Josh) — the bundled scripts predate the current
# app and no longer fit it. The loader and text_examples/ files are left intact
# so restoring a curated set is a one-line change.
SHOW_BUNDLED_EXAMPLES = False
EXAMPLES_INDEX = _build_examples_index() if SHOW_BUNDLED_EXAMPLES else []


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


def _decode_custom_audio(value: str | None, slot: int) -> bytes | None:
    """Decode a base64 (optionally data:...;base64, prefixed) voice-clone upload."""
    if not value:
        return None
    if "," in value and value.strip().lower().startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        audio_bytes = base64.b64decode(value, validate=True)
    except (base64.binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Speaker {slot + 1}: invalid uploaded audio.") from e
    if len(audio_bytes) > MAX_CUSTOM_AUDIO_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Speaker {slot + 1}: uploaded voice clip is too large (max {MAX_CUSTOM_AUDIO_BYTES // (1024 * 1024)} MB).",
        )
    if not audio_bytes:
        return None
    return audio_bytes


def _encode_wav(sample_rate: int, audio: np.ndarray) -> bytes:
    audio = np.asarray(audio)
    if audio.dtype.kind == "f":
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    elif audio.dtype != np.int16:
        audio = audio.astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio)
    return buf.getvalue()


AUDIO_STORE: dict[str, tuple[float, bytes]] = {}


LAST_AUDIO: dict = {}  # {"audio_id", "created", "duration"} of the newest finished take


def _prune_audio_store() -> None:
    now = time.time()
    keep = LAST_AUDIO.get("audio_id")
    # Never prune the most recent take: a lost page/tab must not orphan an
    # hour-long render (learned the hard way, 2026-08-20). It is only evicted
    # when a newer take replaces it.
    stale = [
        k for k, (ts, _) in AUDIO_STORE.items()
        if now - ts > AUDIO_TTL_SECONDS and k != keep
    ]
    for k in stale:
        AUDIO_STORE.pop(k, None)
        MP3_CACHE.pop(k, None)


# --- Abuse guardrails ---
# In-memory, per-process — fine for a single-container Space, not a distributed rate limiter.
# Both generation endpoints hit metered third-party billing (HF Inference, Modal GPU time),
# so an unrestricted public endpoint is a direct route to running up someone else's bill.
_RATE_LOG: dict[str, deque] = defaultdict(deque)
SCRIPT_RATE_LIMIT = (5, 600)      # 5 script generations per 10 min per IP (hits paid HF inference)
AUDIO_RATE_LIMIT = (3, 3600)      # 3 audio generations per hour per IP (hits paid Modal GPU time)
GENERATION_CONCURRENCY = asyncio.Semaphore(2)  # at most 2 Modal generations in flight at once, globally


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(bucket: str, request: Request, limit: int, window_seconds: int) -> None:
    key = f"{bucket}:{_client_ip(request)}"
    now = time.time()
    log = _RATE_LOG[key]
    while log and now - log[0] > window_seconds:
        log.popleft()
    if len(log) >= limit:
        retry_after = max(1, int(window_seconds - (now - log[0])))
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached ({limit} per {window_seconds // 60} min). Try again in about {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )
    log.append(now)


# ========================================================
# FASTAPI APP
# ========================================================

class VersionedStaticFiles(StaticFiles):
    """Let content-hashed URLs (?v=…) be cached forever; everything else revalidates."""

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        scope = kwargs.get("scope") or next((a for a in args if isinstance(a, dict)), {})
        if b"v=" in scope.get("query_string", b""):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


app = FastAPI(title="VibeVoice Conference Generator")
app.mount("/static", VersionedStaticFiles(directory=ROOT / "static"), name="static")
app.mount("/public", StaticFiles(directory=ROOT / "public"), name="public")


# Stamp every /static asset the page loads with a hash of its contents, so a
# deploy can never leave a browser running yesterday's app.js while today's
# index.html expects it. Computed once per process; the HTML itself is
# no-cache so the new URLs are always picked up.
_ASSET_REF_RE = re.compile(r'(?P<attr>(?:href|src)=")(?P<path>/static/[^"?]+)(?P<end>")')
_INDEX_HTML: str | None = None


def _stamped_index_html() -> str:
    global _INDEX_HTML
    if _INDEX_HTML is not None:
        return _INDEX_HTML

    def stamp(match: re.Match) -> str:
        asset = ROOT / "static" / match.group("path")[len("/static/"):]
        try:
            digest = hashlib.sha256(asset.read_bytes()).hexdigest()[:10]
        except OSError:
            return match.group(0)  # referenced file missing — leave the URL alone
        return f'{match.group("attr")}{match.group("path")}?v={digest}{match.group("end")}'

    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    _INDEX_HTML = _ASSET_REF_RE.sub(stamp, html)
    return _INDEX_HTML


@app.get("/", include_in_schema=False)
async def index() -> Response:
    return Response(
        content=_stamped_index_html(),
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/status")
async def api_status() -> dict:
    return {"backend": "ready" if remote_generate_function is not None else "offline"}


@app.get("/api/models")
async def api_models() -> list[str]:
    return AVAILABLE_MODELS


@app.get("/api/voices")
async def api_voices() -> list[dict]:
    return [
        {
            "name": name,
            "gender": info["gender"],
            "tags": info["tags"],
            "color": info["color"],
            "preview_url": f"/public/voices/{name}.mp3",
        }
        for name, info in VOICE_INFO.items()
    ]


@app.get("/api/examples")
async def api_examples() -> list[dict]:
    return EXAMPLES_INDEX


@app.post("/api/parse-script")
async def api_parse_script(
    text: Annotated[str, Form()] = "",
    file: Annotated[UploadFile | None, File()] = None,
) -> dict:
    script = ""
    if file is not None and file.filename:
        raw = await file.read()
        script = raw.decode("utf-8", errors="replace")
        await file.close()
    if not script.strip():
        script = text or ""
    if not script.strip():
        raise HTTPException(status_code=400, detail="Paste a script or upload a .txt file first.")

    turns = parse_script_to_turns(script)
    if not turns:
        raise HTTPException(status_code=400, detail="Couldn't find any dialogue in that script.")

    total_words = sum(len(t["text"].split()) for t in turns)
    num = max(1, min(4, max(t["speaker"] for t in turns)))
    voices = (AVAILABLE_VOICES[:num] + [None, None, None, None])[:4]

    return {
        "turns": turns,
        "num_speakers": num,
        "title": "Uploaded Script",
        "voices": voices,
        "word_count": total_words,
        "over_limit": total_words > MAX_SCRIPT_WORDS,
    }


class ScriptPromptRequest(BaseModel):
    prompt: str
    target_minutes: int = 2


@app.get("/api/duration-options")
async def api_duration_options() -> list[int]:
    return DURATION_OPTIONS_MINUTES


@app.post("/api/generate-script")
async def api_generate_script(payload: ScriptPromptRequest, request: Request) -> dict:
    _enforce_rate_limit("script", request, *SCRIPT_RATE_LIMIT)
    prompt = (payload.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Please enter a prompt.")
    target_minutes = payload.target_minutes if payload.target_minutes in DURATION_OPTIONS_MINUTES else 2

    try:
        script_result, parody_lines = await asyncio.gather(
            asyncio.to_thread(generate_script_from_prompt, prompt, target_minutes),
            asyncio.to_thread(generate_parody_story, prompt),
        )
    except Exception as e:
        print(f"Script generation error: {e}")
        traceback.print_exc()
        msg = str(e)
        if "api_key" in msg or "log in" in msg or "token" in msg.lower():
            raise HTTPException(status_code=502, detail="HF_TOKEN not configured. Add it in Space Settings.")
        if "402" in msg or "Payment Required" in msg:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Hugging Face inference credits are exhausted for this account. "
                    "Check huggingface.co/settings/billing or huggingface.co/settings/inference-providers."
                ),
            )
        raise HTTPException(status_code=502, detail=f"Error: {msg[:200]}")

    turns, detected, title, voice_picks = script_result
    if not turns:
        raise HTTPException(status_code=422, detail="Empty result — try a more descriptive prompt.")

    voices = list(voice_picks)[:4]
    while len(voices) < 4:
        voices.append(None)

    return {
        "turns": turns,
        "num_speakers": detected,
        "title": title,
        "voices": voices,
        "parody_lines": parody_lines,
    }


class GenerateRequest(BaseModel):
    model: str
    num_speakers: int
    turns: list[dict]
    speakers: list[str | None]
    cfg_scale: float
    custom_audio: list[str | None] = [None, None, None, None]  # base64 (or data: URI), one per slot
    voice_consent: bool = False


@app.post("/api/generate")
async def api_generate(payload: GenerateRequest, request: Request) -> StreamingResponse:
    _enforce_rate_limit("audio", request, *AUDIO_RATE_LIMIT)
    if remote_generate_function is None:
        raise HTTPException(status_code=503, detail="Modal backend is offline.")

    script = turns_to_script(payload.turns)
    if not script.strip():
        raise HTTPException(status_code=400, detail="Add dialogue before generating.")

    word_count = len(script.split())
    if word_count > MAX_SCRIPT_WORDS:
        raise HTTPException(
            status_code=400,
            detail=f"Script too long: {word_count} words (max {MAX_SCRIPT_WORDS}). Shorten some turns.",
        )

    speakers = (list(payload.speakers) + [None, None, None, None])[:4]
    custom_audio_raw = (list(payload.custom_audio) + [None, None, None, None])[:4]
    custom_audio = [_decode_custom_audio(v, i) for i, v in enumerate(custom_audio_raw)]
    if any(a is not None for a in custom_audio) and not payload.voice_consent:
        raise HTTPException(
            status_code=400,
            detail="Confirm you have the right to use each uploaded voice before generating.",
        )

    # Only pass custom_audio kwargs when a clone is actually in use, so preset-voice
    # generations still work against a deployed backend that predates the parameters.
    custom_audio_kwargs = (
        {f"custom_audio_{i + 1}": a for i, a in enumerate(custom_audio)}
        if any(a is not None for a in custom_audio)
        else {}
    )

    async def event_stream():
        _prune_audio_store()

        async with GENERATION_CONCURRENCY:
            loop = asyncio.get_event_loop()
            q: asyncio.Queue = asyncio.Queue()
            sentinel = object()

            def worker():
                try:
                    for update in remote_generate_function.remote_gen(
                        num_speakers=payload.num_speakers,
                        script=script,
                        speaker_1=speakers[0],
                        speaker_2=speakers[1],
                        speaker_3=speakers[2],
                        speaker_4=speakers[3],
                        **custom_audio_kwargs,
                        cfg_scale=payload.cfg_scale,
                        model_name=payload.model,
                    ):
                        loop.call_soon_threadsafe(q.put_nowait, update)
                except Exception as e:
                    loop.call_soon_threadsafe(q.put_nowait, {
                        "stage": "error",
                        "status": "Inference failed.",
                        "log": f"{e}\n\n{traceback.format_exc()}",
                    })
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, sentinel)

            threading.Thread(target=worker, daemon=True).start()

            while True:
                item = await q.get()
                if item is sentinel:
                    break
                if not item:
                    yield ": keep-alive\n\n"
                    continue

                event = dict(item)
                audio_payload = event.pop("audio", None)
                if audio_payload is not None:
                    sample_rate, audio_array = audio_payload
                    wav_bytes = _encode_wav(sample_rate, audio_array)
                    audio_id = uuid.uuid4().hex
                    AUDIO_STORE[audio_id] = (time.time(), wav_bytes)
                    event["audio_id"] = audio_id
                    event["audio_duration"] = len(audio_array) / float(sample_rate)
                    LAST_AUDIO.clear()
                    LAST_AUDIO.update(
                        audio_id=audio_id,
                        created=time.time(),
                        duration=event["audio_duration"],
                    )
                yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/last-take")
async def api_last_take() -> dict:
    """The newest finished take, so a lost page can recover its render."""
    if not LAST_AUDIO or LAST_AUDIO.get("audio_id") not in AUDIO_STORE:
        raise HTTPException(status_code=404, detail="No recent take.")
    return {
        "audio_id": LAST_AUDIO["audio_id"],
        "age_seconds": round(time.time() - LAST_AUDIO["created"]),
        "duration": LAST_AUDIO["duration"],
    }


MP3_CACHE: dict[str, bytes] = {}  # audio_id -> encoded mp3, built lazily on first request

# --- Post-processing (the "polish" controls act on the exported file) ---
TONE_SHELVES = {                    # (low-shelf dB, high-shelf dB)
    "neutral": (0.0, 0.0),
    "warm": (4.0, -3.0),
    "bright": (-2.0, 4.5),
}
POLISH_BLOCK_SECONDS = 30           # bounds memory regardless of take length


def _shelf_sos(kind: str, f0: float, gain_db: float, fs: float) -> np.ndarray:
    """RBJ cookbook shelving biquad, as a single second-order section."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt(2.0)
    sqrtA2alpha = 2.0 * np.sqrt(A) * alpha
    if kind == "low":
        b = [A * ((A + 1) - (A - 1) * cos_w0 + sqrtA2alpha),
             2 * A * ((A - 1) - (A + 1) * cos_w0),
             A * ((A + 1) - (A - 1) * cos_w0 - sqrtA2alpha)]
        a = [(A + 1) + (A - 1) * cos_w0 + sqrtA2alpha,
             -2 * ((A - 1) + (A + 1) * cos_w0),
             (A + 1) + (A - 1) * cos_w0 - sqrtA2alpha]
    else:
        b = [A * ((A + 1) + (A - 1) * cos_w0 + sqrtA2alpha),
             -2 * A * ((A - 1) + (A + 1) * cos_w0),
             A * ((A + 1) + (A - 1) * cos_w0 - sqrtA2alpha)]
        a = [(A + 1) - (A - 1) * cos_w0 + sqrtA2alpha,
             2 * ((A - 1) - (A + 1) * cos_w0),
             (A + 1) - (A - 1) * cos_w0 - sqrtA2alpha]
    return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])


def _time_stretch(x: np.ndarray, rate: float) -> np.ndarray:
    """WSOLA time-scaling: rate > 1 shortens (faster), < 1 lengthens (slower).

    Pitch is preserved — the point of the exercise, since plain resampling
    would turn a slowed voice into a drawl an octave down.
    """
    if abs(rate - 1.0) < 1e-3 or len(x) < 4096:
        return x
    N, search = 1024, 128
    Hs = N // 2
    Ha = max(1, int(round(Hs * rate)))
    win = np.hanning(N).astype(np.float32)
    frames = max(1, int((len(x) - N - search) / Ha))
    out = np.zeros(frames * Hs + N, dtype=np.float32)
    norm = np.zeros_like(out)
    prev_tail = x[:N] * win
    for i in range(frames):
        want = i * Ha
        lo = max(0, want - search)
        hi = min(len(x) - N, want + search)
        if hi <= lo:
            seg_start = min(max(0, want), max(0, len(x) - N))
        else:
            cand = x[lo:hi + N]
            # pick the offset whose overlap best matches the previous tail
            windows = np.lib.stride_tricks.sliding_window_view(cand, N)[: hi - lo + 1]
            scores = windows[:, :Hs] @ prev_tail[Hs:]
            seg_start = lo + int(np.argmax(scores))
        seg = x[seg_start:seg_start + N]
        if len(seg) < N:
            break
        seg = seg * win
        o = i * Hs
        out[o:o + N] += seg
        norm[o:o + N] += win
        prev_tail = seg
    np.maximum(norm, 1e-6, out=norm)
    return out / norm


def _polish_wav(wav_bytes: bytes, speed: float, tone: str, level: bool) -> bytes:
    """Apply the polish settings to a stored take and return a new WAV.

    Processed in blocks so a five-hour render doesn't need gigabytes at once.
    """
    from scipy.signal import sosfilt, sosfilt_zi

    sample_rate, samples = wavfile.read(io.BytesIO(wav_bytes))
    if samples.ndim > 1:
        samples = samples[:, 0]
    low_db, high_db = TONE_SHELVES.get(tone, TONE_SHELVES["neutral"])
    sos = None
    if low_db or high_db:
        parts = []
        if low_db:
            parts.append(_shelf_sos("low", 250.0, low_db, sample_rate))
        if high_db:
            parts.append(_shelf_sos("high", 3500.0, high_db, sample_rate))
        sos = np.vstack(parts)
    zi = sosfilt_zi(sos) * 0.0 if sos is not None else None

    block = POLISH_BLOCK_SECONDS * sample_rate
    overlap = int(0.025 * sample_rate)  # crossfade between stretched blocks
    fade = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    pieces: list[np.ndarray] = []
    peak = 1e-6
    sq_sum = 0.0
    sq_n = 0
    for start in range(0, len(samples), block):
        chunk = samples[start:start + block].astype(np.float32) / 32768.0
        if sos is not None:
            chunk, zi = sosfilt(sos, chunk, zi=zi)
            chunk = chunk.astype(np.float32)
        chunk = _time_stretch(chunk, speed)
        if pieces and overlap and len(chunk) > overlap and len(pieces[-1]) > overlap:
            pieces[-1][-overlap:] = pieces[-1][-overlap:] * (1 - fade) + chunk[:overlap] * fade
            chunk = chunk[overlap:]
        if len(chunk):
            peak = max(peak, float(np.abs(chunk).max()))
            sq_sum += float(np.dot(chunk, chunk))
            sq_n += len(chunk)
        pieces.append(chunk)

    if level:
        # Lift quiet speech toward a target loudness; the limiter below only
        # touches peaks, so this raises level without squashing the whole take.
        # Boost-only: a take that is already at a healthy level is left alone
        # rather than pulled down to a target.
        rms = np.sqrt(sq_sum / max(1, sq_n))
        gain = float(min(6.0, max(1.0, 0.18 / max(rms, 1e-6))))
    else:
        gain = float(min(1.0, 0.99 / peak))  # only pull down if it would clip

    KNEE = 0.8
    out = bytearray()
    for chunk in pieces:
        y = chunk * gain
        if level:
            mag = np.abs(y)
            over = mag > KNEE
            if over.any():
                excess = (mag[over] - KNEE) / (1.0 - KNEE)
                y[over] = np.sign(y[over]) * (KNEE + (1.0 - KNEE) * np.tanh(excess))
        np.clip(y, -1.0, 1.0, out=y)
        out += (y * 32767.0).astype(np.int16).tobytes()

    header = io.BytesIO()
    wavfile.write(header, sample_rate, np.frombuffer(bytes(out), dtype=np.int16))
    return header.getvalue()


def _encode_mp3(wav_bytes: bytes) -> bytes:
    """Encode our PCM16 WAV to mono MP3 (96 kbps — transparent for 24kHz speech)."""
    import lameenc

    sample_rate, samples = wavfile.read(io.BytesIO(wav_bytes))
    if samples.ndim > 1:
        samples = samples[:, 0]
    if samples.dtype != np.int16:
        samples = np.asarray(samples, dtype=np.int16)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(96)
    encoder.set_in_sample_rate(int(sample_rate))
    encoder.set_channels(1)
    encoder.set_quality(2)
    mp3 = bytes(encoder.encode(samples.tobytes()))
    mp3 += bytes(encoder.flush())
    return mp3


@app.get("/api/audio/{audio_id}.mp3")
async def api_audio_mp3(audio_id: str) -> Response:
    entry = AUDIO_STORE.get(audio_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Audio not found or expired.")
    if audio_id not in MP3_CACHE:
        _, wav_bytes = entry
        # Encoding hours of audio takes minutes of CPU — keep the event loop free.
        MP3_CACHE[audio_id] = await asyncio.get_event_loop().run_in_executor(
            None, _encode_mp3, wav_bytes
        )
    return Response(
        content=MP3_CACHE[audio_id],
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'attachment; filename="conference.mp3"'},
    )


POLISH_CACHE: dict[tuple, bytes] = {}   # (audio_id, speed, tone, level, fmt) -> bytes


@app.get("/api/audio/{audio_id}/export")
async def api_audio_export(
    audio_id: str,
    speed: float = 1.0,
    tone: str = "neutral",
    level: bool = False,
    fmt: str = "wav",
) -> Response:
    """Download the take with the polish settings baked in."""
    entry = AUDIO_STORE.get(audio_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Audio not found or expired.")
    speed = round(min(1.25, max(0.8, speed)), 2)
    tone = tone if tone in TONE_SHELVES else "neutral"
    fmt = "mp3" if fmt == "mp3" else "wav"
    key = (audio_id, speed, tone, bool(level), fmt)
    if key not in POLISH_CACHE:
        _, wav_bytes = entry
        loop = asyncio.get_event_loop()
        processed = await loop.run_in_executor(
            None, _polish_wav, wav_bytes, speed, tone, bool(level)
        )
        if fmt == "mp3":
            processed = await loop.run_in_executor(None, _encode_mp3, processed)
        POLISH_CACHE.clear()          # one polished export at a time; these are large
        POLISH_CACHE[key] = processed
    data = POLISH_CACHE[key]
    return Response(
        content=data,
        media_type="audio/mpeg" if fmt == "mp3" else "audio/wav",
        headers={"Content-Disposition": f'attachment; filename="chorus-polished.{fmt}"'},
    )


@app.get("/api/audio/{audio_id}/peaks")
async def api_audio_peaks(audio_id: str, buckets: int = 2048) -> dict:
    """RMS envelope for the waveform display.

    Computed here from a strided sample of the PCM so the browser never has to
    pull hundreds of megabytes just to draw a picture. RMS (not per-bucket max)
    keeps its shape on multi-hour takes, where every max-bucket saturates.
    """
    entry = AUDIO_STORE.get(audio_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Audio not found or expired.")
    buckets = max(64, min(4096, buckets))
    _, wav_bytes = entry
    sample_rate, samples = wavfile.read(io.BytesIO(wav_bytes))
    if samples.ndim > 1:
        samples = samples[:, 0]
    total = len(samples)
    if total == 0:
        raise HTTPException(status_code=422, detail="Stored audio is empty.")
    # Cap the work: ~500 points per bucket is plenty for an envelope.
    stride = max(1, total // (buckets * 500))
    sub = samples[::stride].astype(np.float32) / 32768.0
    usable = (len(sub) // buckets) * buckets
    if usable < buckets:
        rms = np.full(buckets, float(np.sqrt(np.mean(np.square(sub)))) if len(sub) else 0.0)
    else:
        rms = np.sqrt(np.mean(np.square(sub[:usable].reshape(buckets, -1)), axis=1))
    peak = float(rms.max()) or 1.0
    return {
        "peaks": [round(float(v) / peak, 4) for v in rms],
        "duration": total / float(sample_rate),
    }


def _byte_range_response(data: bytes, request: Request, media_type: str) -> Response:
    """Serve bytes with Range support so the player can stream and seek.

    Without this the browser has to pull an entire multi-hundred-megabyte WAV
    before it can play a second of it, or seek anywhere in it.
    """
    total = len(data)
    headers = {"Accept-Ranges": "bytes"}
    raw_range = request.headers.get("range")
    match = re.match(r"bytes=(\d*)-(\d*)\s*$", raw_range or "")
    if not match or not (match.group(1) or match.group(2)):
        headers["Content-Length"] = str(total)
        return Response(content=data, media_type=media_type, headers=headers)

    if match.group(1):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else total - 1
    else:  # suffix form: bytes=-N means the final N bytes
        start = max(0, total - int(match.group(2)))
        end = total - 1
    if start >= total:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{total}", "Accept-Ranges": "bytes"},
        )
    end = min(end, total - 1)
    chunk = data[start:end + 1]
    headers.update({
        "Content-Range": f"bytes {start}-{end}/{total}",
        "Content-Length": str(len(chunk)),
    })
    return Response(content=chunk, status_code=206, media_type=media_type, headers=headers)


@app.get("/api/audio/{audio_id}")
async def api_audio(audio_id: str, request: Request) -> Response:
    entry = AUDIO_STORE.get(audio_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Audio not found or expired.")
    _, wav_bytes = entry
    return _byte_range_response(wav_bytes, request, "audio/wav")


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy" if remote_generate_function is not None else "backend offline"}


if __name__ == "__main__":
    import uvicorn

    if remote_generate_function is None:
        print("WARNING: Modal function not deployed — run `modal deploy backend_modal/modal_runner.py`.")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
