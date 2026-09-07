---
title: Conference Generator VibeVoice
emoji: ⭐
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

<p align="center">
  <img src="public/images/banner.png" alt="VibeVoice Conference Generator" width="100%"/>
</p>

# Chorus — long-form multi-speaker audio, powered by VibeVoice

Turn a one-line prompt, or your own script, into a full multi-speaker conversation: meetings, podcasts, interviews, debates, talks. An LLM writes the script to a target length, and a GPU backend renders it with Microsoft's [VibeVoice](https://huggingface.co/microsoft/VibeVoice-1.5B) — chunked in parallel, seamed with crossfades, and quality-gated — so a two-hour take renders in about twenty minutes and starts playing while it's still rendering.

**Try it live:** [Hugging Face Space](https://huggingface.co/spaces/ACloudCenter/Conference-Generator-VibeVoice)

### Listen to the demo

https://github.com/user-attachments/assets/cfe5397f-7aad-4662-b0b8-e62d7546a9fb

_A 3-speaker example — Wizard, Orc, and Mom — generated from a single sentence prompt. Audio visualizer created separately._

---

## Features

**Writing**
- **Prompt-to-script** — describe the scenario ("a 4-person product meeting about pricing") and Qwen2.5-Coder-32B writes the full conversation, with a title
- **Target length** — pick 1 to 45 minutes; the script is extended in continuation rounds until it reaches the word budget
- **Bring your own script** — paste or upload text using `Speaker N:` tags or named characters
- **Turn editor** — reassign speakers or rewrite any line before rendering
- **Gender-aware casting** — characters get a matching voice automatically, with one-click override

**Voices**
- **16 preset voices** — twelve public-domain voices (pre-1923 recordings and LibriVox readers) plus four originals, each with a preview clip
- **Clone a voice** — record in the browser or upload a clip; the first 30 seconds become the reference. Clones are kept in your browser (IndexedDB) and never stored server-side
- **Two model sizes** — VibeVoice-1.5B (Fast) and VibeVoice-7B (Best)
- **Expressiveness** control (classifier-free guidance scale)

**Rendering**
- **Streams as it renders** — the first chunk plays within a minute or two of starting; the waveform grows as later chunks land
- **Real progress** — wave-by-wave progress, elapsed time, and an ETA calibrated from previous runs
- **Synced transcript** — every turn's start time is measured from the rendered audio, with per-speaker orbs that light up as they talk
- **Sound modes** — Original, Studio (normalized to −16 LUFS), Warm, Bright; optional spectral noise cleanup; playback speed
- **Download** as WAV or MP3, with the chosen sound mode baked in

---

## How it renders long takes

Rendering a long script in one pass makes VibeVoice drift in rate and quality. Chorus instead:

1. Splits the script at turn boundaries into ~200-word whole-turn chunks (long monologues split at sentences)
2. Renders chunks in batched waves on one A100 (12 at a time for 1.5B, 6 for 7B), backing off to smaller waves on out-of-memory
3. Runs a quality gate on every chunk — speaking-rate band, silence fraction, spectral flatness, and a replay detector for cloned voices — and re-rolls failures on a fresh seed
4. Crossfades chunks together (0.25 s seams) and measures each turn's onset from the audio for the synced transcript

Measured on production (A100-40GB, VibeVoice-1.5B):

| Take | Result |
| --- | --- |
| 46 min, 4 speakers | 4.46× realtime |
| 85 min | 5.47× realtime, 61 chunks, zero re-rolls |
| **121 min** | **5.96× realtime**, 88 chunks, zero re-rolls |
| **5 h 29 min, cloned voice** | **5.57× realtime**, 247 chunks, 3 re-rolled by the gate |

---

## Architecture

The lightweight FastAPI frontend (this repo, hosted as a Docker Space) is separate from the GPU backend (hosted on [Modal](https://modal.com)).

```
┌────────────────────────────────┐      ┌──────────────────────────────────┐
│  HF Space (FastAPI + static)   │      │   Modal (A100-40GB)              │
│  ────────────────────────────  │      │   ────────────────────────────   │
│  • Prompt → script (Qwen 32B)  │ ───► │  • VibeVoice-1.5B + 7B loaded    │
│  • Script parser / turn editor │      │  • Chunk → batched waves         │
│  • /api/generate SSE stream    │ ◄─── │  • Quality gate + re-roll        │
│  • Chunk relay, polish, MP3    │      │  • Streams chunks + progress     │
└────────────────────────────────┘      └──────────────────────────────────┘
```

- **Frontend** (`app.py` + `static/`): script generation via the HF Inference API, script parsing, an SSE endpoint that relays progress and streamed chunk audio from Modal, post-processing (tone shelves, loudness normalization, spectral denoise, time-stretch), MP3 encoding, and byte-range serving of finished takes. Takes live in memory for 15 minutes.
- **Backend** (`backend_modal/modal_runner.py`): a Modal class that loads both models at container start and exposes `generate_podcast` as a streaming generator. Deployed separately. The VibeVoice model code and reference voice WAVs alongside it are gitignored.
- **Scaling profiles**: the backend deploys with `VIBEVOICE_PROFILE=launch` (one container always warm, one buffer under load) or `tail` (scale to zero). Both cap at 4 concurrent GPUs; extra requests queue. The first generation after idle in `tail` mode takes ~3 minutes to load models, and the UI says so when the GPU is cold.
- **Limits** (public demo): scripts up to 7,000 words (~45 min), 3 generations per hour per IP, 60 per day across all visitors, 4 in flight globally. The backend has no length ceiling; the multi-hour records were rendered by calling it directly.

---

## About VibeVoice

VibeVoice is Microsoft's open-source long-form, multi-speaker TTS model. It uses a frozen LLM backbone with acoustic + semantic tokenizers at 7.5 Hz and a diffusion head to produce up to 90 minutes of conversational audio with up to 4 speakers in a single pass. Chorus's chunked pipeline lifts that ceiling.

<p align="center">
  <img src="public/images/diagram.jpg" alt="VibeVoice architecture" width="85%"/>
</p>

<p align="center">
  <img src="public/images/chart.png" alt="VibeVoice benchmark comparison" width="75%"/>
</p>

---

## Voices

| Voice | Gender | Character | Source |
| --- | :---: | --- | --- |
| Cylinder | M | Antique, 1900s | Public domain (pre-1923 recording) |
| Statesman | M | Historic, Orator | Public domain (pre-1923 recording) |
| Novella | F | Classic, Reader | Public domain (pre-1923 recording) |
| Eyre | F | Elegant, Literary | Public domain (pre-1923 recording) |
| Ishmael | M | Resonant, Epic | LibriVox |
| Traveller | M | Clear, Adventurer | LibriVox |
| Dublin | M | Irish, Lilting | LibriVox |
| Badger | M | Whimsical, Character | LibriVox |
| Meadow | F | Warm, Gentle | LibriVox |
| Abbey | F | British, Refined | LibriVox |
| Avonlea | F | Bright, Youthful | LibriVox |
| Sunshine | F | Cheery, Light | LibriVox |
| Cherry | F | Warm, Storyteller | Original preset |
| Chicago | M | Deep, Narrator | Original preset |
| Janus | M | Bright, Conversational | Original preset |
| Starchild | F | Airy, Dreamy | Original preset |

Preview clips live in `public/voices/`; the 60-second reference WAVs the backend conditions on live under the gitignored `backend_modal/voices/`.

---

## Running locally

```bash
git clone https://github.com/Josh-E-S/vibevoice-conference-generator.git
cd vibevoice-conference-generator
pip install -r requirements.txt

# Hugging Face token for the script-writing LLM (Inference API access)
export HF_TOKEN=your_hf_token_here

# Deploy the GPU backend separately (needs the gitignored model code + voices under backend_modal/)
# VIBEVOICE_PROFILE=tail modal deploy backend_modal/modal_runner.py

python app.py          # http://localhost:7860
```

Tests:

```bash
python -m pytest tests
```

Required env:

- `HF_TOKEN` — Hugging Face token with Inference API access. On the Space this is set in Settings → Secrets; nothing sensitive lives in the repo.

---

## Repo layout

```
.
├── app.py                 # FastAPI frontend: script generation, SSE relay, polish, exports
├── Dockerfile             # HF Docker Space image
├── requirements.txt
├── static/                # Hand-built UI (index.html, app.js, styles.css, vendored wavesurfer.js)
├── public/
│   ├── images/            # Banner, architecture diagram, benchmark chart
│   └── voices/            # Voice preview clips
├── text_examples/         # Example scripts (1–4 speakers)
├── tests/                 # Script-parser tests + example prompts
└── backend_modal/         # Modal runner (tracked); VibeVoice model code + reference voices (gitignored)
```

---

## Credits

- **[VibeVoice](https://github.com/microsoft/VibeVoice)** — Microsoft Research's long-form multi-speaker TTS model
- **[Qwen2.5-Coder-32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)** — script generation
- **[Modal](https://modal.com)** — GPU compute for inference
- **[FastAPI](https://fastapi.tiangolo.com)** + **[Hugging Face Spaces](https://huggingface.co/spaces)** (Docker SDK) — frontend hosting
- **[wavesurfer.js](https://wavesurfer.xyz)** — waveform player
- Public-domain voices via [LibriVox](https://librivox.org) and pre-1923 archival recordings
