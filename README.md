<p align="center">
  <img src="public/images/banner.png" alt="VibeVoice Conference Generator" width="100%"/>
</p>

# Conference Generator — powered by VibeVoice

Generate realistic multi-speaker conference calls, meetings, and podcasts from a single text prompt. The app uses an LLM to write a natural-sounding script, then synthesizes long-form multi-speaker audio with Microsoft's [VibeVoice](https://huggingface.co/microsoft/VibeVoice-1.5B) model.

**Try it live:** [Hugging Face Space](https://huggingface.co/spaces/ACloudCenter/Conference-Generator-VibeVoice)

---

## Features

- **Prompt-to-audio in one step** — describe the scenario ("a 4-person product meeting about pricing") and get a full generated conversation
- **1–4 speakers** with 6 distinct voice presets (male/female tagged)
- **Up to ~90 minutes of continuous speech** via VibeVoice's long-form generation
- **Editable turn-by-turn script** — tweak speaker assignments or dialogue before rendering
- **Title generation** — the LLM names each script automatically
- **Two model sizes** — VibeVoice-1.5B (fast) and VibeVoice-7B (higher quality)
- **Gender-aware voice casting** — female characters get female voices automatically (Mom → Cherry, Wizard → Chicago, etc.) with one-click override
- **Voice preview** — sample any of the 6 voices before committing to a long generation

---

## Walkthrough

### 1. Describe your scenario

Type any scenario — a meeting, podcast, argument, TED talk — and the LLM writes the full script.

<p align="center">
  <img src="public/images/Screenshot-1.png" alt="Step 1: Prompt input" width="100%"/>
</p>

### 2. Review the script and pick voices

Speaker tags auto-assign by gender. Every voice dropdown stays in sync with the tags above. Preview any voice before generating.

<p align="center">
  <img src="public/images/Screenshot-2.png" alt="Step 2: Script editor with voice sync" width="100%"/>
</p>

### 3. Generate the audio

Kick off the GPU job on Modal. A funny parody narration keeps you entertained during the wait.

<p align="center">
  <img src="public/images/Screenshot-3.png" alt="Step 3: Generating" width="85%"/>
</p>

### 4. Listen and download

Full-length multi-speaker audio, ready to play or download as a WAV.

<p align="center">
  <img src="public/images/Screenshot-4.png" alt="Step 4: Complete" width="85%"/>
</p>

---

## Sample output

A 3-speaker example generated from the prompt _"A Wizard and Orc arguing about which spell is most powerful against dragons. Suddenly, their Mom comes downstairs to interrupt their LARPing session."_

<p align="center">
  <audio controls src="public/sample-generations/sample-generation-001.wav">
    Your viewer doesn't support inline audio —
    <a href="public/sample-generations/sample-generation-001.wav">download the sample WAV</a>.
  </audio>
</p>

▶️ **[Download / direct link](public/sample-generations/sample-generation-001.wav)** (for GitHub viewers where inline audio isn't supported)

Voices used: **Chicago (M)** as the Wizard, **Janus (M)** as the Orc, **Cherry (F)** as Mom.

---

## About VibeVoice

VibeVoice is Microsoft's open-source long-form, multi-speaker TTS model. It uses a frozen LLM backbone with acoustic + semantic tokenizers and a diffusion head to produce up to 90 minutes of natural conversational audio with up to 4 distinct speakers.

<p align="center">
  <img src="public/images/diagram.jpg" alt="VibeVoice architecture" width="85%"/>
</p>

<p align="center">
  <em>Speaker voice prompts and a plain text script are fed into the VibeVoice backbone, which streams audio chunks through per-turn diffusion heads.</em>
</p>

### Benchmark performance

<p align="center">
  <img src="public/images/chart.png" alt="VibeVoice benchmark comparison" width="75%"/>
</p>

VibeVoice leads on preference, realism, and richness among long-form multi-speaker TTS models.

---

## Architecture

This project separates the lightweight Gradio frontend (hosted on HF Spaces) from the GPU-heavy model backend (hosted on [Modal](https://modal.com)).

```
┌──────────────────────┐      ┌─────────────────────────┐
│  HF Space (Gradio)   │      │   Modal (GPU backend)   │
│  ─────────────────   │      │   ───────────────────   │
│  • Prompt UI         │ ───► │  • VibeVoice-1.5B / 7B  │
│  • Script editor     │      │  • Voice prompt loader  │
│  • Qwen2.5-Coder 32B │      │  • Long-form synthesis  │
│    (script writing)  │ ◄─── │  • Returns WAV bytes    │
└──────────────────────┘      └─────────────────────────┘
```

- **Frontend** (`app.py`): Gradio UI, script generation via HF Inference API (Qwen2.5-Coder-32B), script parsing, playback.
- **Backend** (`backend_modal/`, not included in this repo): deployed separately on Modal as a class-based GPU service exposing `generate_podcast`.

---

## Voices

| Voice     | Gender |
| --------- | :----: |
| Cherry    |   F    |
| Chicago   |   M    |
| Janus     |   M    |
| Mantis    |   F    |
| Sponge    |   M    |
| Starchild |   F    |

Voice samples live in `public/voices/` and are loaded as short reference clips by the VibeVoice backend.

---

## Running locally

```bash
git clone https://github.com/Josh-E-S/vibevoice-conference-generator.git
cd vibevoice-conference-generator
pip install -r requirements.txt

# Set your HF token (used for the script-writing LLM)
export HF_TOKEN=your_hf_token_here

# Deploy the Modal backend separately (not in this repo)
# modal deploy backend_modal/modal_runner.py

python app.py
```

Required env:

- `HF_TOKEN` — Hugging Face token with Inference API access

---

## Repo layout

```
.
├── app.py                # Gradio frontend + script generation
├── requirements.txt      # gradio, modal, huggingface_hub
├── public/
│   ├── images/              # Banner, architecture diagram, screenshots
│   ├── voices/              # Voice reference clips (Cherry, Chicago, ...)
│   └── sample-generations/  # Example WAV outputs
├── text_examples/           # Example scripts (1p, 2p, 3p, 4p scenarios)
├── tests/                   # Parser tests + example prompts
└── README.md
```

---

## Credits

- **[VibeVoice](https://github.com/microsoft/VibeVoice)** — Microsoft Research's long-form multi-speaker TTS model
- **[Qwen2.5-Coder-32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)** — script generation
- **[Modal](https://modal.com)** — GPU compute for inference
- **[Gradio](https://gradio.app)** + **[Hugging Face Spaces](https://huggingface.co/spaces)** — frontend hosting

---

`<sub>`HF Spaces configuration reference: https://huggingface.co/docs/hub/spaces-config-reference `</sub>`
