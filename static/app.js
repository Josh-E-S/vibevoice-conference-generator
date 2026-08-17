const MAX_TURNS = 50;

const SCRIPT_GEN_MESSAGES = [
  "Writing script...", "Still generating...", "Making magic happen...",
  "Bossing around robot writers...", "Entering the matrix...", "Crafting dialogue...",
  "Teaching AI to be dramatic...", "Consulting the creative robots...", "Spilling digital ink...",
  "Herding AI cats into a script...", "Negotiating with the muse...", "Downloading inspiration...",
  "Warming up the plot engine...", "Shaking the idea tree...", "Feeding the word machine...",
  "Rehearsing in the green room...", "Untangling narrative spaghetti...",
  "Summoning fictional characters...", "Tuning the dialogue generator...",
  "Loading dramatic tension...", "Generating witty banter...", "Almost done, probably...",
  "Finalizing the masterpiece...",
];

const PRIMARY_STAGE_MESSAGES = {
  connecting: ["Submitted", "Provisioning GPU resources... cold starts can take up to a minute."],
  queued: ["Queued", "Worker is spinning up. Cold starts may take 30-60 seconds."],
  loading_model: ["Loading model", "Streaming VibeVoice weights to the GPU."],
  loading_voices: ["Loading voices", null],
  preparing_inputs: ["Preparing", "Formatting the conversation for the model."],
  generating_audio: ["Generating", "Synthesizing speech — this is the longest step."],
  processing_audio: ["Finalizing", "Converting tensors into a playable waveform."],
  complete: ["Complete", "Press play, or download the WAV."],
  error: ["Error", "Check the log for details."],
};

const state = {
  turns: [],
  numSpeakers: 2,
  voices: [],
  voiceSelections: [null, null, null, null],
  customVoiceFiles: [null, null, null, null],
  models: [],
  examples: [],
  parodyLines: [],
  parodyIndex: 0,
  previewAudio: new Audio(),
  playingVoice: null,
};

const el = {};
[
  "runtimeStatus", "runtimeLabel", "aboutBtn", "aboutDialog", "closeAboutBtn",
  "modelSelect", "speakerStepper", "voiceRows", "cfgScale", "cfgScaleValue",
  "voiceConsentRow", "voiceConsentCheckbox",
  "scriptPrompt", "durationSelect", "generateScriptBtn", "examplePills", "openImportBtn", "scriptGenStatus",
  "scriptTitle", "scriptDuration", "turnsList", "addTurnBtn",
  "generateBarMeta", "generateBtn",
  "statusCard", "statusTitle", "statusDesc",
  "dockEmpty", "resultBlock", "resultWaveform", "resultAudio",
  "generationTime", "audioDuration", "resultModel", "downloadBtn",
  "logToggleBtn", "logBox",
  "importDialog", "pastedScript", "scriptFileUpload", "cancelImportBtn", "loadScriptBtn",
].forEach((id) => { el[id] = document.getElementById(id); });

function autoGrow(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${textarea.scrollHeight}px`;
}

function genderOf(name) {
  const v = state.voices.find((x) => x.name === name);
  return v ? v.gender : "?";
}

function estimateDuration(turns) {
  const totalWords = turns.reduce((sum, t) => sum + (t.text || "").trim().split(/\s+/).filter(Boolean).length, 0);
  if (totalWords === 0) return "";
  const minutes = totalWords / 150;
  return minutes < 1 ? `~${Math.round(minutes * 60)}s` : `~${minutes.toFixed(1)}m`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "--";
  return seconds < 60 ? `${seconds.toFixed(1)}s` : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

/* ---------------- About dialog ---------------- */
el.aboutBtn.addEventListener("click", () => el.aboutDialog.showModal());
el.closeAboutBtn.addEventListener("click", () => el.aboutDialog.close());
el.aboutDialog.addEventListener("click", (e) => { if (e.target === el.aboutDialog) el.aboutDialog.close(); });

/* ---------------- Import dialog ---------------- */
el.openImportBtn.addEventListener("click", () => el.importDialog.showModal());
el.cancelImportBtn.addEventListener("click", () => el.importDialog.close());
el.importDialog.addEventListener("click", (e) => { if (e.target === el.importDialog) el.importDialog.close(); });

/* ---------------- Status polling ---------------- */
async function updateStatus() {
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    const payload = await res.json();
    const ready = payload.backend === "ready";
    el.runtimeStatus.classList.toggle("ready", ready);
    el.runtimeLabel.textContent = ready ? "Model ready" : "Backend offline";
  } catch {
    el.runtimeStatus.classList.remove("ready");
    el.runtimeLabel.textContent = "Connecting";
  }
}

/* ---------------- Voice preview ---------------- */
function playVoicePreview(name, button) {
  if (state.playingVoice === name) {
    state.previewAudio.pause();
    return;
  }
  const voice = state.voices.find((v) => v.name === name);
  if (!voice) return;
  state.previewAudio.src = voice.preview_url;
  state.playingVoice = name;
  document.querySelectorAll(".voice-play").forEach((b) => b.classList.toggle("playing", b === button));
  state.previewAudio.play().catch((err) => {
    state.playingVoice = null;
    document.querySelectorAll(".voice-play").forEach((b) => b.classList.remove("playing"));
    console.error("Voice preview failed to play:", err);
  });
}
state.previewAudio.addEventListener("ended", () => {
  state.playingVoice = null;
  document.querySelectorAll(".voice-play").forEach((b) => b.classList.remove("playing"));
});

/* ---------------- Sidebar: model / speakers / voices ---------------- */
const CUSTOM_VOICE_VALUE = "__custom__";
const MAX_CUSTOM_AUDIO_BYTES = 15 * 1024 * 1024;

function isCustomVoice(i) {
  return state.voiceSelections[i] === CUSTOM_VOICE_VALUE;
}

function anyCustomVoiceActive() {
  return Array.from({ length: state.numSpeakers }, (_, i) => i).some(isCustomVoice);
}

function updateVoiceConsentVisibility() {
  el.voiceConsentRow.hidden = !anyCustomVoiceActive();
}

function renderSidebar() {
  el.modelSelect.innerHTML = state.models.map((m) => `<option value="${m}">${m}</option>`).join("");

  el.speakerStepper.querySelectorAll("button").forEach((btn) => {
    btn.classList.toggle("active", Number(btn.dataset.count) === state.numSpeakers);
  });

  el.voiceRows.innerHTML = "";
  for (let i = 0; i < state.numSpeakers; i += 1) {
    const row = document.createElement("div");
    row.className = "voice-row";

    const dot = document.createElement("span");
    dot.className = "voice-dot";
    dot.style.background = `var(--speaker-${i + 1})`;

    const select = document.createElement("select");
    select.innerHTML =
      `<option value="${CUSTOM_VOICE_VALUE}">🎙️ Clone a voice…</option>` +
      state.voices.map((v) => `<option value="${v.name}">${v.name} (${v.gender})</option>`).join("");
    if (state.voiceSelections[i]) select.value = state.voiceSelections[i];

    const playBtn = document.createElement("button");
    playBtn.type = "button";
    playBtn.className = "voice-play";
    playBtn.textContent = "▶";
    playBtn.title = "Preview voice";
    playBtn.addEventListener("click", () => playVoicePreview(select.value, playBtn));

    select.addEventListener("change", () => {
      state.voiceSelections[i] = select.value;
      if (select.value !== CUSTOM_VOICE_VALUE) state.customVoiceFiles[i] = null;
      playBtn.textContent = "▶";
      renderSidebar();
      renderTurns();
    });

    row.append(dot, select, playBtn);
    el.voiceRows.append(row);

    if (isCustomVoice(i)) {
      playBtn.hidden = true;
      const uploadRow = document.createElement("div");
      uploadRow.className = "custom-voice-row";

      const fileLabel = document.createElement("label");
      fileLabel.className = "btn btn-sm upload-mini-btn";
      fileLabel.textContent = state.customVoiceFiles[i] ? state.customVoiceFiles[i].name : "Choose audio file…";
      const fileInput = document.createElement("input");
      fileInput.type = "file";
      fileInput.accept = "audio/*";
      fileInput.hidden = true;
      fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        if (!file) return;
        if (file.size > MAX_CUSTOM_AUDIO_BYTES) {
          alert(`That file is too large (max ${MAX_CUSTOM_AUDIO_BYTES / (1024 * 1024)} MB).`);
          fileInput.value = "";
          return;
        }
        state.customVoiceFiles[i] = file;
        renderSidebar();
      });
      fileLabel.append(fileInput);
      uploadRow.append(fileLabel);

      if (state.customVoiceFiles[i]) {
        const clearBtn = document.createElement("button");
        clearBtn.type = "button";
        clearBtn.className = "btn btn-sm btn-icon-only";
        clearBtn.textContent = "✕";
        clearBtn.title = "Remove file";
        clearBtn.addEventListener("click", () => {
          state.customVoiceFiles[i] = null;
          renderSidebar();
        });
        uploadRow.append(clearBtn);
      }

      el.voiceRows.append(uploadRow);
    }
  }

  updateVoiceConsentVisibility();
}

el.speakerStepper.querySelectorAll("button").forEach((btn) => {
  btn.addEventListener("click", () => {
    state.numSpeakers = Number(btn.dataset.count);
    renderSidebar();
  });
});

el.cfgScale.addEventListener("input", () => {
  el.cfgScaleValue.textContent = Number(el.cfgScale.value).toFixed(2);
});

/* ---------------- Turn editor ---------------- */
function speakerChoiceLabel(i) {
  const sel = state.voiceSelections[i];
  if (sel === CUSTOM_VOICE_VALUE) return `Speaker ${i + 1} · Custom voice`;
  return sel ? `Speaker ${i + 1} · ${sel} (${genderOf(sel)})` : `Speaker ${i + 1}`;
}

function renderTurns() {
  el.turnsList.innerHTML = "";
  if (!state.turns.length) {
    const empty = document.createElement("div");
    empty.className = "empty-transcript";
    empty.id = "emptyTurns";
    empty.innerHTML = "Nothing here yet. Type a scenario above and click <strong>Write with AI</strong>, or start typing your own line below.";
    el.turnsList.append(empty);
    updateMeta();
    return;
  }

  state.turns.forEach((turn, idx) => {
    const spk = Math.min(4, Math.max(1, turn.speaker || 1));
    const card = document.createElement("div");
    card.className = "turn-card";
    card.dataset.speaker = String(spk);

    const head = document.createElement("div");
    head.className = "turn-head";

    const spkSelect = document.createElement("select");
    spkSelect.className = "turn-speaker-select";
    for (let i = 1; i <= 4; i += 1) {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = speakerChoiceLabel(i - 1);
      if (i === spk) opt.selected = true;
      spkSelect.append(opt);
    }
    spkSelect.addEventListener("change", () => {
      state.turns[idx].speaker = Number(spkSelect.value);
      renderTurns();
    });

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "turn-remove";
    removeBtn.textContent = "✕";
    removeBtn.addEventListener("click", () => {
      state.turns.splice(idx, 1);
      renderTurns();
    });

    head.append(spkSelect, removeBtn);

    const textArea = document.createElement("textarea");
    textArea.rows = 1;
    textArea.value = turn.text || "";
    textArea.addEventListener("input", () => {
      state.turns[idx].text = textArea.value;
      autoGrow(textArea);
      updateMeta();
    });

    card.append(head, textArea);
    el.turnsList.append(card);
    requestAnimationFrame(() => autoGrow(textArea));
  });
  updateMeta();
}

function updateMeta() {
  const duration = estimateDuration(state.turns);
  el.scriptDuration.textContent = duration;
  el.generateBarMeta.textContent = state.turns.length
    ? `${state.turns.length} line${state.turns.length === 1 ? "" : "s"} · ${duration || "—"}`
    : "Add dialogue to begin";
}

el.addTurnBtn.addEventListener("click", () => {
  if (state.turns.length >= MAX_TURNS) {
    alert(`Maximum ${MAX_TURNS} turns reached.`);
    return;
  }
  let nextSpeaker = 1;
  if (state.turns.length) {
    const maxSpk = Math.max(...state.turns.map((t) => t.speaker));
    const last = state.turns[state.turns.length - 1].speaker;
    nextSpeaker = (last % maxSpk) + 1;
  }
  state.turns.push({ speaker: nextSpeaker, text: "" });
  renderTurns();
});

function loadScriptResult(result, titleFallback) {
  state.turns = result.turns;
  state.numSpeakers = result.num_speakers;
  const voices = (result.voices || []).slice(0, 4);
  while (voices.length < 4) voices.push(null);
  state.voiceSelections = voices;
  el.scriptTitle.textContent = result.title || titleFallback || "Untitled conversation";
  renderSidebar();
  renderTurns();
  el.dockEmpty.hidden = false;
  el.resultBlock.classList.remove("visible");
  el.statusCard.classList.remove("visible");
}

/* ---------------- Examples ---------------- */
function renderExamplePills() {
  el.examplePills.innerHTML = "";
  state.examples.forEach((example) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
    btn.textContent = example.title;
    btn.addEventListener("click", () => loadScriptResult(example, example.title));
    el.examplePills.append(btn);
  });
}

/* ---------------- Import ---------------- */
el.loadScriptBtn.addEventListener("click", async () => {
  const text = el.pastedScript.value.trim();
  const file = el.scriptFileUpload.files[0];
  if (!text && !file) {
    alert("Paste a script or upload a .txt file first.");
    return;
  }
  const formData = new FormData();
  formData.set("text", text);
  if (file) formData.set("file", file);

  try {
    const res = await fetch("/api/parse-script", { method: "POST", body: formData });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || "Could not parse script.");
    if (payload.over_limit) {
      alert(`Script is ${payload.word_count} words; loading anyway — trim before generating.`);
    }
    loadScriptResult(payload, "Uploaded script");
    el.importDialog.close();
    el.pastedScript.value = "";
    el.scriptFileUpload.value = "";
  } catch (error) {
    alert(error.message);
  }
});

/* ---------------- AI script generation ---------------- */
el.generateScriptBtn.addEventListener("click", async () => {
  const prompt = el.scriptPrompt.value.trim();
  if (!prompt) {
    alert("Describe a scenario first.");
    return;
  }

  el.generateScriptBtn.disabled = true;
  el.generateScriptBtn.textContent = "Writing...";
  let msgIdx = 0;
  el.scriptGenStatus.textContent = SCRIPT_GEN_MESSAGES[0];
  const ticker = window.setInterval(() => {
    msgIdx += 1;
    el.scriptGenStatus.textContent = SCRIPT_GEN_MESSAGES[msgIdx % SCRIPT_GEN_MESSAGES.length];
  }, 3000);

  try {
    const res = await fetch("/api/generate-script", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, target_minutes: Number(el.durationSelect.value) }),
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || "Script generation failed.");
    state.parodyLines = payload.parody_lines || [];
    state.parodyIndex = 0;
    loadScriptResult(payload, "Untitled conversation");
    el.scriptGenStatus.textContent = "";
  } catch (error) {
    el.scriptGenStatus.textContent = error.message;
  } finally {
    window.clearInterval(ticker);
    el.generateScriptBtn.disabled = false;
    el.generateScriptBtn.textContent = "Write with AI";
  }
});

/* ---------------- Waveform ---------------- */
async function drawWaveform(url, canvas, color = "#b5502e") {
  const response = await fetch(url);
  if (!response.ok) return;
  const data = await response.arrayBuffer();
  const context = new AudioContext();
  try {
    const buffer = await context.decodeAudioData(data.slice(0));
    const samples = buffer.getChannelData(0);
    const width = canvas.width;
    const height = canvas.height;
    const blocks = Math.min(110, Math.max(30, Math.floor(width / 6)));
    const blockSize = Math.max(1, Math.floor(samples.length / blocks));
    const peaks = [];
    for (let block = 0; block < blocks; block += 1) {
      let peak = 0;
      const start = block * blockSize;
      const end = Math.min(samples.length, start + blockSize);
      for (let i = start; i < end; i += 1) peak = Math.max(peak, Math.abs(samples[i]));
      peaks.push(peak);
    }
    const maxPeak = Math.max(...peaks, 0.001);
    const draw = canvas.getContext("2d");
    draw.clearRect(0, 0, width, height);
    draw.fillStyle = color;
    const barWidth = Math.max(2, width / blocks - 2);
    peaks.forEach((peak, i) => {
      const normalized = peak / maxPeak;
      const barHeight = Math.max(3, normalized * height * 0.9);
      const x = i * (width / blocks) + 1;
      draw.fillRect(x, (height - barHeight) / 2, barWidth, barHeight);
    });
  } finally {
    await context.close();
  }
}

/* ---------------- Generation status ---------------- */
function nextParodyLine() {
  if (!state.parodyLines.length) return null;
  const line = state.parodyLines[state.parodyIndex % state.parodyLines.length];
  state.parodyIndex += 1;
  return line;
}

function setStatus(stage, fallbackText) {
  const [title, defaultDesc] = PRIMARY_STAGE_MESSAGES[stage] || ["Working", "Processing..."];
  el.statusCard.classList.add("visible");
  el.statusCard.classList.toggle("active", stage !== "complete" && stage !== "error");
  el.statusCard.classList.toggle("complete", stage === "complete");
  el.statusCard.classList.toggle("error", stage === "error");
  el.statusTitle.textContent = title;
  el.statusDesc.textContent = fallbackText || defaultDesc || "";
}

el.logToggleBtn.addEventListener("click", () => {
  const visible = el.logBox.classList.toggle("visible");
  el.logToggleBtn.textContent = visible ? "Hide generation log" : "View generation log";
});

/* ---------------- Generate ---------------- */
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1] || "");
    reader.onerror = () => reject(new Error(`Could not read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

el.generateBtn.addEventListener("click", async () => {
  const script = state.turns.map((t) => (t.text || "").trim()).filter(Boolean).join(" ");
  if (!script) {
    alert("Add dialogue before generating.");
    return;
  }

  for (let i = 0; i < state.numSpeakers; i += 1) {
    if (isCustomVoice(i) && !state.customVoiceFiles[i]) {
      alert(`Upload a voice clip for Speaker ${i + 1}, or pick a preset voice instead.`);
      return;
    }
  }
  if (anyCustomVoiceActive() && !el.voiceConsentCheckbox.checked) {
    alert("Confirm you have the right to use each uploaded voice before generating.");
    return;
  }

  el.generateBtn.disabled = true;
  el.generateBtn.textContent = "Generating...";
  el.resultBlock.classList.remove("visible");
  el.dockEmpty.hidden = false;
  el.logBox.textContent = "";
  el.logBox.classList.remove("visible");
  el.logToggleBtn.hidden = true;
  el.logToggleBtn.textContent = "View generation log";
  const started = performance.now();
  setStatus("connecting", nextParodyLine() || "Provisioning GPU resources...");

  let customAudio;
  try {
    customAudio = await Promise.all(
      Array.from({ length: 4 }, (_, i) =>
        i < state.numSpeakers && isCustomVoice(i) && state.customVoiceFiles[i]
          ? fileToBase64(state.customVoiceFiles[i])
          : Promise.resolve(null)
      )
    );
  } catch (error) {
    setStatus("error", error.message);
    el.generateBtn.disabled = false;
    el.generateBtn.textContent = "Generate Audio";
    return;
  }

  const payload = {
    model: el.modelSelect.value,
    num_speakers: state.numSpeakers,
    turns: state.turns,
    speakers: state.voiceSelections.map((v) => (v === CUSTOM_VOICE_VALUE ? null : v)),
    cfg_scale: Number(el.cfgScale.value),
    custom_audio: customAudio,
    voice_consent: el.voiceConsentCheckbox.checked,
  };

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || "The model could not generate audio.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let done = false;

    while (!done) {
      const chunk = await reader.read();
      done = chunk.done;
      if (chunk.value) buffer += decoder.decode(chunk.value, { stream: true });

      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        if (!rawEvent.startsWith("data: ")) continue;
        const evt = JSON.parse(rawEvent.slice(6));
        const isDone = evt.stage === "complete" || evt.stage === "error";
        const displayLine = isDone ? evt.status : nextParodyLine() || evt.status;
        setStatus(evt.stage, displayLine);
        if (evt.log) {
          el.logBox.textContent = evt.log;
          el.logToggleBtn.hidden = false;
        }

        if (evt.stage === "complete" && evt.audio_id) {
          const audioRes = await fetch(`/api/audio/${evt.audio_id}`);
          const blob = await audioRes.blob();
          const url = URL.createObjectURL(blob);
          el.resultAudio.src = url;
          el.downloadBtn.href = url;
          el.generationTime.textContent = formatDuration((performance.now() - started) / 1000);
          el.audioDuration.textContent = formatDuration(evt.audio_duration);
          el.resultModel.textContent = el.modelSelect.value;
          await drawWaveform(url, el.resultWaveform);
          el.dockEmpty.hidden = true;
          el.resultBlock.classList.add("visible");
        }
      }
    }
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    el.generateBtn.disabled = false;
    el.generateBtn.textContent = "Generate Audio";
  }
});

/* ---------------- Init ---------------- */
async function init() {
  const [models, voices, examples, durationOptions] = await Promise.all([
    fetch("/api/models").then((r) => r.json()),
    fetch("/api/voices").then((r) => r.json()),
    fetch("/api/examples").then((r) => r.json()),
    fetch("/api/duration-options").then((r) => r.json()),
  ]);
  state.models = models;
  state.voices = voices;
  state.examples = examples;
  state.voiceSelections = voices.slice(0, 4).map((v) => v.name);
  while (state.voiceSelections.length < 4) state.voiceSelections.push(null);

  el.durationSelect.innerHTML = durationOptions.map((m) => `<option value="${m}">${m} min</option>`).join("");
  const defaultDuration = durationOptions.includes(2) ? 2 : durationOptions[0];
  el.durationSelect.value = String(defaultDuration);

  renderSidebar();
  renderTurns();
  renderExamplePills();
  updateStatus();
  window.setInterval(updateStatus, 8000);
}

init().catch((error) => {
  el.scriptGenStatus.textContent = `Failed to load app data: ${error.message}`;
});
