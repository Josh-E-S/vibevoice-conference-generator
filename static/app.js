const MAX_TURNS = 50;

const SCRIPT_GEN_MESSAGES = [
  "Writing script...", "Still generating...", "Making magic happen...",
  "Bossing around robot writers...", "Entering the matrix...", "Crafting dialogue...",
  "Teaching AI to be dramatic...", "Consulting the creative robots...", "Spilling digital ink...",
  "Herding AI cats into a script...", "Negotiating with the muse...", "Downloading inspiration...",
  "Warming up the plot engine...", "Shaking the idea tree...", "Feeding the word machine...",
  "Polishing virtual microphones...", "Rehearsing in the AI green room...",
  "Bribing the creativity daemon...", "Untangling narrative spaghetti...",
  "Summoning fictional characters...", "Tuning the dialogue generator...",
  "Spinning up the story factory...", "Convincing electrons to be eloquent...",
  "Wrangling syllables into sentences...", "Loading dramatic tension...",
  "Calibrating the sass levels...", "Generating witty banter...",
  "Overthinking your prompt (in a good way)...", "Adding a pinch of personality...",
  "Almost done, probably...", "Finalizing the masterpiece...",
];

const PRIMARY_STAGE_MESSAGES = {
  connecting: ["Submitted", "Provisioning GPU resources... cold starts can take up to a minute."],
  queued: ["Queued", "Worker is spinning up. Cold starts may take 30-60 seconds."],
  loading_model: ["Loading Model", "Streaming VibeVoice weights to the GPU."],
  loading_voices: ["Loading Voices", null],
  preparing_inputs: ["Preparing", "Formatting the conversation for the model."],
  generating_audio: ["Generating", "Synthesizing speech — this is the longest step."],
  processing_audio: ["Finalizing", "Converting tensors into a playable waveform."],
  complete: ["Complete", "Press play below or download your audio."],
  error: ["Error", "Check the log for details."],
};

const state = {
  turns: [],
  numSpeakers: 2,
  voices: [],
  voiceSelections: [null, null, null, null],
  models: [],
  examples: [],
  parodyLines: [],
  parodyIndex: 0,
  currentAudioId: null,
};

const el = {
  runtimeStatus: document.querySelector("#runtimeStatus"),
  runtimeLabel: document.querySelector("#runtimeLabel"),
  scriptPrompt: document.querySelector("#scriptPrompt"),
  generateScriptBtn: document.querySelector("#generateScriptBtn"),
  scriptGenStatus: document.querySelector("#scriptGenStatus"),
  examplePills: document.querySelector("#examplePills"),
  pastedScript: document.querySelector("#pastedScript"),
  scriptFileUpload: document.querySelector("#scriptFileUpload"),
  loadScriptBtn: document.querySelector("#loadScriptBtn"),
  scriptTitle: document.querySelector("#scriptTitle"),
  scriptDuration: document.querySelector("#scriptDuration"),
  turnsScroll: document.querySelector("#turnsScroll"),
  addTurnBtn: document.querySelector("#addTurnBtn"),
  modelSelect: document.querySelector("#modelSelect"),
  cfgScale: document.querySelector("#cfgScale"),
  cfgScaleValue: document.querySelector("#cfgScaleValue"),
  previewVoiceSelect: document.querySelector("#previewVoiceSelect"),
  previewAudio: document.querySelector("#previewAudio"),
  generateBtn: document.querySelector("#generateBtn"),
  primaryStatus: document.querySelector("#primaryStatus"),
  primaryStatusTitle: document.querySelector("#primaryStatusTitle"),
  primaryStatusDesc: document.querySelector("#primaryStatusDesc"),
  outputEmpty: document.querySelector("#outputEmpty"),
  outputResult: document.querySelector("#outputResult"),
  resultWaveform: document.querySelector("#resultWaveform"),
  resultAudio: document.querySelector("#resultAudio"),
  generationTime: document.querySelector("#generationTime"),
  audioDuration: document.querySelector("#audioDuration"),
  resultModel: document.querySelector("#resultModel"),
  downloadBtn: document.querySelector("#downloadBtn"),
  logBox: document.querySelector("#logBox"),
};

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

/* ---------------- Views ---------------- */
document.querySelectorAll(".view-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".view-tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll("#generate-view, #architecture-view").forEach((section) => {
      section.classList.toggle("visible", section.id === `${tab.dataset.view}-view`);
    });
  });
});

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

/* ---------------- Voice / model selects ---------------- */
function renderVoiceControls() {
  el.modelSelect.innerHTML = state.models.map((m) => `<option value="${m}">${m}</option>`).join("");

  document.querySelectorAll(".voice-select").forEach((select) => {
    const idx = Number(select.dataset.index);
    select.innerHTML = state.voices
      .map((v) => `<option value="${v.name}">${v.name} (${v.gender})</option>`)
      .join("");
    if (state.voiceSelections[idx]) select.value = state.voiceSelections[idx];
    select.closest(".voice-slot").hidden = idx >= state.numSpeakers;
  });

  el.previewVoiceSelect.innerHTML = state.voices.map((v) => `<option value="${v.name}">${v.name} (${v.gender})</option>`).join("");
  if (state.voices.length) {
    el.previewVoiceSelect.value = state.voices[0].name;
    el.previewAudio.src = state.voices[0].preview_url;
  }
}

document.querySelectorAll(".voice-select").forEach((select) => {
  select.addEventListener("change", () => {
    state.voiceSelections[Number(select.dataset.index)] = select.value;
    renderTurns();
  });
});

el.previewVoiceSelect.addEventListener("change", () => {
  const v = state.voices.find((x) => x.name === el.previewVoiceSelect.value);
  if (v) el.previewAudio.src = v.preview_url;
});

el.cfgScale.addEventListener("input", () => {
  el.cfgScaleValue.textContent = Number(el.cfgScale.value).toFixed(2);
});

/* ---------------- Turn editor ---------------- */
function speakerChoiceLabel(i) {
  const sel = state.voiceSelections[i];
  return sel ? `Speaker ${i + 1} - ${sel} (${genderOf(sel)})` : `Speaker ${i + 1}`;
}

function renderTurns() {
  el.turnsScroll.innerHTML = "";
  if (!state.turns.length) {
    const empty = document.createElement("div");
    empty.className = "empty-turns";
    empty.id = "emptyTurns";
    empty.innerHTML = "Your conversation will appear here.<br />Type a prompt above and click <strong>Write Script with AI</strong>, or pick an example to get started.";
    el.turnsScroll.append(empty);
    updateDuration();
    return;
  }

  state.turns.forEach((turn, idx) => {
    const spk = Math.min(4, Math.max(1, turn.speaker || 1));
    const row = document.createElement("div");
    row.className = `turn-row speaker-${spk}`;

    const spkSelect = document.createElement("select");
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

    const textArea = document.createElement("textarea");
    textArea.rows = 2;
    textArea.value = turn.text || "";
    textArea.addEventListener("input", () => {
      state.turns[idx].text = textArea.value;
      updateDuration();
    });

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "btn btn-danger";
    delBtn.textContent = "✕";
    delBtn.addEventListener("click", () => {
      state.turns.splice(idx, 1);
      renderTurns();
    });

    row.append(spkSelect, textArea, delBtn);
    el.turnsScroll.append(row);
  });
  updateDuration();
}

function updateDuration() {
  el.scriptDuration.textContent = estimateDuration(state.turns);
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
  el.scriptTitle.textContent = result.title || titleFallback || "Script";
  renderVoiceControls();
  renderTurns();
  el.outputEmpty.hidden = false;
  el.outputResult.classList.remove("visible");
}

/* ---------------- Examples ---------------- */
function renderExamplePills() {
  el.examplePills.innerHTML = "";
  state.examples.forEach((example) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-sm btn-pill";
    btn.textContent = example.title;
    btn.addEventListener("click", () => loadScriptResult(example, example.title));
    el.examplePills.append(btn);
  });
}

/* ---------------- Paste / upload ---------------- */
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
    loadScriptResult(payload, "Uploaded Script");
  } catch (error) {
    alert(error.message);
  }
});

/* ---------------- AI script generation ---------------- */
el.generateScriptBtn.addEventListener("click", async () => {
  const prompt = el.scriptPrompt.value.trim();
  if (!prompt) {
    alert("Please enter a prompt.");
    return;
  }

  el.generateScriptBtn.disabled = true;
  el.generateScriptBtn.textContent = "Writing...";
  el.loadScriptBtn.disabled = true;
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
      body: JSON.stringify({ prompt }),
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || "Script generation failed.");
    state.parodyLines = payload.parody_lines || [];
    state.parodyIndex = 0;
    loadScriptResult(payload, "Script");
    el.scriptGenStatus.textContent = "";
  } catch (error) {
    el.scriptGenStatus.textContent = error.message;
  } finally {
    window.clearInterval(ticker);
    el.generateScriptBtn.disabled = false;
    el.generateScriptBtn.textContent = "Write Script with AI";
    el.loadScriptBtn.disabled = false;
  }
});

/* ---------------- Waveform ---------------- */
async function drawWaveform(url, canvas, color = "#6366f1") {
  const response = await fetch(url);
  if (!response.ok) return;
  const data = await response.arrayBuffer();
  const context = new AudioContext();
  try {
    const buffer = await context.decodeAudioData(data.slice(0));
    const samples = buffer.getChannelData(0);
    const width = canvas.width;
    const height = canvas.height;
    const blocks = Math.min(160, Math.max(40, Math.floor(width / 6)));
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

/* ---------------- Generation status banner ---------------- */
function nextParodyLine() {
  if (!state.parodyLines.length) return null;
  const line = state.parodyLines[state.parodyIndex % state.parodyLines.length];
  state.parodyIndex += 1;
  return line;
}

function setPrimaryStatus(stage, fallbackText) {
  const [title, defaultDesc] = PRIMARY_STAGE_MESSAGES[stage] || ["Working", "Processing..."];
  el.primaryStatus.classList.add("visible");
  el.primaryStatus.classList.toggle("active", stage !== "complete" && stage !== "error");
  el.primaryStatus.classList.toggle("complete", stage === "complete");
  el.primaryStatus.classList.toggle("error", stage === "error");
  el.primaryStatusTitle.textContent = title;
  el.primaryStatusDesc.textContent = fallbackText || defaultDesc || "";
}

/* ---------------- Generate ---------------- */
el.generateBtn.addEventListener("click", async () => {
  const script = state.turns.map((t) => (t.text || "").trim()).filter(Boolean).join(" ");
  if (!script) {
    alert("Add dialogue before generating.");
    return;
  }

  el.generateBtn.disabled = true;
  el.generateBtn.textContent = "Generating...";
  el.generateScriptBtn.disabled = true;
  el.outputResult.classList.remove("visible");
  el.outputEmpty.hidden = false;
  el.logBox.textContent = "";
  const started = performance.now();
  setPrimaryStatus("connecting", nextParodyLine() || "Provisioning GPU resources...");

  const payload = {
    model: el.modelSelect.value,
    num_speakers: state.numSpeakers,
    turns: state.turns,
    speakers: state.voiceSelections,
    cfg_scale: Number(el.cfgScale.value),
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
        setPrimaryStatus(evt.stage, displayLine);
        if (evt.log) el.logBox.textContent = evt.log;

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
          el.outputEmpty.hidden = true;
          el.outputResult.classList.add("visible");
        }
      }
    }
  } catch (error) {
    setPrimaryStatus("error", error.message);
  } finally {
    el.generateBtn.disabled = false;
    el.generateBtn.textContent = "Generate Conference Audio";
    el.generateScriptBtn.disabled = false;
  }
});

/* ---------------- Init ---------------- */
async function init() {
  const [models, voices, examples] = await Promise.all([
    fetch("/api/models").then((r) => r.json()),
    fetch("/api/voices").then((r) => r.json()),
    fetch("/api/examples").then((r) => r.json()),
  ]);
  state.models = models;
  state.voices = voices;
  state.examples = examples;
  state.voiceSelections = voices.slice(0, 4).map((v) => v.name);
  while (state.voiceSelections.length < 4) state.voiceSelections.push(null);

  renderVoiceControls();
  renderTurns();
  renderExamplePills();
  updateStatus();
  window.setInterval(updateStatus, 8000);
}

init().catch((error) => {
  el.scriptGenStatus.textContent = `Failed to load app data: ${error.message}`;
});
