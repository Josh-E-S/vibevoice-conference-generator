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

const QUALITY_LABELS = { "VibeVoice-1.5B": "Fast", "VibeVoice-7B": "Best" };
const GENDER_LABELS = { F: "Feminine", M: "Masculine" };
const SPEAKER_FALLBACK_COLORS = ["#e2582a", "#2f6f63", "#cc8a2e", "#7b4b94"];
const CUSTOM_VOICE_VALUE = "__custom__";
const MAX_CUSTOM_AUDIO_BYTES = 15 * 1024 * 1024;

const state = {
  turns: [],
  numSpeakers: 2,
  voices: [],
  voiceSelections: [null, null, null, null],
  customVoiceFiles: [null, null, null, null],
  models: [],
  model: null,
  examples: [],
  parodyLines: [],
  parodyIndex: 0,
  previewAudio: new Audio(),
  playingVoice: null,
  librarySearch: "",
  libraryFilter: "all",
  resultTurns: [],       // snapshot of turns for the synced transcript
  resultTitle: "",
  wavePeaks: null,
  activeSyncIndex: -1,
};

const el = {};
[
  "runtimeStatus", "runtimeLabel", "browseVoicesBtn", "aboutBtn", "aboutDialog", "closeAboutBtn",
  "speakerStepper", "voiceRows", "qualityPills", "cfgScale", "cfgScaleValue",
  "voiceConsentRow", "voiceConsentCheckbox",
  "scriptPrompt", "durationSelect", "generateScriptBtn", "examplePills", "openImportBtn", "scriptGenStatus",
  "scriptTitle", "scriptDuration", "turnsList", "addTurnBtn",
  "generateBarMeta", "generateBtn",
  "statusCard", "statusTitle", "statusDesc",
  "dockEmpty", "resultBlock", "resultWaveform", "resultAudio",
  "playBtn", "playerTime", "syncedTranscript", "openPlayerBtn",
  "composerCollapsedStrip", "collapsedSummary", "composerBody",
  "playerStage", "stageTitle", "stagePlayBtn", "stageWaveform", "stageTime",
  "stageDot", "stageLine", "stageSpeaker", "stageBackBtn", "stageDownloadBtn",
  "stageScriptToggle", "stageTranscript",
  "generationTime", "audioDuration", "resultModel", "downloadBtn",
  "logToggleBtn", "logBox",
  "voiceLibraryDialog", "closeLibraryBtn", "librarySearch", "libraryFilters", "libraryGrid",
  "importDialog", "pastedScript", "scriptFileUpload", "cancelImportBtn", "loadScriptBtn",
].forEach((id) => { el[id] = document.getElementById(id); });

function autoGrow(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${textarea.scrollHeight}px`;
}

function voiceByName(name) {
  return state.voices.find((v) => v.name === name) || null;
}

function isCustomVoice(i) {
  return state.voiceSelections[i] === CUSTOM_VOICE_VALUE;
}

function anyCustomVoiceActive() {
  return Array.from({ length: state.numSpeakers }, (_, i) => i).some(isCustomVoice);
}

function slotColor(i) {
  if (isCustomVoice(i)) return SPEAKER_FALLBACK_COLORS[i];
  const voice = voiceByName(state.voiceSelections[i]);
  return voice ? voice.color : SPEAKER_FALLBACK_COLORS[i];
}

function slotVoiceLabel(i) {
  if (isCustomVoice(i)) return "Custom voice";
  return state.voiceSelections[i] || `Voice ${i + 1}`;
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

function formatClock(seconds) {
  if (!Number.isFinite(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

/* ---------------- Dialogs ---------------- */
el.aboutBtn.addEventListener("click", () => el.aboutDialog.showModal());
el.closeAboutBtn.addEventListener("click", () => el.aboutDialog.close());
el.aboutDialog.addEventListener("click", (e) => { if (e.target === el.aboutDialog) el.aboutDialog.close(); });

el.openImportBtn.addEventListener("click", () => el.importDialog.showModal());
el.cancelImportBtn.addEventListener("click", () => el.importDialog.close());
el.importDialog.addEventListener("click", (e) => { if (e.target === el.importDialog) el.importDialog.close(); });

function openLibrary() {
  renderLibrary();
  el.voiceLibraryDialog.showModal();
}
el.browseVoicesBtn.addEventListener("click", openLibrary);
el.closeLibraryBtn.addEventListener("click", () => el.voiceLibraryDialog.close());
el.voiceLibraryDialog.addEventListener("click", (e) => {
  if (e.target === el.voiceLibraryDialog) el.voiceLibraryDialog.close();
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

/* ---------------- Voice preview ---------------- */
function refreshPreviewButtons() {
  document.querySelectorAll(".voice-play").forEach((b) => {
    const playing = b.dataset.voice === state.playingVoice;
    b.classList.toggle("playing", playing);
    b.textContent = playing ? "❚❚" : "▶";
  });
  document.querySelectorAll(".voice-preview-link").forEach((b) => {
    b.textContent = b.dataset.voice === state.playingVoice ? "Playing..." : "Preview";
  });
}

function playVoicePreview(name) {
  if (state.playingVoice === name) {
    state.previewAudio.pause();
    state.playingVoice = null;
    refreshPreviewButtons();
    return;
  }
  const voice = voiceByName(name);
  if (!voice) return;
  state.previewAudio.src = voice.preview_url;
  state.playingVoice = name;
  refreshPreviewButtons();
  state.previewAudio.play().catch((err) => {
    state.playingVoice = null;
    refreshPreviewButtons();
    console.error("Voice preview failed to play:", err);
  });
}
state.previewAudio.addEventListener("ended", () => {
  state.playingVoice = null;
  refreshPreviewButtons();
});

/* ---------------- Sidebar: cast / quality / expressiveness ---------------- */
function updateVoiceConsentVisibility() {
  el.voiceConsentRow.hidden = !anyCustomVoiceActive();
}

function fillEmptySlots() {
  if (!state.voices.length) return;
  for (let i = 0; i < state.numSpeakers; i += 1) {
    if (state.voiceSelections[i]) continue;
    const used = new Set(state.voiceSelections.filter(Boolean));
    const pick = state.voices.find((v) => !used.has(v.name)) || state.voices[i % state.voices.length];
    state.voiceSelections[i] = pick.name;
  }
}

function renderCast() {
  fillEmptySlots();
  el.speakerStepper.querySelectorAll("button").forEach((btn) => {
    btn.classList.toggle("active", Number(btn.dataset.count) === state.numSpeakers);
  });

  el.voiceRows.innerHTML = "";
  for (let i = 0; i < state.numSpeakers; i += 1) {
    const card = document.createElement("div");
    card.className = "slot-card";

    const dot = document.createElement("span");
    dot.className = "voice-dot";
    dot.style.background = slotColor(i);

    const info = document.createElement("div");
    info.className = "slot-info";
    const name = document.createElement("div");
    name.className = "slot-name";
    name.textContent = slotVoiceLabel(i);
    const meta = document.createElement("div");
    meta.className = "slot-meta";
    if (isCustomVoice(i)) {
      meta.textContent = state.customVoiceFiles[i] ? state.customVoiceFiles[i].name : "Upload a clip below";
    } else {
      const voice = voiceByName(state.voiceSelections[i]);
      meta.textContent = voice ? [GENDER_LABELS[voice.gender], ...(voice.tags || [])].join(" · ") : "";
    }
    info.append(name, meta);
    card.append(dot, info);

    if (!isCustomVoice(i) && state.voiceSelections[i]) {
      const playBtn = document.createElement("button");
      playBtn.type = "button";
      playBtn.className = "voice-play";
      playBtn.dataset.voice = state.voiceSelections[i];
      playBtn.textContent = "▶";
      playBtn.title = "Preview voice";
      playBtn.addEventListener("click", () => playVoicePreview(playBtn.dataset.voice));
      card.append(playBtn);
    }

    const changeBtn = document.createElement("button");
    changeBtn.type = "button";
    changeBtn.className = "slot-change";
    changeBtn.textContent = "Change";
    changeBtn.addEventListener("click", openLibrary);
    card.append(changeBtn);

    el.voiceRows.append(card);

    if (isCustomVoice(i)) {
      const uploadRow = document.createElement("div");
      uploadRow.className = "custom-voice-row";

      const fileLabel = document.createElement("label");
      fileLabel.className = "btn upload-mini-btn";
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
        renderCast();
      });
      fileLabel.append(fileInput);
      uploadRow.append(fileLabel);

      if (state.customVoiceFiles[i]) {
        const clearBtn = document.createElement("button");
        clearBtn.type = "button";
        clearBtn.className = "btn btn-icon-only";
        clearBtn.textContent = "✕";
        clearBtn.title = "Remove file";
        clearBtn.addEventListener("click", () => {
          state.customVoiceFiles[i] = null;
          renderCast();
        });
        uploadRow.append(clearBtn);
      }

      el.voiceRows.append(uploadRow);
    }
  }

  updateVoiceConsentVisibility();
  refreshPreviewButtons();
}

function renderQuality() {
  el.qualityPills.innerHTML = "";
  state.models.forEach((model) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "single-line";
    btn.title = model;
    const label = document.createElement("span");
    label.textContent = QUALITY_LABELS[model] || model;
    btn.append(label);
    btn.classList.toggle("active", model === state.model);
    btn.addEventListener("click", () => {
      state.model = model;
      renderQuality();
    });
    el.qualityPills.append(btn);
  });
}

el.speakerStepper.querySelectorAll("button").forEach((btn) => {
  btn.addEventListener("click", () => {
    state.numSpeakers = Number(btn.dataset.count);
    renderCast();
  });
});

function expressivenessWord(value) {
  if (value < 1.35) return "Calm";
  if (value < 1.7) return "Balanced";
  return "Dynamic";
}

function updateCfgLabel() {
  const value = Number(el.cfgScale.value);
  el.cfgScaleValue.textContent = `${value.toFixed(2)} · ${expressivenessWord(value)}`;
}
el.cfgScale.addEventListener("input", updateCfgLabel);
updateCfgLabel();

/* ---------------- Voice library ---------------- */
function libraryFilterOptions() {
  const tags = Array.from(new Set(state.voices.flatMap((v) => v.tags || [])));
  return [
    { key: "all", label: "All" },
    { key: "F", label: "Feminine" },
    { key: "M", label: "Masculine" },
    ...tags.map((t) => ({ key: t, label: t })),
  ];
}

function renderLibraryFilters() {
  el.libraryFilters.innerHTML = "";
  libraryFilterOptions().forEach((opt) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "filter-chip";
    chip.textContent = opt.label;
    chip.classList.toggle("active", state.libraryFilter === opt.key);
    chip.addEventListener("click", () => {
      state.libraryFilter = opt.key;
      renderLibraryFilters();
      renderLibraryGrid();
    });
    el.libraryFilters.append(chip);
  });
}

function makeSlotAssignRow(isAssigned, assignedColor, onAssign) {
  const row = document.createElement("div");
  row.className = "slot-assign";
  const label = document.createElement("span");
  label.className = "slot-assign-label";
  label.textContent = "Assign to";
  row.append(label);
  for (let i = 0; i < state.numSpeakers; i += 1) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = String(i + 1);
    if (isAssigned(i)) {
      btn.classList.add("assigned");
      btn.style.background = assignedColor(i);
    }
    btn.addEventListener("click", () => onAssign(i));
    row.append(btn);
  }
  return row;
}

function assignVoiceToSlot(i, value) {
  state.voiceSelections[i] = value;
  if (value !== CUSTOM_VOICE_VALUE) state.customVoiceFiles[i] = null;
  renderCast();
  renderTurns();
  renderLibraryGrid();
}

function renderLibraryGrid() {
  el.libraryGrid.innerHTML = "";
  const q = state.librarySearch.trim().toLowerCase();
  const filtered = state.voices.filter((v) => {
    const matchesSearch = !q || v.name.toLowerCase().includes(q);
    const matchesFilter =
      state.libraryFilter === "all" ||
      v.gender === state.libraryFilter ||
      (v.tags || []).includes(state.libraryFilter);
    return matchesSearch && matchesFilter;
  });

  filtered.forEach((voice) => {
    const card = document.createElement("div");
    card.className = "voice-card";

    const avatar = document.createElement("span");
    avatar.className = "voice-avatar";
    avatar.style.background = voice.color;
    avatar.textContent = voice.name[0];

    const body = document.createElement("div");
    body.className = "voice-card-body";

    const head = document.createElement("div");
    head.className = "voice-card-head";
    const name = document.createElement("span");
    name.className = "voice-card-name";
    name.textContent = voice.name;
    const preview = document.createElement("button");
    preview.type = "button";
    preview.className = "voice-preview-link";
    preview.dataset.voice = voice.name;
    preview.textContent = "Preview";
    preview.addEventListener("click", () => playVoicePreview(voice.name));
    head.append(name, preview);

    const meta = document.createElement("div");
    meta.className = "voice-card-meta";
    meta.textContent = [GENDER_LABELS[voice.gender], ...(voice.tags || [])].join(" · ");

    body.append(head, meta, makeSlotAssignRow(
      (i) => state.voiceSelections[i] === voice.name,
      () => voice.color,
      (i) => assignVoiceToSlot(i, voice.name),
    ));
    card.append(avatar, body);
    el.libraryGrid.append(card);
  });

  // Clone-a-voice card, always available
  const clone = document.createElement("div");
  clone.className = "voice-card clone-card";
  const cloneAvatar = document.createElement("span");
  cloneAvatar.className = "voice-avatar";
  cloneAvatar.textContent = "🎙️";
  const cloneBody = document.createElement("div");
  cloneBody.className = "voice-card-body";
  const cloneHead = document.createElement("div");
  cloneHead.className = "voice-card-head";
  const cloneName = document.createElement("span");
  cloneName.className = "voice-card-name";
  cloneName.textContent = "Clone a voice";
  cloneHead.append(cloneName);
  const cloneMeta = document.createElement("div");
  cloneMeta.className = "voice-card-meta";
  cloneMeta.textContent = "Upload a short clip of a voice you have rights to use";
  cloneBody.append(cloneHead, cloneMeta, makeSlotAssignRow(
    (i) => isCustomVoice(i),
    () => "#2a2016",
    (i) => assignVoiceToSlot(i, CUSTOM_VOICE_VALUE),
  ));
  clone.append(cloneAvatar, cloneBody);
  el.libraryGrid.append(clone);

  refreshPreviewButtons();
}

function renderLibrary() {
  el.librarySearch.value = state.librarySearch;
  renderLibraryFilters();
  renderLibraryGrid();
}

el.librarySearch.addEventListener("input", () => {
  state.librarySearch = el.librarySearch.value;
  renderLibraryGrid();
});

/* ---------------- Turn editor ---------------- */
function speakerChoiceLabel(i) {
  const sel = state.voiceSelections[i];
  if (sel === CUSTOM_VOICE_VALUE) return `Speaker ${i + 1} · Custom voice`;
  return sel ? `Speaker ${i + 1} · ${sel}` : `Speaker ${i + 1}`;
}

function renderTurns() {
  el.turnsList.innerHTML = "";
  if (!state.turns.length) {
    setComposerCollapsed(false);
    const empty = document.createElement("div");
    empty.className = "empty-transcript";
    empty.id = "emptyTurns";
    empty.innerHTML =
      '<div class="empty-title">No scene yet</div>' +
      "Type a scenario above and click <strong>Write with AI</strong>, pick an example, or start typing your own line below.";
    el.turnsList.append(empty);
    updateMeta();
    return;
  }

  state.turns.forEach((turn, idx) => {
    const spk = Math.min(4, Math.max(1, turn.speaker || 1));
    const card = document.createElement("div");
    card.className = "turn-card";
    card.style.borderLeftColor = slotColor(spk - 1);

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

/* Collapse the composer to a one-line strip once a script exists; the editor
   becomes the star. Clicking the strip (or emptying the script) expands it. */
function setComposerCollapsed(collapsed, summary) {
  el.composerBody.hidden = collapsed;
  el.composerCollapsedStrip.hidden = !collapsed;
  if (summary != null) el.collapsedSummary.textContent = summary;
}

el.composerCollapsedStrip.addEventListener("click", () => setComposerCollapsed(false));

function loadScriptResult(result, titleFallback, summary) {
  state.turns = result.turns;
  state.numSpeakers = result.num_speakers;
  const voices = (result.voices || []).slice(0, 4);
  while (voices.length < 4) voices.push(null);
  state.voiceSelections = voices;
  el.scriptTitle.textContent = result.title || titleFallback || "Untitled conversation";
  renderCast();
  renderTurns();
  setComposerCollapsed(true, summary || result.title || titleFallback || "");
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
    btn.addEventListener("click", () => loadScriptResult(example, example.title, `Example: ${example.title}`));
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
    loadScriptResult(payload, "Uploaded script", "Imported script");
    el.importDialog.close();
    el.pastedScript.value = "";
    el.scriptFileUpload.value = "";
  } catch (error) {
    alert(error.message);
  }
});

/* ---------------- AI script generation ---------------- */
const PROMPT_HINTS = [
  "Describe a scenario above first.",
  "Describe a new scenario above — or press Generate Audio to voice the script you already have.",
];

el.scriptPrompt.addEventListener("input", () => {
  if (PROMPT_HINTS.includes(el.scriptGenStatus.textContent)) el.scriptGenStatus.textContent = "";
});

el.generateScriptBtn.addEventListener("click", async () => {
  const prompt = el.scriptPrompt.value.trim();
  if (!prompt) {
    el.scriptPrompt.focus();
    el.scriptGenStatus.textContent = state.turns.length ? PROMPT_HINTS[1] : PROMPT_HINTS[0];
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
    loadScriptResult(payload, "Untitled conversation", `“${prompt}”`);
    el.scriptGenStatus.textContent = "";
  } catch (error) {
    el.scriptGenStatus.textContent = error.message;
  } finally {
    window.clearInterval(ticker);
    el.generateScriptBtn.disabled = false;
    el.generateScriptBtn.textContent = "Write with AI";
  }
});

/* ---------------- Player: waveform + synced transcript ---------------- */
async function decodeWavePeaks(url, blocks) {
  const response = await fetch(url);
  if (!response.ok) return null;
  const data = await response.arrayBuffer();
  const context = new AudioContext();
  try {
    const buffer = await context.decodeAudioData(data.slice(0));
    const samples = buffer.getChannelData(0);
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
    return peaks.map((p) => p / maxPeak);
  } finally {
    await context.close();
  }
}

function renderWave(progress) {
  drawWaveOn(el.resultWaveform, progress);
  if (el.playerStage.open) drawWaveOn(el.stageWaveform, progress);
}

function drawWaveOn(canvas, progress) {
  const cssWidth = canvas.clientWidth || 240;
  const cssHeight = canvas.clientHeight || 44;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(cssWidth * dpr)) {
    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);
  }
  const draw = canvas.getContext("2d");
  draw.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw.clearRect(0, 0, cssWidth, cssHeight);
  const peaks = state.wavePeaks || Array.from({ length: 48 }, () => 0.3);
  const blocks = peaks.length;
  const step = cssWidth / blocks;
  const barWidth = Math.max(2, step - 2);
  peaks.forEach((peak, i) => {
    const barHeight = Math.max(3, peak * cssHeight * 0.9);
    const x = i * step + 1;
    const played = (i + 0.5) / blocks <= progress;
    draw.fillStyle = played ? "#e2582a" : "#e4d8c2";
    draw.fillRect(x, (cssHeight - barHeight) / 2, barWidth, barHeight);
  });
}

function buildSyncedTranscript(snapshot) {
  // Approximate per-line timing: apportion total duration by word count.
  const words = snapshot.map((t) => t.text.split(/\s+/).filter(Boolean).length || 1);
  const total = words.reduce((a, b) => a + b, 0);
  let acc = 0;
  state.resultTurns = snapshot.map((t, i) => {
    const startRatio = acc / total;
    acc += words[i];
    const endRatio = acc / total;
    // Sub-split the turn into sentences for the stage caption, so a long
    // monologue advances line by line instead of dumping the whole turn.
    const parts = t.text.split(/(?<=[.!?…])\s+/).filter(Boolean);
    const partWords = parts.map((s) => s.split(/\s+/).filter(Boolean).length || 1);
    const partTotal = partWords.reduce((a, b) => a + b, 0);
    let partAcc = 0;
    const sentences = parts.map((text, j) => {
      const s = startRatio + (partAcc / partTotal) * (endRatio - startRatio);
      partAcc += partWords[j];
      const e = startRatio + (partAcc / partTotal) * (endRatio - startRatio);
      return { text, startRatio: s, endRatio: e };
    });
    return { ...t, startRatio, endRatio, sentences };
  });
  state.activeSyncIndex = -1;
  lastCaptionKey = "";

  el.syncedTranscript.innerHTML = "";
  el.stageTranscript.innerHTML = "";
  state.resultTurns.forEach((turn, i) => {
    state.resultTurns[i].rows = [el.syncedTranscript, el.stageTranscript].map((container) => {
      const row = document.createElement("div");
      row.className = "sync-line";
      const dot = document.createElement("span");
      dot.className = "sync-dot";
      dot.style.background = slotColor(turn.speaker - 1);
      const body = document.createElement("div");
      const label = document.createElement("div");
      label.className = "sync-label";
      label.textContent = `Speaker ${turn.speaker} · ${slotVoiceLabel(turn.speaker - 1)}`;
      const text = document.createElement("div");
      text.className = "sync-text";
      text.textContent = turn.text;
      body.append(label, text);
      row.append(dot, body);
      row.addEventListener("click", () => {
        const audio = el.resultAudio;
        if (Number.isFinite(audio.duration) && audio.duration > 0) {
          audio.currentTime = turn.startRatio * audio.duration;
          if (audio.paused) audio.play().catch(() => {});
        }
      });
      container.append(row);
      return row;
    });
  });
}

let lastCaptionKey = "";

function updateStageCaption(ratio = 0) {
  const turn = state.resultTurns[Math.max(0, state.activeSyncIndex)];
  if (!turn) {
    el.stageLine.textContent = "";
    el.stageSpeaker.textContent = "";
    return;
  }
  let sentence = turn.sentences[0];
  for (const s of turn.sentences) {
    if (ratio >= s.startRatio) sentence = s;
  }
  const key = `${state.activeSyncIndex}:${sentence ? sentence.text : ""}`;
  if (key === lastCaptionKey) return;
  lastCaptionKey = key;
  el.stageDot.style.background = slotColor(turn.speaker - 1);
  el.stageLine.textContent = sentence ? sentence.text : turn.text;
  el.stageSpeaker.textContent = `Speaker ${turn.speaker} · ${slotVoiceLabel(turn.speaker - 1)}`;
}

function updatePlaybackUI() {
  const audio = el.resultAudio;
  const duration = audio.duration;
  if (!Number.isFinite(duration) || duration <= 0) return;
  const ratio = audio.currentTime / duration;
  renderWave(ratio);
  const timeLabel = `${formatClock(audio.currentTime)} / ${formatClock(duration)}`;
  el.playerTime.textContent = timeLabel;
  el.stageTime.textContent = timeLabel;

  let active = -1;
  for (let i = 0; i < state.resultTurns.length; i += 1) {
    if (ratio >= state.resultTurns[i].startRatio) active = i;
  }
  if (active !== state.activeSyncIndex) {
    state.resultTurns.forEach((t, i) =>
      (t.rows || []).forEach((row) => row.classList.toggle("active", i === active)));
    state.activeSyncIndex = active;
    const rows = (state.resultTurns[active] && state.resultTurns[active].rows) || [];
    rows.forEach((row) => {
      if (row.offsetParent !== null) row.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
  }
  if (el.playerStage.open) updateStageCaption(ratio);
}

function togglePlayback() {
  const audio = el.resultAudio;
  if (!audio.src) return;
  if (audio.paused) audio.play().catch(() => {});
  else audio.pause();
}

function setPlayIcons(icon) {
  el.playBtn.textContent = icon;
  el.stagePlayBtn.textContent = icon;
}

el.playBtn.addEventListener("click", togglePlayback);
el.stagePlayBtn.addEventListener("click", togglePlayback);
el.resultAudio.addEventListener("play", () => setPlayIcons("❚❚"));
el.resultAudio.addEventListener("pause", () => setPlayIcons("►"));
el.resultAudio.addEventListener("ended", () => setPlayIcons("►"));
el.resultAudio.addEventListener("timeupdate", updatePlaybackUI);
el.resultAudio.addEventListener("loadedmetadata", updatePlaybackUI);

function seekFromClick(canvas, e) {
  const audio = el.resultAudio;
  if (!Number.isFinite(audio.duration) || audio.duration <= 0) return;
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  audio.currentTime = ratio * audio.duration;
  updatePlaybackUI();
}
el.resultWaveform.addEventListener("click", (e) => seekFromClick(el.resultWaveform, e));
el.stageWaveform.addEventListener("click", (e) => seekFromClick(el.stageWaveform, e));

/* Now Playing stage */
function openPlayerStage() {
  el.stageTitle.textContent = (state.resultTitle || "Untitled conversation").toUpperCase();
  const audio = el.resultAudio;
  updateStageCaption(audio.duration ? audio.currentTime / audio.duration : 0);
  el.playerStage.showModal();
  drawWaveOn(el.stageWaveform, el.resultAudio.duration ? el.resultAudio.currentTime / el.resultAudio.duration : 0);
}

el.openPlayerBtn.addEventListener("click", openPlayerStage);
el.stageBackBtn.addEventListener("click", () => el.playerStage.close());
el.playerStage.addEventListener("click", (e) => { if (e.target === el.playerStage) el.playerStage.close(); });

el.stageScriptToggle.addEventListener("click", () => {
  el.stageTranscript.hidden = !el.stageTranscript.hidden;
  el.stageScriptToggle.textContent = el.stageTranscript.hidden ? "View full script" : "Hide script";
});

document.addEventListener("keydown", (e) => {
  if (!el.playerStage.open || e.code !== "Space") return;
  if (/^(input|textarea|select)$/i.test(e.target.tagName)) return;
  e.preventDefault();
  togglePlayback();
});

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

  const turnsSnapshot = state.turns
    .filter((t) => (t.text || "").trim())
    .map((t) => ({ speaker: Math.min(4, Math.max(1, t.speaker || 1)), text: t.text.trim() }));

  el.generateBtn.disabled = true;
  el.generateBtn.textContent = "Generating...";
  el.resultBlock.classList.remove("visible");
  el.resultAudio.pause();
  el.dockEmpty.hidden = true;
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
    model: state.model,
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
          el.stageDownloadBtn.href = url;
          el.generationTime.textContent = formatDuration((performance.now() - started) / 1000);
          el.audioDuration.textContent = formatDuration(evt.audio_duration);
          el.resultModel.textContent = state.model;
          el.playerTime.textContent = `0:00 / ${formatClock(evt.audio_duration)}`;
          el.stageTime.textContent = `0:00 / ${formatClock(evt.audio_duration)}`;
          setPlayIcons("►");
          state.resultTitle = el.scriptTitle.textContent;
          buildSyncedTranscript(turnsSnapshot);
          el.dockEmpty.hidden = true;
          el.resultBlock.classList.add("visible");
          state.wavePeaks = await decodeWavePeaks(url, 48);
          renderWave(0);
          openPlayerStage();
        }
      }
    }
  } catch (error) {
    setStatus("error", error.message);
    el.dockEmpty.hidden = false;
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
  state.model = models[0] || null;
  state.voices = voices;
  state.examples = examples;
  state.voiceSelections = voices.slice(0, 4).map((v) => v.name);
  while (state.voiceSelections.length < 4) state.voiceSelections.push(null);

  el.durationSelect.innerHTML = durationOptions
    .map((m) => `<option value="${m}">${m >= 60 ? "1 hr" : `${m} min`}</option>`)
    .join("");
  const defaultDuration = durationOptions.includes(2) ? 2 : durationOptions[0];
  el.durationSelect.value = String(defaultDuration);

  renderCast();
  renderQuality();
  renderTurns();
  renderExamplePills();
  updateStatus();
  window.setInterval(updateStatus, 8000);
}

init().catch((error) => {
  el.scriptGenStatus.textContent = `Failed to load app data: ${error.message}`;
});
