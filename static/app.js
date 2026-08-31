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
  connecting: ["Submitted", "GPU resources warming up — a cold start can take up to a minute."],
  queued: ["Queued", "GPU resources warming up — the worker is spinning up."],
  loading_model: ["Loading model", "Streaming VibeVoice weights to the GPU."],
  loading_voices: ["Loading voices", null],
  preparing_inputs: ["Preparing", "Formatting the conversation for the model."],
  generating_audio: ["Generating", "Synthesizing speech — this is the longest step."],
  processing_audio: ["Finalizing", "Converting tensors into a playable waveform."],
  complete: ["Complete", "Press play, or download the WAV."],
  error: ["Error", "Check the log for details."],
  cancelled: ["Stopped", "Generation cancelled. The next run may need a cold start."],
};

const QUALITY_LABELS = { "VibeVoice-1.5B": "Fast", "VibeVoice-7B": "Best" };
const GENDER_LABELS = { F: "Feminine", M: "Masculine" };
const SPEAKER_FALLBACK_COLORS = ["#e2582a", "#2f6f63", "#cc8a2e", "#7b4b94"];
const CUSTOM_PREFIX = "custom:";
const CUSTOM_COLORS = ["#8a5a44", "#4a6b8a", "#7b5e7d", "#5e7b60"];
const MAX_CUSTOM_VOICES = 4;
const MAX_CUSTOM_AUDIO_BYTES = 15 * 1024 * 1024;

const state = {
  turns: [],
  numSpeakers: 2,
  voices: [],
  voiceSelections: [null, null, null, null],
  customVoices: [],  // saved clones: {id, name, duration, blob, url, createdAt}
  models: [],
  model: null,
  examples: [],
  parodyLines: [],
  parodyIndex: 0,
  previewAudio: new Audio(),
  playingVoice: null,
  librarySearch: "",
  libraryFilter: "all",
  libraryTargetSlot: null,  // non-null: library picks a voice for this one slot
  cloneTargetSlots: [],     // speaker slots the saved clone gets assigned to (0..4 of them)
  cloneBlob: null,
  cloneName: "",
  cloneDuration: 0,
  cloneReplaceId: null,     // when the voice library is full: which saved voice to override
  cloneRawBuffer: null,     // last capture's undecoded bytes, so Optimize can be re-applied
  mediaRecorder: null,
  resultTurns: [],       // snapshot of turns for the synced transcript
  resultTitle: "",
  wavePeaks: null,
  takeDuration: 0,
  dockWS: null,          // wavesurfer instances: dock mini player / stage
  stageWS: null,
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
  "statusCard", "statusTitle", "statusDesc", "stopGenBtn",
  "progressTrack", "progressFill", "progressMeta",
  "stageGenPane", "stagePlayPane", "genStageTitle", "genStagePct", "genStageTrack",
  "genStageFill", "genStageMeta", "genStageDesc", "genStageLog", "genStageStopBtn",
  "genPreviewRow", "genPreviewLabel", "genPreviewMute",
  "dockEmpty", "resultBlock", "resultWaveform", "resultAudio",
  "playBtn", "playerTime", "syncedTranscript", "openPlayerBtn",
  "composerCollapsedStrip", "collapsedSummary", "composerBody",
  "playerStage", "stageTitle", "stagePlayBtn", "stageWaveform", "stageTime",
  "stageDot", "stageLine", "stageSpeaker", "stageCloseBtn", "stageDownloadBtn", "stageOrbs",
  "stageScriptToggle", "stageTranscript", "soundSeg", "soundHint", "polishStatus",
  "generationTime", "audioDuration", "resultModel", "downloadBtn",
  "realtimeRow", "realtimeFactor", "warmupRow", "warmupTime",
  "downloadMp3Btn", "stageDownloadMp3Btn",
  "logToggleBtn", "logBox",
  "voiceLibraryDialog", "closeLibraryBtn", "librarySearch", "libraryFilters", "libraryGrid", "libraryTitle",
  "cloneVoiceBtn", "cloneDialog", "closeCloneBtn", "recordBtn", "cloneFileInput", "recordTimer",
  "clonePreview", "cloneAudio", "cloneMeta", "cloneSlotRow", "cloneConsentCheckbox",
  "cloneNameInput", "cloneReplaceRow", "cloneReplacePills", "optimizeCheckbox",
  "readScriptToggle", "readScriptCard", "readScriptText", "readScriptMeta", "readScriptShuffle",
  "cancelCloneBtn", "useCloneBtn",
  "importDialog", "pastedScript", "scriptFileUpload", "scriptFileName", "cancelImportBtn", "loadScriptBtn",
].forEach((id) => { el[id] = document.getElementById(id); });

function autoGrow(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${textarea.scrollHeight}px`;
}

function voiceByName(name) {
  return state.voices.find((v) => v.name === name) || null;
}

function isCustomVoice(i) {
  return (state.voiceSelections[i] || "").startsWith(CUSTOM_PREFIX);
}

function customVoiceById(id) {
  return state.customVoices.find((v) => v.id === id) || null;
}

function customVoiceForSlot(i) {
  return isCustomVoice(i) ? customVoiceById(state.voiceSelections[i].slice(CUSTOM_PREFIX.length)) : null;
}

function customColor(voice) {
  const idx = state.customVoices.indexOf(voice);
  return CUSTOM_COLORS[Math.max(0, idx) % CUSTOM_COLORS.length];
}

function anyCustomVoiceActive() {
  return Array.from({ length: state.numSpeakers }, (_, i) => i).some(isCustomVoice);
}

function slotColor(i) {
  if (isCustomVoice(i)) {
    const voice = customVoiceForSlot(i);
    return voice ? customColor(voice) : SPEAKER_FALLBACK_COLORS[i];
  }
  const voice = voiceByName(state.voiceSelections[i]);
  return voice ? voice.color : SPEAKER_FALLBACK_COLORS[i];
}

function slotVoiceLabel(i) {
  if (isCustomVoice(i)) {
    const voice = customVoiceForSlot(i);
    return voice ? voice.name : "Custom voice";
  }
  return state.voiceSelections[i] || `Voice ${i + 1}`;
}

/* ---- Saved clone persistence (IndexedDB: survives reloads, stays on this device) ---- */
function idbOpen() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open("chorus-voices", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("voices", { keyPath: "id" });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbRequest(mode, fn) {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const req = fn(db.transaction("voices", mode).objectStore("voices"));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function loadCustomVoices() {
  try {
    const rows = await idbRequest("readonly", (s) => s.getAll());
    rows.sort((a, b) => a.createdAt - b.createdAt);
    state.customVoices = rows.map((r) => ({ ...r, url: URL.createObjectURL(r.blob) }));
  } catch {
    state.customVoices = [];  // private mode etc. — clones work, just don't survive reload
  }
}

async function persistCustomVoice(v) {
  try {
    await idbRequest("readwrite", (s) =>
      s.put({ id: v.id, name: v.name, duration: v.duration, createdAt: v.createdAt, blob: v.blob }));
  } catch { /* in-memory only */ }
}

async function removeCustomVoiceRecord(id) {
  try { await idbRequest("readwrite", (s) => s.delete(id)); } catch { /* in-memory only */ }
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
el.scriptFileUpload.addEventListener("change", () => {
  el.scriptFileName.textContent = el.scriptFileUpload.files[0]
    ? el.scriptFileUpload.files[0].name
    : "No file chosen";
});
el.cancelImportBtn.addEventListener("click", () => el.importDialog.close());
el.importDialog.addEventListener("click", (e) => { if (e.target === el.importDialog) el.importDialog.close(); });

function openLibrary(targetSlot = null) {
  state.libraryTargetSlot = targetSlot;
  el.libraryTitle.textContent =
    targetSlot === null ? "Choose your voices" : `Choose a voice for Speaker ${targetSlot + 1}`;
  renderLibrary();
  el.voiceLibraryDialog.showModal();
}
el.browseVoicesBtn.addEventListener("click", () => openLibrary(null));
el.closeLibraryBtn.addEventListener("click", () => el.voiceLibraryDialog.close());
// The dialog's close event fires for the X, backdrop clicks, Esc, AND the
// programmatic closes after choosing a voice — one hook silences them all.
el.voiceLibraryDialog.addEventListener("close", stopVoicePreview);
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

function previewSrcFor(key) {
  if (key.startsWith(CUSTOM_PREFIX)) {
    const voice = customVoiceById(key.slice(CUSTOM_PREFIX.length));
    return voice ? voice.url : null;
  }
  const voice = voiceByName(key);
  return voice ? voice.preview_url : null;
}

function playVoicePreview(key) {
  if (state.playingVoice === key) {
    state.previewAudio.pause();
    state.playingVoice = null;
    refreshPreviewButtons();
    return;
  }
  const src = previewSrcFor(key);
  if (!src) return;
  state.previewAudio.src = src;
  state.playingVoice = key;
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

function stopVoicePreview() {
  if (!state.playingVoice) return;
  state.previewAudio.pause();
  state.playingVoice = null;
  refreshPreviewButtons();
}

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
      const voice = customVoiceForSlot(i);
      meta.textContent = voice ? `Cloned voice · ${Math.round(voice.duration)}s` : "Clip missing — re-record";
    } else {
      const voice = voiceByName(state.voiceSelections[i]);
      meta.textContent = voice ? [GENDER_LABELS[voice.gender], ...(voice.tags || [])].join(" · ") : "";
    }
    info.append(name, meta);
    card.append(dot, info);

    if (state.voiceSelections[i] && previewSrcFor(state.voiceSelections[i])) {
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
    changeBtn.addEventListener("click", () => openLibrary(i));
    card.append(changeBtn);

    el.voiceRows.append(card);

    if (isCustomVoice(i) && !customVoiceForSlot(i)) {
      const uploadRow = document.createElement("div");
      uploadRow.className = "custom-voice-row";
      const fixBtn = document.createElement("button");
      fixBtn.type = "button";
      fixBtn.className = "btn upload-mini-btn";
      fixBtn.textContent = "Record or upload a clip…";
      fixBtn.addEventListener("click", () => openCloneDialog(i));
      uploadRow.append(fixBtn);
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
  renderCast();
  renderTurns();
  renderLibraryGrid();
}

function makeUseVoiceButton(slot, key) {
  const useBtn = document.createElement("button");
  useBtn.type = "button";
  useBtn.className = "btn btn-accent use-voice-btn";
  const current = state.voiceSelections[slot] === key;
  useBtn.textContent = current ? "Current voice" : "Use voice";
  useBtn.disabled = current;
  useBtn.addEventListener("click", () => {
    assignVoiceToSlot(slot, key);
    el.voiceLibraryDialog.close();
  });
  return useBtn;
}

function deleteCustomVoice(voice) {
  state.customVoices = state.customVoices.filter((v) => v !== voice);
  URL.revokeObjectURL(voice.url);
  removeCustomVoiceRecord(voice.id);
  const key = CUSTOM_PREFIX + voice.id;
  state.voiceSelections = state.voiceSelections.map((sel) => (sel === key ? null : sel));
  renderCast();
  renderTurns();
  renderLibraryGrid();
}

function buildCustomVoiceCard(voice) {
  const key = CUSTOM_PREFIX + voice.id;
  const card = document.createElement("div");
  card.className = "voice-card";

  const avatar = document.createElement("span");
  avatar.className = "voice-avatar";
  avatar.style.background = customColor(voice);
  avatar.textContent = (voice.name || "?")[0].toUpperCase();

  const body = document.createElement("div");
  body.className = "voice-card-body";

  const head = document.createElement("div");
  head.className = "voice-card-head";
  const name = document.createElement("span");
  name.className = "voice-card-name";
  name.textContent = voice.name;

  const actions = document.createElement("span");
  actions.className = "voice-card-actions";
  const preview = document.createElement("button");
  preview.type = "button";
  preview.className = "voice-preview-link";
  preview.dataset.voice = key;
  preview.textContent = "Preview";
  preview.addEventListener("click", () => playVoicePreview(key));

  const renameBtn = document.createElement("button");
  renameBtn.type = "button";
  renameBtn.className = "voice-mini-action";
  renameBtn.textContent = "✎";
  renameBtn.title = "Rename";
  renameBtn.addEventListener("click", () => {
    const input = document.createElement("input");
    input.className = "voice-rename-input";
    input.value = voice.name;
    input.maxLength = 40;
    name.replaceWith(input);
    input.focus();
    input.select();
    const commit = () => {
      voice.name = input.value.trim() || voice.name;
      persistCustomVoice(voice);
      renderCast();
      renderTurns();
      renderLibraryGrid();
    };
    input.addEventListener("blur", commit);
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") input.blur(); });
  });

  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.className = "voice-mini-action";
  deleteBtn.textContent = "✕";
  deleteBtn.title = "Delete this voice";
  deleteBtn.addEventListener("click", () => deleteCustomVoice(voice));

  actions.append(preview, renameBtn, deleteBtn);
  head.append(name, actions);

  const meta = document.createElement("div");
  meta.className = "voice-card-meta";
  meta.textContent = `Cloned voice · ${Math.round(voice.duration)}s`;

  const slot = state.libraryTargetSlot;
  if (slot === null) {
    body.append(head, meta, makeSlotAssignRow(
      (i) => state.voiceSelections[i] === key,
      () => customColor(voice),
      (i) => assignVoiceToSlot(i, key),
    ));
  } else {
    body.append(head, meta, makeUseVoiceButton(slot, key));
  }
  card.append(avatar, body);
  return card;
}

function librarySectionLabel(text) {
  const label = document.createElement("div");
  label.className = "library-section-label";
  label.textContent = text;
  return label;
}

function renderLibraryGrid() {
  el.libraryGrid.innerHTML = "";
  const q = state.librarySearch.trim().toLowerCase();

  const myVoices = state.customVoices.filter((v) => !q || v.name.toLowerCase().includes(q));
  if (myVoices.length && state.libraryFilter === "all") {
    el.libraryGrid.append(librarySectionLabel("My voices"));
    myVoices.forEach((v) => el.libraryGrid.append(buildCustomVoiceCard(v)));
    el.libraryGrid.append(librarySectionLabel("Presets"));
  }

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

    const slot = state.libraryTargetSlot;
    if (slot === null) {
      body.append(head, meta, makeSlotAssignRow(
        (i) => state.voiceSelections[i] === voice.name,
        () => voice.color,
        (i) => assignVoiceToSlot(i, voice.name),
      ));
    } else {
      // Slot-first mode: one click puts this voice in the target slot.
      body.append(head, meta, makeUseVoiceButton(slot, voice.name));
    }
    card.append(avatar, body);
    el.libraryGrid.append(card);
  });

  // Clone-a-voice card routes to the dedicated clone dialog
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
  cloneMeta.textContent = "Record or upload 15–30s of a voice you have rights to use";
  const cloneBtn = document.createElement("button");
  cloneBtn.type = "button";
  cloneBtn.className = "btn btn-accent use-voice-btn";
  cloneBtn.textContent = "Record or upload";
  cloneBtn.addEventListener("click", () => {
    const target = state.libraryTargetSlot;
    el.voiceLibraryDialog.close();
    openCloneDialog(target);
  });
  cloneBody.append(cloneHead, cloneMeta, cloneBtn);
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

/* ---------------- Clone-a-voice dialog ---------------- */
// Read-along passages tuned for cloning: natural first-person voice, a
// question and an exclamation in each (pitch range), varied sounds, and
// ~25-30s when read at a comfortable pace.
const READ_SCRIPTS = [
  "Okay, so here's the thing about mornings: I always swear I'll get up early, and somehow the snooze button wins every single time. Last Tuesday I actually did it — coffee, a quick walk, the whole routine — and honestly? Best day I'd had in months. The air was cool, the streets were quiet, and for once nobody needed a single thing from me. I watched the bakery on the corner pull its first trays out of the oven, and the smell alone was worth the alarm. So naturally, I told everyone I was a morning person now. That lasted exactly four days! But I keep coming back to it, because those quiet hours feel like borrowed time. Maybe tomorrow I'll try it again — no promises, though.",
  "When I was about nine, my grandfather taught me to fish off the old wooden dock behind his house. He'd say, \"Patience isn't waiting — it's what you do while you wait.\" I had no idea what that meant back then. I just liked the sandwiches, and the way the water slapped against the posts. We'd sit for hours, mostly in silence, watching dragonflies stitch back and forth across the surface. Did we ever catch much? Almost never! One summer the biggest thing we pulled up was somebody's old boot, and he laughed so hard he nearly fell in. Now, every time I'm stuck in line or watching the kettle boil, I hear his voice again, and I catch myself smiling without meaning to. Funny how the smallest afternoons turn out to be the ones you keep.",
  "You want to know the best meal I've ever had? A tiny noodle shop, eleven o'clock at night, rain hammering against the windows. Six seats, no menu, and a cook who never said a word. He just looked at you, nodded once, and started cooking. That broth changed my life! Rich and smoky and somehow gentle at the same time, with noodles he'd pulled by hand maybe ninety seconds earlier. I sat there dripping wet and completely happy, and I ordered a second bowl before I'd finished the first. I've chased that flavor everywhere since — big cities, little towns, my own kitchen at two in the morning — and nothing has ever come close. Some things only taste right once. The trick is knowing it while it's happening, and that night, I did.",
  "There's a particular hour just before sunset when everything slows down. The light turns gold, shadows stretch long across the yard, and even the birds seem to lower their voices. I like to sit outside then, with a cup of tea going cold beside me, and let my thoughts wander wherever they want. Sometimes they drift to old friends, or to trips I still mean to take; sometimes they don't go anywhere at all, and that's fine too. The neighbor's dog usually wanders over, flops down on the warm stones, and sighs like he's had the longest day of anyone. Honestly, he might be right! Then the streetlights blink on, the cool air moves in, and the moment is over. It never lasts long. Isn't that exactly why it matters?",
  "Here's my confession: I talk to my plants. Not just a quick hello, either — full conversations. The fern gets encouragement, the cactus gets tough love, and the orchid? The orchid gets bribed. I promise it better light, less draft, a bigger pot in the spring — whatever it takes, because it blooms exactly when it feels like it and not a moment sooner. My sister thinks I've completely lost the plot. She stood in my kitchen last month, listening to me thank the basil for pulling through a rough week, and just slowly shook her head. But does any of it work? Who knows! All I can tell you is that every single one of them is still alive, which is more than I can say for every plant I owned before, so I'm not changing a thing.",
];

const CLONE_MIN_SECONDS = 5;        // hard floor
const CLONE_GOOD_SECONDS = 10;      // below this: warn, above: good to go
const CLONE_MAX_RECORD_SECONDS = 60;

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i += 1) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); writeStr(8, "WAVE");
  writeStr(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  writeStr(36, "data"); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function trimSilence(samples, rate) {
  const threshold = 0.012;
  const pad = Math.round(rate * 0.15);
  let start = 0;
  let end = samples.length - 1;
  while (start < samples.length && Math.abs(samples[start]) < threshold) start += 1;
  while (end > start && Math.abs(samples[end]) < threshold) end -= 1;
  if (end - start < rate) return samples;  // mostly silence — leave it alone
  return samples.slice(Math.max(0, start - pad), Math.min(samples.length, end + pad));
}

function normalizePeak(samples, target = 0.85) {
  let peak = 0;
  for (let i = 0; i < samples.length; i += 1) peak = Math.max(peak, Math.abs(samples[i]));
  if (peak < 0.001 || peak >= target) return samples;
  const gain = target / peak;
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) out[i] = samples[i] * gain;
  return out;
}

// Decode any browser-supported audio and re-render as 24kHz mono WAV, so the
// backend always receives a format it can read. With optimize: a high-pass at
// 80Hz kills rumble and handling noise, then silence is trimmed and the peak
// normalized so quiet laptop-mic takes arrive at a healthy level.
async function toMonoWav(arrayBuffer, optimize) {
  const probe = new AudioContext();
  let decoded;
  try {
    decoded = await probe.decodeAudioData(arrayBuffer.slice(0));
  } finally {
    await probe.close();
  }
  const rate = 24000;
  const frames = Math.ceil(decoded.duration * rate);
  const offline = new OfflineAudioContext(1, frames, rate);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  if (optimize) {
    const highpass = offline.createBiquadFilter();
    highpass.type = "highpass";
    highpass.frequency.value = 80;
    source.connect(highpass);
    highpass.connect(offline.destination);
  } else {
    source.connect(offline.destination);
  }
  source.start();
  const rendered = await offline.startRendering();
  let samples = rendered.getChannelData(0);
  if (optimize) {
    samples = normalizePeak(trimSilence(samples, rate));
  }
  return { blob: encodeWav(samples, rate), duration: samples.length / rate };
}

function setCloneClip(blob, name, duration) {
  state.cloneBlob = blob;
  state.cloneName = name;
  state.cloneDuration = duration;
  el.cloneAudio.src = URL.createObjectURL(blob);
  el.clonePreview.hidden = false;
  const secs = Math.round(duration);
  let note = `${name} · ${secs}s`;
  if (duration < CLONE_MIN_SECONDS) note += " — too short; record at least 5 seconds.";
  else if (duration < CLONE_GOOD_SECONDS) note += " — usable, but 15–30s clones much better.";
  else note += " — looks good.";
  el.cloneMeta.textContent = note;
  updateCloneConfirm();
}

function cloneAtCapacity() {
  return state.customVoices.length >= MAX_CUSTOM_VOICES;
}

function updateCloneSaveLabel() {
  const slots = state.cloneTargetSlots;
  if (!slots.length) {
    el.useCloneBtn.textContent = "Save voice";
  } else if (slots.length === 1) {
    el.useCloneBtn.textContent = `Save & use as Speaker ${slots[0] + 1}`;
  } else {
    const nums = [...slots].sort().map((i) => i + 1);
    el.useCloneBtn.textContent =
      `Save & use as Speakers ${nums.slice(0, -1).join(", ")} & ${nums[nums.length - 1]}`;
  }
}

function renderCloneSlots() {
  el.cloneSlotRow.innerHTML = "";
  for (let i = 0; i < state.numSpeakers; i += 1) {
    const pill = document.createElement("button");
    pill.type = "button";
    pill.className = "clone-slot-pill";
    pill.textContent = `Speaker ${i + 1} · ${slotVoiceLabel(i)}`;
    pill.classList.toggle("active", state.cloneTargetSlots.includes(i));
    pill.addEventListener("click", () => {
      state.cloneTargetSlots = state.cloneTargetSlots.includes(i)
        ? state.cloneTargetSlots.filter((s) => s !== i)
        : [...state.cloneTargetSlots, i];
      renderCloneSlots();
      updateCloneSaveLabel();
    });
    el.cloneSlotRow.append(pill);
  }
}

function updateCloneConfirm() {
  el.useCloneBtn.disabled = !(
    state.cloneBlob &&
    state.cloneDuration >= CLONE_MIN_SECONDS &&
    el.cloneConsentCheckbox.checked &&
    (!cloneAtCapacity() || state.cloneReplaceId)
  );
}

function renderCloneReplacePills() {
  el.cloneReplaceRow.hidden = !cloneAtCapacity();
  el.cloneReplacePills.innerHTML = "";
  if (!cloneAtCapacity()) return;
  state.customVoices.forEach((v) => {
    const pill = document.createElement("button");
    pill.type = "button";
    pill.className = "clone-slot-pill";
    pill.textContent = v.name;
    pill.classList.toggle("active", v.id === state.cloneReplaceId);
    pill.addEventListener("click", () => {
      state.cloneReplaceId = v.id;
      renderCloneReplacePills();
      updateCloneConfirm();
    });
    el.cloneReplacePills.append(pill);
  });
}

function stopRecording() {
  if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") state.mediaRecorder.stop();
}

function resetRecordButton() {
  clearInterval(recordTicker);
  el.recordBtn.textContent = "● Record";
  el.recordBtn.classList.remove("recording");
  el.recordTimer.classList.remove("rec-live");
  el.recordTimer.hidden = true;
}

let recordTicker = null;

async function startRecording() {
  let stream;
  const optimize = el.optimizeCheckbox.checked;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { noiseSuppression: optimize, echoCancellation: optimize, autoGainControl: optimize },
    });
  } catch {
    el.cloneMeta.textContent = "Microphone access was denied — allow it in your browser, or upload a file instead.";
    el.clonePreview.hidden = false;
    return;
  }
  const chunks = [];
  const recorder = new MediaRecorder(stream);
  state.mediaRecorder = recorder;
  recorder.addEventListener("dataavailable", (e) => { if (e.data.size) chunks.push(e.data); });
  recorder.addEventListener("stop", async () => {
    stream.getTracks().forEach((t) => t.stop());
    resetRecordButton();
    try {
      const raw = await new Blob(chunks).arrayBuffer();
      state.cloneRawBuffer = raw;
      const { blob, duration } = await toMonoWav(raw, el.optimizeCheckbox.checked);
      setCloneClip(blob, "Recorded clip", duration);
    } catch {
      el.cloneMeta.textContent = "Could not process the recording — try again or upload a file.";
      el.clonePreview.hidden = false;
    }
  });
  recorder.start();
  const startedAt = Date.now();
  el.recordBtn.textContent = "■ Stop";
  el.recordBtn.classList.add("recording");
  el.recordTimer.hidden = false;
  el.recordTimer.classList.add("rec-live");
  el.recordTimer.textContent = "0:00";
  clearInterval(recordTicker);
  recordTicker = setInterval(() => {
    const elapsed = (Date.now() - startedAt) / 1000;
    el.recordTimer.textContent = formatClock(elapsed);
    if (elapsed >= CLONE_MAX_RECORD_SECONDS) stopRecording();
  }, 250);
}

el.recordBtn.addEventListener("click", () => {
  if (state.mediaRecorder && state.mediaRecorder.state === "recording") stopRecording();
  else startRecording();
});

el.cloneFileInput.addEventListener("change", async () => {
  const file = el.cloneFileInput.files[0];
  el.cloneFileInput.value = "";
  if (!file) return;
  if (file.size > MAX_CUSTOM_AUDIO_BYTES) {
    alert(`That file is too large (max ${MAX_CUSTOM_AUDIO_BYTES / (1024 * 1024)} MB).`);
    return;
  }
  try {
    const raw = await file.arrayBuffer();
    const { blob, duration } = await toMonoWav(raw, el.optimizeCheckbox.checked);
    state.cloneRawBuffer = raw;
    setCloneClip(blob, file.name, duration);
  } catch {
    // Undecodable in this browser — pass the raw file through; the backend may still read it.
    state.cloneRawBuffer = null;
    state.cloneBlob = file;
    state.cloneName = file.name;
    state.cloneDuration = CLONE_GOOD_SECONDS;
    el.cloneAudio.removeAttribute("src");
    el.clonePreview.hidden = false;
    el.cloneMeta.textContent = `${file.name} — couldn't preview this format; it will be sent as-is.`;
    updateCloneConfirm();
  }
});

el.cloneConsentCheckbox.addEventListener("change", updateCloneConfirm);

/* Read-along script: gives recorders something natural to say */
let readScriptIndex = 0;

function showReadScript() {
  const text = READ_SCRIPTS[readScriptIndex % READ_SCRIPTS.length];
  el.readScriptText.textContent = text;
  const words = text.split(/\s+/).length;
  el.readScriptMeta.textContent =
    `~${Math.round((words / 150) * 60)}s at a relaxed pace · read it like you'd say it, not like an announcement`;
}

el.readScriptToggle.addEventListener("click", () => {
  el.readScriptCard.hidden = !el.readScriptCard.hidden;
  el.readScriptToggle.textContent = el.readScriptCard.hidden
    ? "Not sure what to say? Show a script to read →"
    : "Hide the script";
  if (!el.readScriptCard.hidden) showReadScript();
});

el.readScriptShuffle.addEventListener("click", () => {
  readScriptIndex += 1;
  showReadScript();
});

// Re-apply (or undo) optimization on the captured clip when the box is toggled.
el.optimizeCheckbox.addEventListener("change", async () => {
  if (!state.cloneRawBuffer) return;
  try {
    const { blob, duration } = await toMonoWav(state.cloneRawBuffer, el.optimizeCheckbox.checked);
    setCloneClip(blob, state.cloneName, duration);
  } catch { /* keep the current clip */ }
});

function openCloneDialog(targetSlot = null) {
  // Default landing spot: the slot this dialog was opened for, else Speaker 1
  // so a fresh clone always has a visible home. Deselect all to library-only.
  state.cloneTargetSlots = [targetSlot === null ? 0 : Math.min(targetSlot, state.numSpeakers - 1)];
  state.cloneBlob = null;
  state.cloneName = "";
  state.cloneDuration = 0;
  state.cloneReplaceId = null;
  state.cloneRawBuffer = null;
  el.clonePreview.hidden = true;
  el.cloneAudio.removeAttribute("src");
  el.cloneMeta.textContent = "";
  el.cloneNameInput.value = "";
  el.cloneConsentCheckbox.checked = el.voiceConsentCheckbox.checked;
  renderCloneSlots();
  updateCloneSaveLabel();
  renderCloneReplacePills();
  updateCloneConfirm();
  el.cloneDialog.showModal();
}

function closeCloneDialog() {
  stopRecording();
  resetRecordButton();
  el.cloneDialog.close();
}

el.cloneVoiceBtn.addEventListener("click", () => openCloneDialog(null));
el.closeCloneBtn.addEventListener("click", closeCloneDialog);
el.cancelCloneBtn.addEventListener("click", closeCloneDialog);
el.cloneDialog.addEventListener("click", (e) => { if (e.target === el.cloneDialog) closeCloneDialog(); });

el.useCloneBtn.addEventListener("click", () => {
  const typed = el.cloneNameInput.value.trim();
  const fromFile = state.cloneName && state.cloneName !== "Recorded clip"
    ? state.cloneName.replace(/\.[a-z0-9]+$/i, "")
    : "";
  const name = typed || fromFile || `My voice ${state.customVoices.length + 1}`;

  const existing = state.cloneReplaceId ? customVoiceById(state.cloneReplaceId) : null;
  let voice;
  if (existing) {
    URL.revokeObjectURL(existing.url);
    Object.assign(existing, {
      name, duration: state.cloneDuration, blob: state.cloneBlob,
      url: URL.createObjectURL(state.cloneBlob),
    });
    voice = existing;
  } else {
    voice = {
      id: crypto.randomUUID(), name,
      duration: state.cloneDuration, blob: state.cloneBlob,
      url: URL.createObjectURL(state.cloneBlob), createdAt: Date.now(),
    };
    state.customVoices.push(voice);
  }
  persistCustomVoice(voice);
  state.cloneTargetSlots.forEach((i) => {
    state.voiceSelections[i] = CUSTOM_PREFIX + voice.id;
  });
  el.voiceConsentCheckbox.checked = el.cloneConsentCheckbox.checked;
  closeCloneDialog();
  renderCast();
  renderTurns();
});

/* ---------------- Turn editor ---------------- */
function speakerChoiceLabel(i) {
  const sel = state.voiceSelections[i];
  if (!sel) return `Speaker ${i + 1}`;
  return `Speaker ${i + 1} · ${slotVoiceLabel(i)}`;
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
      "Describe a scenario above and click <strong>Write with AI</strong>, import a script, or start typing your own line below.";
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
  const suggested = (result.voices || []).slice(0, 4);
  while (suggested.length < 4) suggested.push(null);
  // Keep the user's cloned-voice assignments; only refresh preset suggestions.
  state.voiceSelections = state.voiceSelections.map((current, i) =>
    (current || "").startsWith(CUSTOM_PREFIX) ? current : suggested[i]);
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
  el.examplePills.hidden = !state.examples.length;  // no empty gap when unset
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
    el.scriptFileName.textContent = "No file chosen";
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

/* ---------------- Player: waveform (wavesurfer.js) + synced transcript ----------------
   Peaks are computed with a streaming RMS scan over the WAV's raw PCM: a
   2-hour take (~430MB) never has to pass through the browser's audio decoder,
   and RMS (unlike per-bucket max) keeps its shape at any length — peak
   bucketing over multi-minute windows saturates into one flat bar. */
const WAVE_SKIN = {
  waveColor: "#e4d8c2",
  progressColor: "#e2582a",
  cursorColor: "#c1481f",
  cursorWidth: 2,
  barWidth: 3,
  barGap: 1.5,
  barRadius: 2,
};

function makeWave(container, height, withTimeline) {
  const plugins = [];
  if (withTimeline && window.WaveSurfer && WaveSurfer.Timeline) {
    plugins.push(WaveSurfer.Timeline.create({ height: 14 }));
  }
  return WaveSurfer.create({
    container,
    height,
    media: el.resultAudio,
    peaks: [state.wavePeaks || Array.from({ length: 128 }, () => 0.4)],
    duration: state.takeDuration || el.resultAudio.duration || 0,
    interact: true,
    plugins,
    ...WAVE_SKIN,
  });
}

function rebuildDockWave() {
  if (state.dockWS) state.dockWS.destroy();
  el.resultWaveform.innerHTML = "";
  state.dockWS = makeWave(el.resultWaveform, 44, false);
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

/* Best case: the backend segmented every turn's start from the audio itself.
   Apply those directly — no estimating, no snapping. */
function applyExactTurnTimings(turnStarts, durationSeconds) {
  state.timingsExact = false;
  const turns = state.resultTurns;
  if (!Array.isArray(turnStarts) || turnStarts.length !== turns.length
      || !turns.length || !durationSeconds) return false;
  turns.forEach((turn, i) => {
    const oldStart = turn.startRatio;
    const oldSpan = turn.endRatio - oldStart || 1e-9;
    const newStart = turnStarts[i] / durationSeconds;
    const newEnd = (i + 1 < turns.length ? turnStarts[i + 1] : durationSeconds) / durationSeconds;
    turn.sentences.forEach((s) => {
      s.startRatio = newStart + ((s.startRatio - oldStart) / oldSpan) * (newEnd - newStart);
      s.endRatio = newStart + ((s.endRatio - oldStart) / oldSpan) * (newEnd - newStart);
    });
    turn.startRatio = newStart;
    turn.endRatio = newEnd;
  });
  state.timingsExact = true;
  return true;
}

/* The backend renders the take in chunks and reports exactly where each chunk
   starts and how many turns it covers. Those are hard anchors: retime the
   word-proportional estimate piecewise between them, so timing error can never
   accumulate past a chunk (~80s). Falls back silently when the take has no
   timing map (cache hits, recovered takes). */
function applyChunkAnchors(anchors, durationSeconds) {
  state.anchoredTurns = new Set();
  const turns = state.resultTurns;
  if (!anchors || !turns.length || !durationSeconds) return;
  const starts = anchors.starts || [];
  const counts = anchors.counts || [];
  if (starts.length !== counts.length || starts.length < 2) return;
  if (counts.reduce((a, b) => a + b, 0) !== turns.length) return; // map doesn't fit this script
  let first = 0;
  for (let k = 0; k < counts.length; k += 1) {
    const chunkTurns = turns.slice(first, first + counts[k]);
    const newStart = starts[k] / durationSeconds;
    const newEnd = (k + 1 < starts.length ? starts[k + 1] : durationSeconds) / durationSeconds;
    const oldStart = chunkTurns[0].startRatio;
    const oldSpan = chunkTurns[chunkTurns.length - 1].endRatio - oldStart || 1e-9;
    const scale = (newEnd - newStart) / oldSpan;
    chunkTurns.forEach((turn) => {
      const remap = (r) => newStart + (r - oldStart) * scale;
      turn.sentences.forEach((s) => {
        s.startRatio = remap(s.startRatio);
        s.endRatio = remap(s.endRatio);
      });
      turn.startRatio = remap(turn.startRatio);
      turn.endRatio = remap(turn.endRatio);
    });
    state.anchoredTurns.add(first); // this boundary is measured — never re-snap it
    first += counts[k];
  }
}

/* Word-count apportioning drifts because speech has pauses the text doesn't:
   captions were landing ahead of the audio. The RMS peaks envelope (already
   fetched for the waveform) shows exactly where speech pauses, so snap each
   turn boundary to the nearest silence→speech onset and rescale the turn's
   sentence timings into the corrected span. */
function refineTurnTimings() {
  if (state.timingsExact) return; // backend measured every turn; nothing to refine
  const peaks = state.wavePeaks;
  const turns = state.resultTurns;
  if (!peaks || peaks.length < 64 || !turns || turns.length < 2) return;
  const n = peaks.length;
  // Peaks are normalized to max 1. The quietest few percent is the pause floor;
  // stay just above it, but never so high that quiet speech reads as silence.
  const sorted = [...peaks].slice().sort((a, b) => a - b);
  const noiseFloor = sorted[Math.floor(n * 0.03)] || 0;
  const speechThresh = Math.min(0.2, Math.max(0.08, noiseFloor + 0.06));

  const anchored = state.anchoredTurns || new Set();
  const adjusted = [0];
  let drift = 0; // estimate error accumulates through the take; carry the correction forward
  for (let i = 1; i < turns.length; i += 1) {
    if (anchored.has(i)) {
      // Measured chunk boundary — exact already; snapping could only hurt.
      adjusted.push(Math.max(turns[i].startRatio, adjusted[i - 1] + 1 / n));
      drift = 0;
      continue;
    }
    const est = Math.round(turns[i].startRatio * n) + drift;
    // Search within 35% of this turn's own span: wide enough for real drift,
    // too narrow to ever snap to a neighboring turn's pause.
    const radius = Math.max(3, Math.round((est - adjusted[i - 1] * n) * 0.35));
    let best = -1;
    let bestDist = Infinity;
    const lo = Math.max(1, est - radius);
    const hi = Math.min(n - 1, est + radius);
    for (let b = lo; b <= hi; b += 1) {
      // A bucket over the threshold right after one under it = speech onset.
      if (peaks[b] >= speechThresh && peaks[b - 1] < speechThresh) {
        const dist = Math.abs(b - est);
        if (dist < bestDist) { bestDist = dist; best = b; }
      }
    }
    let candidate;
    if (best >= 0) {
      candidate = best / n;
      drift = best - Math.round(turns[i].startRatio * n);
    } else {
      candidate = Math.min(1, Math.max(0, est / n));
    }
    // Keep boundaries strictly increasing — a bad snap must not swallow a turn.
    adjusted.push(Math.max(candidate, adjusted[i - 1] + 1 / n));
  }

  turns.forEach((turn, i) => {
    const oldStart = turn.startRatio;
    const oldSpan = turn.endRatio - oldStart || 1e-9;
    const newStart = adjusted[i];
    const newEnd = i + 1 < turns.length ? adjusted[i + 1] : 1;
    turn.startRatio = newStart;
    turn.endRatio = newEnd;
    turn.sentences.forEach((s) => {
      s.startRatio = newStart + ((s.startRatio - oldStart) / oldSpan) * (newEnd - newStart);
      s.endRatio = newStart + ((s.endRatio - oldStart) / oldSpan) * (newEnd - newStart);
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
      // Scroll only the transcript's own box — scrollIntoView also scrolls
      // ancestor containers, which visibly yanked the whole dialog on every
      // turn change.
      const box = row.parentElement;
      if (!box || row.offsetParent === null || box.scrollHeight <= box.clientHeight) return;
      const rowRect = row.getBoundingClientRect();
      const boxRect = box.getBoundingClientRect();
      const target = box.scrollTop + (rowRect.top - boxRect.top)
        - (box.clientHeight - rowRect.height) / 2;
      box.scrollTo({ top: Math.max(0, target), behavior: "smooth" });
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

/* ---------------- Sound ----------------
   One choice instead of a mixing desk: Original is the untouched take;
   Studio/Warm/Bright are loudness-normalized to the podcast standard, with an
   optional tone shelf. Applied live through a small Web Audio graph, and
   downloads are rendered server-side with the same settings. */
const SOUND_MODES = {
  original: { tone: "neutral", norm: false, hint: "The untouched take, exactly as generated" },
  studio: { tone: "neutral", norm: true, hint: "Loudness normalized to −16 LUFS podcast standard" },
  warm: { tone: "warm", norm: true, hint: "Normalized · warm tone — richer lows, softer highs" },
  bright: { tone: "bright", norm: true, hint: "Normalized · bright tone — crisper, more present" },
};
const DEFAULT_SOUND = "studio";

const TONE_CURVES = {
  neutral: { low: 0, high: 0 },
  warm: { low: 4, high: -3 },
  bright: { low: -2, high: 4.5 },
};

const polish = { ctx: null, low: null, high: null, gain: null, limiter: null, analyser: null, mode: DEFAULT_SOUND, audioId: null };

const NORM_TARGET_DB = -16; // keep in sync with the server's NORM_TARGET_DB

// Live-preview gain for loudness normalization, from the loudness the peaks
// endpoint measured. The limiter at the end of the graph catches peaks.
function normalizeGain() {
  const measured = state.takeLoudnessDb;
  if (!Number.isFinite(measured)) return 1;
  return Math.min(8, Math.max(0.05, 10 ** ((NORM_TARGET_DB - measured) / 20)));
}

function ensureAudioGraph() {
  if (polish.ctx) return true;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return false;
  try {
    polish.ctx = new Ctx();
    // createMediaElementSource may only be called once per element; from here
    // on the element's audio reaches the speakers through this graph.
    const source = polish.ctx.createMediaElementSource(el.resultAudio);
    polish.low = polish.ctx.createBiquadFilter();
    polish.low.type = "lowshelf";
    polish.low.frequency.value = 250;
    polish.high = polish.ctx.createBiquadFilter();
    polish.high.type = "highshelf";
    polish.high.frequency.value = 3500;
    polish.gain = polish.ctx.createGain();
    // A brick-wall-ish limiter always sits last: the old graph applied makeup
    // gain after compression with nothing to catch peaks, which clipped and
    // was heard as distortion.
    polish.limiter = polish.ctx.createDynamicsCompressor();
    polish.limiter.threshold.value = -1.5;
    polish.limiter.knee.value = 0;
    polish.limiter.ratio.value = 20;
    polish.limiter.attack.value = 0.003;
    polish.limiter.release.value = 0.12;
    source.connect(polish.low);
    polish.low.connect(polish.high);
    polish.high.connect(polish.gain);
    polish.gain.connect(polish.limiter);
    polish.limiter.connect(polish.ctx.destination);
    // Tap for the speaker orbs — reads the signal, outputs nowhere.
    polish.analyser = polish.ctx.createAnalyser();
    polish.analyser.fftSize = 512;
    polish.limiter.connect(polish.analyser);
    applySound();
    return true;
  } catch (error) {
    console.warn("Audio graph unavailable:", error);
    polish.ctx = null;
    return false;
  }
}

function applySound() {
  const mode = SOUND_MODES[polish.mode] || SOUND_MODES[DEFAULT_SOUND];
  el.soundSeg.querySelectorAll("button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.sound === polish.mode);
  });
  el.soundHint.textContent = `${mode.hint} · downloads match what you hear`;

  updateExportLinks();

  if (!polish.ctx) return;
  const curve = TONE_CURVES[mode.tone] || TONE_CURVES.neutral;
  polish.low.gain.value = curve.low;
  polish.high.gain.value = curve.high;
  polish.gain.gain.value = mode.norm ? normalizeGain() : 1;
}

/* Downloads carry the sound choice: Original streams the stored take
   untouched, anything else is rendered by the server. */
function updateExportLinks() {
  if (!polish.audioId) return;
  const plain = `/api/audio/${polish.audioId}`;
  const mode = SOUND_MODES[polish.mode] || SOUND_MODES[DEFAULT_SOUND];
  const isOriginal = polish.mode === "original";
  const query = `speed=1&tone=${mode.tone}&level=false&norm=${mode.norm}`;
  el.downloadBtn.href = isOriginal ? plain : `${plain}/export?${query}&fmt=wav`;
  el.stageDownloadBtn.href = el.downloadBtn.href;
  el.downloadMp3Btn.href = isOriginal ? `${plain}.mp3` : `${plain}/export?${query}&fmt=mp3`;
  el.stageDownloadMp3Btn.href = el.downloadMp3Btn.href;
}

// Encoding/rendering happens on the server when the link is clicked, and a
// long take takes real time, so say so instead of appearing to do nothing.
function noteExportStarted(kind) {
  el.polishStatus.hidden = false;
  el.polishStatus.textContent = `Preparing your ${kind}… the download starts when it's ready (long takes can take a minute or two).`;
  clearTimeout(noteExportStarted.timer);
  noteExportStarted.timer = setTimeout(() => { el.polishStatus.hidden = true; }, 20000);
}

const TAKE_GONE_MESSAGE =
  "This take is no longer on the server — it is held in memory, so a Space restart or " +
  "redeploy clears it. Generate a new one; download it before deploying next time.";

function showTakeGone() {
  el.polishStatus.hidden = false;
  el.polishStatus.textContent = TAKE_GONE_MESSAGE;
  el.dockEmpty.hidden = false;
  el.dockEmpty.textContent = TAKE_GONE_MESSAGE;
  setStatus("error", "The finished audio is no longer available on the server.");
}

// A dead link would otherwise surface as the browser's opaque "File wasn't
// available on site". Probe the stored take (one byte) before letting the
// download proceed, and say plainly when it has been cleared.
[["downloadBtn", "WAV"], ["stageDownloadBtn", "WAV"],
 ["downloadMp3Btn", "MP3"], ["stageDownloadMp3Btn", "MP3"]].forEach(([id, kind]) => {
  el[id].addEventListener("click", async (event) => {
    const anchor = el[id];
    if (anchor.dataset.verified === "1") {
      anchor.dataset.verified = "";
      return;  // second pass: this is the real download
    }
    if (!polish.audioId) return;
    event.preventDefault();
    let alive = false;
    try {
      // Probe the raw take, never the export URL — that would start a render.
      const probe = await fetch(`/api/audio/${polish.audioId}`, { headers: { Range: "bytes=0-0" } });
      alive = probe.ok || probe.status === 206;
    } catch {
      alive = false;
    }
    if (!alive) {
      showTakeGone();
      return;
    }
    if (kind === "MP3" || polish.mode !== "original") noteExportStarted(kind);
    anchor.dataset.verified = "1";
    anchor.click();
  });
});

// Same story if playback itself can't load the take.
el.resultAudio.addEventListener("error", () => {
  if (el.resultAudio.getAttribute("src")) showTakeGone();
});

el.soundSeg.querySelectorAll("button").forEach((btn) => {
  btn.addEventListener("click", () => {
    polish.mode = btn.dataset.sound;
    ensureAudioGraph();
    if (polish.ctx && polish.ctx.state === "suspended") polish.ctx.resume();
    applySound();
  });
});
// A suspended context would silence the element now that audio routes through it.
el.resultAudio.addEventListener("play", () => {
  if (polish.ctx && polish.ctx.state === "suspended") polish.ctx.resume();
});
applySound();  // paint the default chip before any take exists

/* ---------------- Speaker orbs ----------------
   One glowing orb per speaker on the stage; the one who's talking swells with
   the live signal level (Web Audio analyser), the others drift idle. */
const orbs = { bySpeaker: new Map(), raf: 0, level: new Uint8Array(0) };

function buildStageOrbs() {
  orbs.bySpeaker.clear();
  el.stageOrbs.innerHTML = "";
  const speakers = [...new Set(state.resultTurns.map((t) => t.speaker))].sort((a, b) => a - b);
  el.stageOrbs.hidden = speakers.length === 0;
  speakers.forEach((s, i) => {
    const wrap = document.createElement("div");
    wrap.className = "stage-orb-wrap";
    const orb = document.createElement("div");
    orb.className = "stage-orb";
    orb.style.setProperty("--orb-color", slotColor(s - 1));
    orb.style.animationDelay = `${i * 0.7}s`;
    const label = document.createElement("div");
    label.className = "stage-orb-label";
    label.textContent = slotVoiceLabel(s - 1);
    wrap.append(orb, label);
    el.stageOrbs.append(wrap);
    orbs.bySpeaker.set(s, orb);
  });
}

function currentAudioLevel() {
  if (!polish.analyser) return 0.5; // no Web Audio — pulse at a fixed level
  if (orbs.level.length !== polish.analyser.fftSize) {
    orbs.level = new Uint8Array(polish.analyser.fftSize);
  }
  polish.analyser.getByteTimeDomainData(orbs.level);
  let sum = 0;
  for (let i = 0; i < orbs.level.length; i += 1) {
    const d = (orbs.level[i] - 128) / 128;
    sum += d * d;
  }
  // RMS of speech sits low; stretch it so the orb visibly moves.
  return Math.min(1, Math.sqrt(sum / orbs.level.length) * 4);
}

function orbFrame() {
  const audio = el.resultAudio;
  const activeTurn = state.resultTurns[state.activeSyncIndex];
  const activeSpeaker = activeTurn ? activeTurn.speaker : -1;
  const level = audio.paused ? 0 : currentAudioLevel();
  orbs.bySpeaker.forEach((orb, speaker) => {
    const talking = !audio.paused && speaker === activeSpeaker;
    orb.classList.toggle("talking", talking);
    if (talking) {
      // Modest scale — the energy shows in the glow, and the orb must never
      // swell over the name label beneath it.
      orb.style.transform = `scale(${(1.04 + level * 0.18).toFixed(3)})`;
      orb.style.setProperty("--glow", (0.3 + level * 0.7).toFixed(3));
    } else {
      orb.style.transform = "";
      orb.style.removeProperty("--glow");
    }
  });
  orbs.raf = audio.paused ? 0 : requestAnimationFrame(orbFrame);
}

function startOrbLoop() {
  if (!orbs.raf && orbs.bySpeaker.size) orbs.raf = requestAnimationFrame(orbFrame);
}

function stopOrbLoop() {
  cancelAnimationFrame(orbs.raf);
  orbs.raf = 0;
  orbs.bySpeaker.forEach((orb) => {
    orb.classList.remove("talking");
    orb.style.transform = "";
  });
}

el.resultAudio.addEventListener("play", () => {
  // Building the graph needs a user gesture anyway, and play is one — so this
  // is the moment the analyser can come to life.
  ensureAudioGraph();
  if (polish.ctx && polish.ctx.state === "suspended") polish.ctx.resume();
  startOrbLoop();
});
el.resultAudio.addEventListener("pause", stopOrbLoop);
el.resultAudio.addEventListener("ended", stopOrbLoop);

/* Now Playing stage */
function openPlayerStage() {
  showStagePane("player");
  el.stageTitle.textContent = (state.resultTitle || "Untitled conversation").toUpperCase();
  const audio = el.resultAudio;
  updateStageCaption(audio.duration ? audio.currentTime / audio.duration : 0);
  // Already open when a render hands off from the generating pane — calling
  // showModal() twice throws InvalidStateError.
  if (!el.playerStage.open) el.playerStage.showModal();
  // Built after showModal so the container has real width to render into.
  if (state.stageWS) state.stageWS.destroy();
  el.stageWaveform.innerHTML = "";
  state.stageWS = makeWave(el.stageWaveform, 80, true);
}

el.playerStage.addEventListener("close", () => {
  if (state.stageWS) {
    state.stageWS.destroy();
    state.stageWS = null;
    el.stageWaveform.innerHTML = "";
  }
});

el.openPlayerBtn.addEventListener("click", openPlayerStage);
el.stageCloseBtn.addEventListener("click", () => el.playerStage.close());
el.playerStage.addEventListener("click", (e) => { if (e.target === el.playerStage) el.playerStage.close(); });

el.stageScriptToggle.addEventListener("click", () => {
  el.stageTranscript.hidden = !el.stageTranscript.hidden;
  el.stageScriptToggle.textContent = el.stageTranscript.hidden ? "View full script" : "Hide script";
  if (!el.stageTranscript.hidden) {
    const active = state.resultTurns[state.activeSyncIndex];
    const target = (active && active.rows && active.rows[1]) || el.stageTranscript;
    target.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
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
  const running = stage !== "complete" && stage !== "error" && stage !== "cancelled";
  el.statusCard.classList.add("visible");
  el.statusCard.classList.toggle("active", running);
  el.statusCard.classList.toggle("complete", stage === "complete");
  el.statusCard.classList.toggle("error", stage === "error" || stage === "cancelled");
  el.statusTitle.textContent = title;
  el.statusDesc.textContent = fallbackText || defaultDesc || "";
  el.genStageDesc.textContent = `${title} — ${fallbackText || defaultDesc || ""}`;
  el.stopGenBtn.hidden = !running;
  el.genStageStopBtn.hidden = !running;
}

let generateAbort = null;
el.stopGenBtn.addEventListener("click", () => {
  if (generateAbort) generateAbort.abort();
});

/* ---- Real generation progress ----------------------------------------
   The backend's own pct is a time-based hint that pins at 88% after 90s —
   useless on a 70-minute render. Its status line carries the real signal
   ("Rendering wave 3/21 …"), so drive progress off completed waves and
   keep a client-side clock ticking between events. */
const progress = {
  startedAt: 0, wave: 0, totalWaves: 0, ticker: null, label: "",
  waveStartedAt: 0, waveDurations: [],
  genStartedAt: 0, warmupSecs: 0,
  chunks: 0, scriptWords: 0,
};

/* Per-model timing learned from completed runs, so single-wave renders (which
   never produce a completed-wave timing mid-run) can still show an estimated
   percentage and ETA instead of an endless sweep. */
const CALIBRATION_KEY = "chorus-gen-calibration-v1";

function loadCalibration() {
  try { return JSON.parse(localStorage.getItem(CALIBRATION_KEY)) || {}; } catch { return {}; }
}

function calibrationFor(model) {
  return loadCalibration()[model] || {};
}

function saveCalibration(model, patch) {
  try {
    const all = loadCalibration();
    all[model] = { ...all[model], ...patch };
    localStorage.setItem(CALIBRATION_KEY, JSON.stringify(all));
  } catch { /* private mode etc. — estimates just stay at defaults */ }
}

// Expected seconds to render one wave: audio length of one chunk times how many
// seconds of compute one second of audio costs (measured on past runs; the
// default is a rough 7B figure that a single completed run replaces).
function waveEstimateSecs() {
  if (!progress.chunks || !progress.scriptWords) return 0;
  const audioSecs = (progress.scriptWords / 150) * 60;
  const chunkAudio = audioSecs / progress.chunks;
  const factor = calibrationFor(state.model).secsPerAudioSec || 1.6;
  return Math.max(15, chunkAudio * factor);
}

// The backend log names the batching plan before the first wave starts.
function noteChunksFromLog(logText) {
  if (!logText || progress.chunks) return;
  const m = logText.match(/Parallel mode:\s*(\d+)\s*chunks?(?:\s*\(fast-start\s*(\d+)\))?,\s*batches of up to\s*(\d+)/i);
  if (m) {
    progress.chunks = Number(m[1]);
    const fast = Number(m[2] || 0);
    if (!progress.totalWaves) {
      progress.totalWaves = fast > 1
        ? 1 + Math.ceil((Number(m[1]) - fast) / Number(m[3]))
        : Math.ceil(Number(m[1]) / Number(m[3]));
    }
  } else if (/Short script: single-pass/i.test(logText)) {
    progress.chunks = 1;
    if (!progress.totalWaves) progress.totalWaves = 1;
  }
}

// GPU warm-up is real work but it isn't generation — folding it into "elapsed"
// makes a render look slower than it was and skews the ETA, so the clock is
// rebased the moment synthesis actually starts and warm-up is reported apart.
function markGenerationStarted() {
  if (progress.genStartedAt || !progress.startedAt) return;
  progress.genStartedAt = Date.now();
  progress.warmupSecs = (progress.genStartedAt - progress.startedAt) / 1000;
}

function generationSeconds() {
  const base = progress.genStartedAt || progress.startedAt;
  return base ? (Date.now() - base) / 1000 : 0;
}

function formatMinutes(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return null;
  if (seconds < 90) return `${Math.round(seconds)}s`;
  const mins = Math.round(seconds / 60);
  return mins < 60 ? `${mins} min` : `${Math.floor(mins / 60)}h ${mins % 60}m`;
}

function paintProgress() {
  if (!progress.startedAt) return;
  const elapsed = generationSeconds();
  const parts = [];
  let pctText = "Starting…";
  if (!progress.genStartedAt) {
    // Still warming up — say so plainly instead of ticking a generation clock.
    // The bar stays present: hiding it would resize the dialog.
    el.progressMeta.hidden = false;
    el.progressTrack.hidden = false;
    el.genStageTrack.hidden = false;
    el.genStagePct.classList.add("warming");
    const expectedWarmup = calibrationFor(state.model).warmupSecs || 0;
    let warm;
    if (expectedWarmup && elapsed > 10) {
      // Ramp toward the last observed cold start — an estimate in motion beats
      // a bare sweep that looks frozen for three minutes.
      const width = `${(Math.min(0.95, elapsed / expectedWarmup) * 100).toFixed(1)}%`;
      el.progressTrack.classList.remove("indeterminate");
      el.genStageTrack.classList.remove("indeterminate");
      el.progressFill.style.width = width;
      el.genStageFill.style.width = width;
      warm = `GPU cold start · ${formatMinutes(elapsed) || "0s"} of ~${formatMinutes(expectedWarmup)} (last run)`;
    } else {
      el.progressTrack.classList.add("indeterminate");
      el.genStageTrack.classList.add("indeterminate");
      warm = `GPU warming up · ${formatMinutes(elapsed) || "0s"}` +
        (elapsed > 20 ? " — a cold start loads the whole model, usually a few minutes" : "");
    }
    el.progressMeta.textContent = warm;
    el.genStagePct.textContent = "Warming up…";
    el.genStageMeta.textContent = warm;
    return;
  }
  el.genStagePct.classList.remove("warming");
  if (progress.chunksTotal && progress.chunksDone) {
    // Streamed chunks are the ground truth — every rendered chunk is a real,
    // gate-approved fraction of the take. Between chunk deliveries (waves
    // land minutes apart), creep toward the next milestone on the calibrated
    // wave estimate so the bar never sits frozen; the monotonic guard below
    // stops it ever moving backwards.
    let frac = progress.chunksDone / progress.chunksTotal;
    const est = waveEstimateSecs();
    if (est && progress.lastChunkAt) {
      const since = (Date.now() - progress.lastChunkAt) / 1000;
      frac += Math.min(since / est, 0.92) * (1 - frac) * 0.9;
    }
    frac = Math.min(0.985, Math.max(frac, progress.peakFrac || 0));
    progress.peakFrac = frac;
    el.progressTrack.hidden = false;
    el.genStageTrack.hidden = false;
    el.progressTrack.classList.remove("indeterminate");
    el.genStageTrack.classList.remove("indeterminate");
    const width = `${Math.max(1.5, frac * 100).toFixed(1)}%`;
    el.progressFill.style.width = width;
    el.genStageFill.style.width = width;
    pctText = `${Math.round(frac * 100)}%`;
    document.title = `${pctText} · Chorus`;
    parts.push(`${progress.chunksDone} of ${progress.chunksTotal} chunks`);
    const doneFrac = progress.chunksDone / progress.chunksTotal;
    if (doneFrac > 0.05 && doneFrac < 1) {
      const eta = formatMinutes(elapsed * (1 - doneFrac) / doneFrac);
      if (eta) parts.push(`~${eta} left`);
    }
  } else if (progress.totalWaves) {
    const done = Math.max(0, progress.wave - 1);
    // Waves land in steps minutes apart, so a bare completed-wave count would
    // sit at 0% through all of wave 1 and then jump. Once a wave has finished
    // we know roughly how long one takes, so interpolate inside the current
    // one; until then the bar runs indeterminate rather than faking a number.
    const avgWave = progress.waveDurations.length
      ? progress.waveDurations.reduce((a, b) => a + b, 0) / progress.waveDurations.length
      : 0;
    const estWave = avgWave || waveEstimateSecs();
    const sinceWave = progress.waveStartedAt ? (Date.now() - progress.waveStartedAt) / 1000 : 0;
    // Real timings from completed waves win; before any exist (always true for a
    // single-wave render) fall back to the calibrated estimate, capped so the
    // bar never claims to finish before the model does.
    const inWave = estWave
      ? Math.min(avgWave ? 0.98 : 0.95, sinceWave / estWave)
      : 0;
    const indeterminate = !estWave;
    let frac = (done + inWave) / progress.totalWaves;
    if (!indeterminate) {
      frac = Math.max(frac, progress.peakFrac || 0);
      progress.peakFrac = frac;
    }

    el.progressTrack.hidden = false;
    el.genStageTrack.hidden = false;
    el.progressTrack.classList.toggle("indeterminate", indeterminate);
    el.genStageTrack.classList.toggle("indeterminate", indeterminate);
    if (!indeterminate) {
      const width = `${Math.max(1.5, frac * 100).toFixed(1)}%`;
      el.progressFill.style.width = width;
      el.genStageFill.style.width = width;
      pctText = `${Math.round(frac * 100)}%`;
      document.title = `${pctText} · Chorus`;
    } else {
      pctText = "Rendering…";
    }

    parts.push(`Wave ${Math.max(1, progress.wave)} of ${progress.totalWaves}`);
    if (done >= 1) {
      const remaining = (elapsed / (done + inWave)) * (progress.totalWaves - done - inWave);
      const eta = formatMinutes(remaining);
      if (eta) parts.push(`~${eta} left`);
    } else if (!avgWave && estWave) {
      const remaining = estWave * progress.totalWaves - sinceWave;
      const eta = formatMinutes(remaining);
      if (eta) parts.push(`~${eta} left (est.)`);
    }
  } else if (progress.label) {
    parts.push(progress.label);
  }
  parts.push(`${formatMinutes(elapsed) || "0s"} generating`);
  if (progress.warmupSecs >= 5) {
    parts.push(`${formatMinutes(progress.warmupSecs)} warm-up`);
  }
  const meta = parts.join(" · ");
  el.progressMeta.hidden = false;
  el.progressMeta.textContent = meta;
  el.genStagePct.textContent = pctText;
  el.genStageMeta.textContent = meta;
}

/* The stage doubles as generation mission-control, then becomes the player. */
function showStagePane(which) {
  el.stageGenPane.hidden = which !== "generating";
  el.stagePlayPane.hidden = which !== "player";
}

function openGenerateStage(title) {
  el.genStageTitle.textContent = (title || "Untitled conversation").toUpperCase();
  el.genStagePct.textContent = "Starting…";
  el.genStageTrack.hidden = false;
  el.genStageTrack.classList.add("indeterminate");
  el.genStageFill.style.width = "0%";
  el.genStageLog.textContent = "";
  showStagePane("generating");
  if (!el.playerStage.open) el.playerStage.showModal();
}

el.genStageStopBtn.addEventListener("click", () => {
  if (generateAbort) generateAbort.abort();
});

function startProgress() {
  progress.startedAt = Date.now();
  progress.wave = 0;
  progress.totalWaves = 0;
  progress.label = "";
  progress.waveStartedAt = 0;
  progress.waveDurations = [];
  progress.genStartedAt = 0;
  progress.warmupSecs = 0;
  progress.chunks = 0;
  progress.chunksDone = 0;
  progress.chunksTotal = 0;
  progress.lastChunkAt = 0;
  progress.peakFrac = 0;
  progress.scriptWords = state.turns.reduce((n, t) => n + (t.text || "").split(/\s+/).filter(Boolean).length, 0);
  resetPreview();
  el.progressFill.style.width = "0%";
  el.progressTrack.classList.add("indeterminate");
  el.genStageTrack.classList.add("indeterminate");
  el.progressTrack.hidden = false;
  clearInterval(progress.ticker);
  progress.ticker = setInterval(paintProgress, 1000);
  paintProgress();
}

function stopProgress() {
  clearInterval(progress.ticker);
  progress.ticker = null;
  progress.startedAt = 0;
  el.progressTrack.hidden = true;
  el.progressMeta.hidden = true;
  el.genStagePct.classList.remove("warming");
  document.title = "Chorus — AI Voice Studio";
}

function noteProgressFromStatus(text) {
  if (!text) return;
  const m = text.match(/wave\s+(\d+)\s*\/\s*(\d+)/i);
  if (m) {
    const wave = Number(m[1]);
    if (wave !== progress.wave) {
      // A new wave means the previous one just finished — time it, so the
      // in-wave interpolation and the ETA have something real to work from.
      if (progress.wave && progress.waveStartedAt) {
        progress.waveDurations.push((Date.now() - progress.waveStartedAt) / 1000);
      }
      progress.waveStartedAt = Date.now();
      progress.wave = wave;
    }
    progress.totalWaves = Number(m[2]);
  } else if (/re-?rolling/i.test(text)) {
    progress.label = "Re-rolling flagged chunks";
  }
  paintProgress();
}

el.logToggleBtn.addEventListener("click", () => {
  const visible = el.logBox.classList.toggle("visible");
  el.logToggleBtn.textContent = visible ? "Hide generation log" : "View generation log";
});

/* ---------------- Take download & presentation ---------------- */
async function fetchPeaks(audioId) {
  try {
    const res = await fetch(`/api/audio/${audioId}/peaks?buckets=2048`);
    if (!res.ok) return null;
    const body = await res.json();
    state.takeLoudnessDb = Number.isFinite(body.loudness_db) ? body.loudness_db : null;
    return body.peaks || null;
  } catch {
    return null;  // waveform is decoration; never fail a finished take over it
  }
}

// The take is streamed straight from the server: the player seeks with Range
// requests and the download buttons are plain links, so the browser's own
// download manager handles hundreds of megabytes instead of the page buffering
// the whole file before anything can be heard.
async function presentTake(audioId, durationSeconds, snapshot, anchors = null) {
  setStatus("complete");
  const url = `/api/audio/${audioId}`;
  el.resultAudio.src = url;
  polish.audioId = audioId;
  el.downloadMp3Btn.hidden = false;
  el.stageDownloadMp3Btn.hidden = false;
  el.polishStatus.hidden = true;
  updateExportLinks();
  el.audioDuration.textContent = formatDuration(durationSeconds);
  el.playerTime.textContent = `0:00 / ${formatClock(durationSeconds)}`;
  el.stageTime.textContent = `0:00 / ${formatClock(durationSeconds)}`;
  setPlayIcons("►");
  applySound();  // carry the listener's sound choice onto the new take
  buildSyncedTranscript(snapshot);
  el.dockEmpty.hidden = true;
  el.resultBlock.classList.add("visible");
  state.takeDuration = durationSeconds;
  if (!applyExactTurnTimings(anchors && anchors.turnStarts, durationSeconds)) {
    applyChunkAnchors(anchors, durationSeconds);
  }
  buildStageOrbs();
  state.wavePeaks = await fetchPeaks(audioId);
  refineTurnTimings();
  applySound();  // the new take's measured loudness changes the live norm gain
  rebuildDockWave();
  openPlayerStage();
}

/* On load: if the server still holds a finished take this page doesn't know
   about (lost tab, accidental navigation), offer to recover it. */
async function checkLastTake() {
  try {
    const res = await fetch("/api/last-take", { cache: "no-store" });
    if (!res.ok) return;
    const info = await res.json();
    const age = info.age_seconds < 120
      ? "moments ago"
      : `${Math.round(info.age_seconds / 60)} min ago`;
    el.dockEmpty.textContent =
      `A finished take (${formatDuration(info.duration)}, generated ${age}) is still on the server.`;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-accent";
    btn.style.cssText = "display:block; margin:12px auto 0;";
    btn.textContent = "Recover last take";
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try {
        el.generationTime.textContent = "--";
        el.resultModel.textContent = "recovered";
        state.resultTitle = "Recovered take";
        await presentTake(info.audio_id, info.duration, []);
      } catch (error) {
        setStatus("error", error.message);
        btn.disabled = false;
      }
    });
    el.dockEmpty.append(btn);
  } catch { /* nothing to recover */ }
}

/* ---------------- Live preview while rendering ----------------
   The backend streams each rendered chunk (post quality gate) as a URL.
   Chunks are decoded and scheduled through Web Audio with the same 0.25 s
   linear crossfade the server bakes into the final take, so what plays here
   is what the take will sound like. This is deliberately independent of the
   main player: when the full take lands, playback hands off to it. */
const PREVIEW_XF = 0.25;
const preview = {
  ctx: null, master: null, total: 0, nextIndex: 0, buffers: new Map(),
  sources: [], timeline: [], nextCtxTime: 0, nextTakeTime: 0,
  active: false, muted: false, done: true, normGain: 1,
};

function resetPreview() {
  stopPreview(0);
  preview.total = 0;
  preview.nextIndex = 0;
  preview.buffers.clear();
  preview.timeline = [];
  preview.nextCtxTime = 0;
  preview.nextTakeTime = 0;
  preview.normGain = 1;
  preview.done = false;
  el.genPreviewRow.hidden = true;
}

function stopPreview(fadeSecs = 0.3) {
  preview.done = true;
  preview.active = false;
  if (!preview.ctx) return;
  const ctx = preview.ctx;
  preview.ctx = null;
  try {
    if (preview.master && fadeSecs > 0) {
      preview.master.gain.setValueAtTime(preview.master.gain.value, ctx.currentTime);
      preview.master.gain.linearRampToValueAtTime(0, ctx.currentTime + fadeSecs);
      setTimeout(() => ctx.close().catch(() => {}), fadeSecs * 1000 + 100);
    } else {
      ctx.close().catch(() => {});
    }
  } catch { /* already closed */ }
  preview.master = null;
  preview.sources = [];
}

function ensurePreviewCtx() {
  if (preview.ctx) return true;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return false;
  try {
    preview.ctx = new Ctx();
    preview.master = preview.ctx.createGain();
    preview.master.connect(preview.ctx.destination);
    if (preview.ctx.state === "suspended") preview.ctx.resume().catch(() => {});
    return true;
  } catch {
    preview.ctx = null;
    return false;
  }
}

// Match the default Studio sound: measure the first chunk's gated loudness
// and level the whole preview toward the same target the player uses.
function previewNormGain(buf) {
  const data = buf.getChannelData(0);
  const frame = Math.max(1, Math.round(buf.sampleRate * 0.05));
  let maxRms = 0;
  const rmses = [];
  for (let i = 0; i + frame <= data.length; i += frame) {
    let sum = 0;
    for (let j = i; j < i + frame; j += 1) sum += data[j] * data[j];
    const r = Math.sqrt(sum / frame);
    rmses.push(r);
    if (r > maxRms) maxRms = r;
  }
  const active = rmses.filter((r) => r > maxRms * 0.2);
  if (!active.length) return 1;
  const rms = Math.sqrt(active.reduce((a, r) => a + r * r, 0) / active.length);
  const db = 20 * Math.log10(Math.max(rms, 1e-6));
  return Math.min(8, Math.max(0.05, 10 ** ((NORM_TARGET_DB - db) / 20)));
}

async function handleStreamChunk(evt) {
  if (preview.done || evt.chunk_index == null || !evt.chunk_url) return;
  preview.total = evt.chunk_total || preview.total;
  if (evt.chunk_index < preview.nextIndex || preview.buffers.has(evt.chunk_index)) return;
  try {
    const res = await fetch(evt.chunk_url);
    if (!res.ok || preview.done) return;
    if (!ensurePreviewCtx()) return;
    const audioBuf = await preview.ctx.decodeAudioData(await res.arrayBuffer());
    if (preview.done) return;
    preview.buffers.set(evt.chunk_index, audioBuf);
    pumpPreview();
  } catch { /* preview is best-effort; the full take still arrives */ }
}

function pumpPreview() {
  while (!preview.done && preview.buffers.has(preview.nextIndex)) {
    const buf = preview.buffers.get(preview.nextIndex);
    preview.buffers.delete(preview.nextIndex);
    schedulePreviewChunk(buf, preview.nextIndex);
    preview.nextIndex += 1;
  }
  updatePreviewUI();
}

function schedulePreviewChunk(buf, idx) {
  const ctx = preview.ctx;
  const now = ctx.currentTime;
  if (idx === 0) {
    const useNorm = (SOUND_MODES[polish.mode] || SOUND_MODES[DEFAULT_SOUND]).norm;
    preview.normGain = useNorm ? previewNormGain(buf) : 1;
    preview.master.gain.value = preview.muted ? 0 : preview.normGain;
  }
  const gain = ctx.createGain();
  gain.connect(preview.master);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(gain);

  let start;
  if (idx === 0) {
    start = now + 0.15;
  } else if (preview.nextCtxTime - PREVIEW_XF > now + 0.02) {
    start = preview.nextCtxTime - PREVIEW_XF; // on time: overlap into the tail fade
  } else {
    start = now + 0.05; // arrived after a stall: butt-join with a quick fade-in
  }
  const fadeIn = idx === 0 ? 0 : Math.min(PREVIEW_XF, Math.max(0.05, preview.nextCtxTime - start));
  if (fadeIn > 0) {
    gain.gain.setValueAtTime(0, start);
    gain.gain.linearRampToValueAtTime(1, start + fadeIn);
  }
  // Fade every tail so the next chunk can crossfade over it (mirrors the
  // server's linear seam). The very last tail fade is covered by handoff.
  const end = start + buf.duration;
  gain.gain.setValueAtTime(1, Math.max(start + fadeIn, end - PREVIEW_XF));
  gain.gain.linearRampToValueAtTime(0, end);
  src.start(start);
  preview.sources.push(src);
  preview.timeline.push({ ctxStart: start, takeStart: preview.nextTakeTime, dur: buf.duration });
  preview.nextCtxTime = end;
  preview.nextTakeTime += buf.duration - PREVIEW_XF;
  preview.active = true;
}

// Where preview playback currently sits, in final-take seconds — the timeline
// mirrors the server concat (each seam shortens by the crossfade).
function previewPositionSeconds() {
  if (!preview.ctx || !preview.timeline.length) return 0;
  const now = preview.ctx.currentTime;
  let pos = 0;
  for (const seg of preview.timeline) {
    if (now >= seg.ctxStart) pos = seg.takeStart + Math.min(now - seg.ctxStart, seg.dur);
  }
  return pos;
}

function updatePreviewUI() {
  if (preview.done || !preview.active) return;
  el.genPreviewRow.hidden = false;
  const n = Math.min(preview.nextIndex, preview.total || preview.nextIndex);
  if (preview.ctx && preview.ctx.state === "suspended") {
    // Browser blocked audio start without a fresh gesture.
    el.genPreviewLabel.textContent = "Preview ready — click Unmute to listen while it renders";
    return;
  }
  el.genPreviewLabel.textContent = preview.total
    ? `Live preview playing · ${n} of ${preview.total} chunks rendered`
    : "Live preview playing";
}

el.genPreviewMute.addEventListener("click", () => {
  preview.muted = !preview.muted;
  el.genPreviewMute.textContent = preview.muted ? "Unmute" : "Mute";
  if (preview.master) preview.master.gain.value = preview.muted ? 0 : preview.normGain;
  if (preview.ctx && preview.ctx.state === "suspended") preview.ctx.resume().catch(() => {});
});

/* ---------------- Generate ---------------- */
// Editing during a render is meaningless — the payload is already submitted —
// so freeze the workspace controls until it finishes or is stopped.
function setWorkspaceEnabled(enabled) {
  document.body.classList.toggle("is-generating", !enabled);
  document
    .querySelectorAll(".sidebar button, .sidebar input, .sidebar select, .canvas button, .canvas input, .canvas select, .canvas textarea")
    .forEach((node) => {
      if (node === el.generateBtn) return;  // managed by the generate handler
      node.disabled = !enabled;
    });
}

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
    if (isCustomVoice(i) && !customVoiceForSlot(i)) {
      alert(`Speaker ${i + 1}'s cloned voice is missing its clip — re-record it or pick a preset voice.`);
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
  startProgress();
  // The render is already submitted — edits here can't reach it, so lock the
  // workspace rather than let controls silently no-op, and put progress
  // front and centre.
  setWorkspaceEnabled(false);
  openGenerateStage(el.scriptTitle.textContent);
  setStatus("connecting", nextParodyLine() || "Provisioning GPU resources...");

  let customAudio;
  try {
    customAudio = await Promise.all(
      Array.from({ length: 4 }, (_, i) => {
        const voice = i < state.numSpeakers ? customVoiceForSlot(i) : null;
        return voice ? fileToBase64(voice.blob) : Promise.resolve(null);
      })
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
    speakers: state.voiceSelections.map((v) => ((v || "").startsWith(CUSTOM_PREFIX) ? null : v)),
    cfg_scale: Number(el.cfgScale.value),
    custom_audio: customAudio,
    voice_consent: el.voiceConsentCheckbox.checked,
  };

  generateAbort = new AbortController();
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: generateAbort.signal,
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
        if (evt.stage === "generating_audio") markGenerationStarted();
        if (evt.stage === "chunk_audio") {
          if (Number.isFinite(evt.chunk_index)) {
            progress.chunksDone = Math.max(progress.chunksDone, evt.chunk_index + 1);
            progress.chunksTotal = evt.chunk_total || progress.chunksTotal;
            progress.lastChunkAt = Date.now();
          }
          handleStreamChunk(evt); // async on purpose — never block the event stream
        }
        // Read the real status for progress before parody flavour replaces it.
        noteProgressFromStatus(evt.status);
        const displayLine = isDone ? evt.status : nextParodyLine() || evt.status;
        setStatus(evt.stage, displayLine);
        if (evt.log) {
          noteChunksFromLog(evt.log);
          const atBottom =
            el.logBox.scrollHeight - el.logBox.scrollTop - el.logBox.clientHeight < 40;
          const stageAtBottom =
            el.genStageLog.scrollHeight - el.genStageLog.scrollTop - el.genStageLog.clientHeight < 40;
          el.logBox.textContent = evt.log;
          el.genStageLog.textContent = evt.log;
          el.logToggleBtn.hidden = false;
          el.logBox.classList.add("visible");  // the log is worth seeing by default
          el.logToggleBtn.textContent = "Hide generation log";
          if (atBottom) el.logBox.scrollTop = el.logBox.scrollHeight;
          if (stageAtBottom) el.genStageLog.scrollTop = el.genStageLog.scrollHeight;
        }

        if (evt.stage === "complete" && evt.audio_id) {
          // Measure before the download so the figure is render time, not
          // render + warm-up + transferring hundreds of megabytes.
          const renderSeconds = generationSeconds();
          const warmupSeconds = progress.warmupSecs;
          el.generationTime.textContent = formatDuration(renderSeconds);
          if (renderSeconds > 0 && evt.audio_duration) {
            el.realtimeRow.hidden = false;
            el.realtimeFactor.textContent = `${(evt.audio_duration / renderSeconds).toFixed(2)}× realtime`;
          }
          el.warmupRow.hidden = warmupSeconds < 5;
          el.warmupTime.textContent = formatDuration(warmupSeconds);
          // Remember this run's timings so the next render's progress bar can
          // show a calibrated estimate instead of an indeterminate sweep.
          if (renderSeconds > 5 && evt.audio_duration && progress.chunks) {
            const waves = Math.max(1, progress.totalWaves || 1);
            const chunkAudio = evt.audio_duration / progress.chunks;
            const factor = renderSeconds / (waves * chunkAudio);
            if (factor > 0.2 && factor < 8) saveCalibration(state.model, { secsPerAudioSec: factor });
          }
          if (warmupSeconds > 30) saveCalibration(state.model, { warmupSecs: warmupSeconds });
          el.resultModel.textContent = state.model;
          state.resultTitle = el.scriptTitle.textContent;
          const anchors = Array.isArray(evt.chunk_starts_sec) && Array.isArray(evt.chunk_turn_counts)
            ? { starts: evt.chunk_starts_sec, counts: evt.chunk_turn_counts, turnStarts: evt.turn_starts_sec }
            : null;
          await presentTake(evt.audio_id, evt.audio_duration, turnsSnapshot, anchors);
          // Hand playback from the live preview to the real player without
          // making the listener start over.
          const handoffPos = previewPositionSeconds();
          const wasListening = preview.active && !preview.muted;
          stopPreview(0.35);
          if (wasListening && handoffPos > 1 && Number.isFinite(evt.audio_duration)) {
            const audio = el.resultAudio;
            const target = Math.max(0, Math.min(handoffPos - 0.1, evt.audio_duration - 0.5));
            const seekPlay = () => {
              try { audio.currentTime = target; } catch { /* not seekable yet */ }
              audio.play().catch(() => {});
            };
            if (audio.readyState >= 1) seekPlay();
            else audio.addEventListener("loadedmetadata", seekPlay, { once: true });
          }
        }
      }
    }
  } catch (error) {
    if (error.name === "AbortError") {
      setStatus("cancelled");
    } else {
      setStatus("error", error.message);
    }
    el.dockEmpty.hidden = false;
    checkLastTake();  // if a finished take survived the failure, offer it
  } finally {
    stopPreview(0); // no-op after a normal handoff; silences aborts and errors
    stopProgress();
    setWorkspaceEnabled(true);
    generateAbort = null;
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
    loadCustomVoices(),
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
  checkLastTake();
  window.setInterval(updateStatus, 8000);
}

init().catch((error) => {
  el.scriptGenStatus.textContent = `Failed to load app data: ${error.message}`;
});
