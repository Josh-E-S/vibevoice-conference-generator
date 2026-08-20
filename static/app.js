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
  downloading: ["Downloading", "Transferring the finished audio to your browser."],
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
  "dockEmpty", "resultBlock", "resultWaveform", "resultAudio",
  "playBtn", "playerTime", "syncedTranscript", "openPlayerBtn",
  "composerCollapsedStrip", "collapsedSummary", "composerBody",
  "playerStage", "stageTitle", "stagePlayBtn", "stageWaveform", "stageTime",
  "stageDot", "stageLine", "stageSpeaker", "stageCloseBtn", "stageDownloadBtn",
  "stageScriptToggle", "stageTranscript",
  "generationTime", "audioDuration", "resultModel", "downloadBtn",
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
  // No stop while downloading: the render is already done, only the transfer remains.
  el.stopGenBtn.hidden = !running || stage === "downloading";
}

let generateAbort = null;
el.stopGenBtn.addEventListener("click", () => {
  if (generateAbort) generateAbort.abort();
});

el.logToggleBtn.addEventListener("click", () => {
  const visible = el.logBox.classList.toggle("visible");
  el.logToggleBtn.textContent = visible ? "Hide generation log" : "View generation log";
});

/* ---------------- Take download & presentation ---------------- */
// Long takes are hundreds of MB — stream the download with progress so
// "Complete" never looks like a hang while the WAV transfers.
async function downloadTakeBlob(audioId) {
  setStatus("downloading");
  const audioRes = await fetch(`/api/audio/${audioId}`);
  if (!audioRes.ok) throw new Error("The finished audio could not be fetched from the server.");
  const totalBytes = Number(audioRes.headers.get("Content-Length")) || 0;
  const audioReader = audioRes.body.getReader();
  const parts = [];
  let received = 0;
  let lastShown = -1;
  // Stall watchdog: a big transfer through the HF proxy can hang silently;
  // without this, `await read()` would wait forever with no feedback.
  const STALL_MS = 60000;
  while (true) {
    const part = await Promise.race([
      audioReader.read(),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error(
          "The audio transfer stalled. Your take is safe on the server — use “Recover last take” to retry."
        )), STALL_MS)),
    ]);
    if (part.done) break;
    parts.push(part.value);
    received += part.value.length;
    const mb = Math.floor(received / 1048576);
    if (mb !== lastShown) {
      lastShown = mb;
      setStatus("downloading", totalBytes
        ? `Downloading your take… ${mb} / ${Math.ceil(totalBytes / 1048576)} MB`
        : `Downloading your take… ${mb} MB`);
    }
  }
  return new Blob(parts, { type: audioRes.headers.get("Content-Type") || "audio/wav" });
}

async function presentTake(blob, durationSeconds, snapshot) {
  setStatus("complete");
  const url = URL.createObjectURL(blob);
  el.resultAudio.src = url;
  el.downloadBtn.href = url;
  el.stageDownloadBtn.href = url;
  el.audioDuration.textContent = formatDuration(durationSeconds);
  el.playerTime.textContent = `0:00 / ${formatClock(durationSeconds)}`;
  el.stageTime.textContent = `0:00 / ${formatClock(durationSeconds)}`;
  setPlayIcons("►");
  buildSyncedTranscript(snapshot);
  el.dockEmpty.hidden = true;
  el.resultBlock.classList.add("visible");
  try {
    // Very long takes (hours of WAV) can exceed the browser's decode
    // memory — the placeholder waveform is fine, never fail the take.
    state.wavePeaks = await decodeWavePeaks(url, 48);
  } catch {
    state.wavePeaks = null;
  }
  renderWave(0);
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
        const blob = await downloadTakeBlob(info.audio_id);
        el.generationTime.textContent = "--";
        el.resultModel.textContent = "recovered";
        state.resultTitle = "Recovered take";
        await presentTake(blob, info.duration, []);
      } catch (error) {
        setStatus("error", error.message);
        btn.disabled = false;
      }
    });
    el.dockEmpty.append(btn);
  } catch { /* nothing to recover */ }
}

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
  const started = performance.now();
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
        const displayLine = isDone ? evt.status : nextParodyLine() || evt.status;
        setStatus(evt.stage, displayLine);
        if (evt.log) {
          el.logBox.textContent = evt.log;
          el.logToggleBtn.hidden = false;
        }

        if (evt.stage === "complete" && evt.audio_id) {
          const blob = await downloadTakeBlob(evt.audio_id);
          el.generationTime.textContent = formatDuration((performance.now() - started) / 1000);
          el.resultModel.textContent = state.model;
          state.resultTitle = el.scriptTitle.textContent;
          await presentTake(blob, evt.audio_duration, turnsSnapshot);
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
