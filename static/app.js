/* Smart Security Camera - browser client.
 *
 * Analysis runs as a background job that this page polls; the live camera
 * reports status separately from its MJPEG preview. Every state the server can
 * be in maps onto one of four signal states - idle, busy, ok, alert - which
 * drive the pill in the app bar, the status strip and the colours, so the page
 * reads the same way as the overlay burnt into the footage.
 */

const $ = (id) => document.getElementById(id);

const els = {
  // console
  form: $("videoForm"),
  dropZone: $("dropZone"),
  videoFile: $("videoFile"),
  fileName: $("fileName"),
  submitBtn: $("submitBtn"),
  cancelBtn: $("cancelBtn"),
  classifierSelect: $("classifierSelect"),
  classesField: $("classesField"),
  classesInput: $("classesInput"),

  // zones
  zoneEditor: $("zoneEditor"),
  zoneImage: $("zoneImage"),
  zoneCanvas: $("zoneCanvas"),
  zonePreviewStatus: $("zonePreviewStatus"),
  roiInput: $("roiInput"),
  ignoreInput: $("ignoreInput"),

  // monitor
  statePill: $("statePill"),
  stateText: $("stateText"),
  statusStrip: $("statusStrip"),
  statusMessage: $("statusMessage"),
  monitorTitle: $("monitorTitle"),
  monitorSub: $("monitorSub"),
  progressWrap: $("progressWrap"),
  progressBar: $("progressBar"),
  progressLabel: $("progressLabel"),
  resultsList: $("resultsList"),
  timeline: $("timeline"),
  timelineTrack: $("timelineTrack"),
  timelineLabel: $("timelineLabel"),
  timelineEnd: $("timelineEnd"),

  // live
  liveDevice: $("liveDevice"),
  liveSensitivity: $("liveSensitivity"),
  liveClassifier: $("liveClassifier"),
  liveTracking: $("liveTracking"),
  liveRecording: $("liveRecording"),
  startLiveBtn: $("startLiveBtn"),
  stopLiveBtn: $("stopLiveBtn"),
  rescanBtn: $("rescanBtn"),
  snapshotBtn: $("snapshotBtn"),
  liveStream: $("liveStream"),
  stageWrap: $("stageWrap"),
  stageBadge: $("stageBadge"),
  stageBadgeText: $("stageBadgeText"),
  liveInfo: $("liveInfo"),
  statResolution: $("statResolution"),
  statFps: $("statFps"),
  statFrames: $("statFrames"),
  statIncidents: $("statIncidents"),
  statIncidentsTile: $("statIncidentsTile"),
  statTracked: $("statTracked"),
  statUptime: $("statUptime"),
  liveZonesBtn: $("liveZonesBtn"),
  liveZonesDone: $("liveZonesDone"),
  liveZoneEditor: $("liveZoneEditor"),
  liveZoneImage: $("liveZoneImage"),
  liveZoneCanvas: $("liveZoneCanvas"),
  liveZoneHint: $("liveZoneHint"),

  // incidents
  incidentsGrid: $("incidentsGrid"),
  incidentsStatus: $("incidentsStatus"),
  incidentFilter: $("incidentFilter"),
  refreshIncidentsBtn: $("refreshIncidentsBtn"),
  incidentCount: $("incidentCount"),

  // chrome
  themeToggle: $("themeToggle"),
  toastHost: $("toastHost"),
  lightbox: $("lightbox"),
  lightboxVideo: $("lightboxVideo"),
  lightboxTitle: $("lightboxTitle"),
  lightboxMeta: $("lightboxMeta"),
  lightboxClose: $("lightboxClose"),
  lightboxPrev: $("lightboxPrev"),
  lightboxNext: $("lightboxNext"),
  lightboxDownload: $("lightboxDownload"),
};

let currentJobId = null;
let jobPoller = null;
let livePoller = null;
let incidentsLoaded = false;
let uploadZones = null;
let liveZoneEditor = null;
let liveZoneState = { roi: null, ignore: [] };
let lastResults = [];
let lightboxIndex = -1;

/* ---------------------------------------------------------------- helpers */

// Server data is interpolated into HTML below, so it is escaped first.
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));
}

function formatSeconds(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return "—";
  if (value < 60) return `${value.toFixed(1)}s`;
  const mins = Math.floor(value / 60);
  return `${mins}m ${String(Math.round(value % 60)).padStart(2, "0")}s`;
}

function formatClock(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const m = Math.floor(value / 60);
  const s = Math.floor(value % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatBytes(bytes) {
  const value = Number(bytes);
  if (!value) return "—";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function formatDateTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? escapeHtml(value) : date.toLocaleString();
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  let data = {};
  try {
    data = await response.json();
  } catch (error) {
    throw new Error(`The server returned an unreadable response (HTTP ${response.status}).`);
  }
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `Request failed (HTTP ${response.status}).`);
  }
  return data;
}

/* ------------------------------------------------------------ signal state */

/** One place decides how the page looks in each state. */
function setState(state, pillText, message) {
  els.statePill.dataset.state = state;
  els.stateText.textContent = pillText;
  els.statusStrip.dataset.state = state;
  if (message !== undefined) els.statusMessage.innerHTML = message;
}

function setMonitor(title, sub) {
  els.monitorTitle.textContent = title;
  els.monitorSub.textContent = sub;
}

function setProgress(percent, label) {
  els.progressWrap.hidden = percent === null;
  if (percent === null) return;
  const clamped = Math.max(0, Math.min(100, percent));
  els.progressBar.style.width = `${clamped}%`;
  els.progressLabel.textContent = label || `${clamped}%`;
}

/* -------------------------------------------------------------- toasts */

function toast(message, kind = "info", timeout = 4200) {
  const node = document.createElement("div");
  node.className = `toast ${kind}`;
  node.innerHTML = `<span class="bar"></span><span>${escapeHtml(message)}</span>`;
  els.toastHost.append(node);

  const remove = () => {
    node.classList.add("leaving");
    setTimeout(() => node.remove(), 220);
  };
  node.addEventListener("click", remove);
  setTimeout(remove, timeout);
}

/* --------------------------------------------------------------- theme */

const THEME_KEY = "ssc-theme";

function applyTheme(theme) {
  if (theme === "light") document.documentElement.setAttribute("data-theme", "light");
  else document.documentElement.removeAttribute("data-theme");
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch (error) {
    /* private mode: the choice simply does not persist */
  }
}

els.themeToggle.addEventListener("click", () => {
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  applyTheme(isLight ? "dark" : "light");
});

try {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved) applyTheme(saved);
} catch (error) {
  /* ignore */
}

/* ------------------------------------------------------------------ tabs */

const TABS = ["upload", "live", "incidents"];

/** Show one tab. The name is mirrored into the URL, so a reload or a shared
 *  link comes back to the same place. */
function activateTab(name) {
  const tab = TABS.includes(name) ? name : "upload";
  document.body.dataset.tab = tab;
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    const isActive = btn.getAttribute("data-tab") === tab;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-selected", String(isActive));
  });
  document.querySelectorAll(".tab-content").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${tab}-tab`);
  });
  if (tab === "incidents" && !incidentsLoaded) loadIncidents();
  if (tab === "live") loadDevices(false);
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const tab = btn.getAttribute("data-tab");
    activateTab(tab);
    history.replaceState(null, "", tab === "upload" ? "#" : `#${tab}`);
  });
});

window.addEventListener("hashchange", () => activateTab(location.hash.slice(1)));

/* ------------------------------------------------------------ file picker */

els.dropZone.addEventListener("click", () => els.videoFile.click());
els.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    els.videoFile.click();
  }
});
els.videoFile.addEventListener("change", () => {
  const file = els.videoFile.files[0];
  els.dropZone.classList.toggle("has-file", Boolean(file));
  els.fileName.textContent = file ? `${file.name} · ${formatBytes(file.size)}` : "No file selected";
  if (file) loadPreviewFrame(file);
});

["dragenter", "dragover"].forEach((name) =>
  els.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    els.dropZone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((name) =>
  els.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    els.dropZone.classList.remove("dragover");
  })
);
els.dropZone.addEventListener("drop", (event) => {
  const files = event.dataTransfer?.files;
  if (!files?.length) return;
  const transfer = new DataTransfer();
  transfer.items.add(files[0]);
  els.videoFile.files = transfer.files;
  els.videoFile.dispatchEvent(new Event("change"));
});

/* -------------------------------------------------------------- analysis */

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  syncZonesToInputs();
  const formData = new FormData(els.form);
  const hasFile = els.videoFile.files.length > 0;
  const url = (formData.get("video_url") || "").toString().trim();

  if (!hasFile && !url) {
    setState("error", "Missing input", "Choose a video file or paste a direct video URL first.");
    toast("Choose a video file or paste a URL first", "error");
    return;
  }

  els.submitBtn.disabled = true;
  els.cancelBtn.hidden = false;
  els.resultsList.innerHTML = "";
  els.timeline.hidden = true;
  lastResults = [];
  setState("busy", "Queued", "Uploading and queuing the analysis…");
  setMonitor("Analysis", "starting");
  setProgress(0, "queued");

  try {
    const data = await requestJson("/api/jobs", { method: "POST", body: formData });
    currentJobId = data.job_id;
    // Keep the id in the URL so a reload (or a shared link) comes back to
    // this run's progress and results instead of an empty page.
    history.replaceState(null, "", `?job=${data.job_id}${location.hash}`);
    setMonitor("Analysis", data.source_name);
    setState("busy", "Analysing", `Analysing <strong>${escapeHtml(data.source_name)}</strong>…`);
    pollJob();
  } catch (error) {
    setState("error", "Error", escapeHtml(error.message));
    setProgress(null);
    toast(error.message, "error");
    els.submitBtn.disabled = false;
    els.cancelBtn.hidden = true;
  }
});

els.cancelBtn.addEventListener("click", async () => {
  if (!currentJobId) return;
  els.cancelBtn.disabled = true;
  try {
    await requestJson(`/api/jobs/${currentJobId}/cancel`, { method: "POST" });
    toast("Cancelling the analysis…", "info");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    els.cancelBtn.disabled = false;
  }
});

function pollJob() {
  clearInterval(jobPoller);
  // Fetch once immediately, so a resumed run shows its state at once rather
  // than sitting on "restoring" for the length of the polling interval.
  readJob();
  jobPoller = setInterval(readJob, 600);
}

async function readJob() {
    if (!currentJobId) return;
    try {
      const job = await requestJson(`/api/jobs/${currentJobId}`);
      const frames = job.frames_total
        ? `${job.frames_done}/${job.frames_total} frames`
        : `${job.frames_done} frames`;

      if (job.status === "running" || job.status === "queued") {
        setState("busy", `${job.percent}%`,
          `Analysing <strong>${escapeHtml(job.source_name)}</strong> — ${job.incidents_detected} incident(s) so far`);
        setProgress(job.percent, `${job.percent}% · ${frames}`);
        return;
      }

      clearInterval(jobPoller);
      els.submitBtn.disabled = false;
      els.cancelBtn.hidden = true;

      if (job.status === "failed") {
        setState("error", "Failed", escapeHtml(job.error || "Analysis failed."));
        setProgress(null);
        toast(job.error || "Analysis failed", "error", 7000);
        return;
      }

      if (job.status === "cancelled") {
        setState("idle", "Cancelled", `Cancelled after ${frames}.`);
        setProgress(null);
        toast("Analysis cancelled", "info");
      } else {
        const count = job.incidents_detected;
        setState(count ? "alert" : "ok", count ? `${count} incident(s)` : "Clear",
          `<strong>${escapeHtml(job.source_name)}</strong> · ${escapeHtml(job.sensitivity)} sensitivity · ` +
          `<strong>${count}</strong> incident(s) in ${job.elapsed_s}s`);
        setProgress(100, `${frames} in ${job.elapsed_s}s`);
        setMonitor("Analysis complete", job.source_name);
        toast(count ? `${count} incident(s) recorded` : "No unauthorised motion found",
              count ? "ok" : "info");
      }
      renderResults(job);
      incidentsLoaded = false;
      refreshIncidentCount();
    } catch (error) {
      clearInterval(jobPoller);
      setState("error", "Error", escapeHtml(error.message));
      toast(error.message, "error");
      els.submitBtn.disabled = false;
      els.cancelBtn.hidden = true;
    }
}

function renderResults(job) {
  const clips = job.incidents || [];
  lastResults = clips;

  if (!clips.length) {
    els.timeline.hidden = true;
    els.resultsList.innerHTML = `
      <div class="empty-state">
        <svg><use href="#i-empty" /></svg>
        <span>No unauthorised motion was detected.</span>
        <span class="hint">Raise the sensitivity, shorten "motion needed", or check
        that the monitored zone covers the area you care about.</span>
      </div>`;
    return;
  }

  renderTimeline(job, clips);

  els.resultsList.innerHTML = clips
    .map(
      (clip, index) => `
        <article class="result-card">
          <div class="thumb" data-open="${index}" role="button" tabindex="0"
               aria-label="Play incident ${clip.index}">
            ${clip.thumbnail_url
              ? `<img src="${escapeHtml(clip.thumbnail_url)}" alt="" loading="lazy" />`
              : `<div class="thumb-placeholder">no preview</div>`}
            <span class="play"><span>▶</span></span>
            <span class="dur">${formatSeconds(clip.duration_s)}</span>
          </div>
          <div class="result-body">
            <h3>Incident ${clip.index}</h3>
            <div class="result-meta">
              <span>starts <span class="num">${formatClock(clip.start_time_s)}</span></span>
              <span><span class="num">${clip.max_objects}</span> object(s)</span>
            </div>
            <div class="result-actions">
              <button class="link-btn" type="button" data-open="${index}">Play</button>
              <a class="link-btn" href="${escapeHtml(clip.media_url)}" download>Download</a>
            </div>
          </div>
        </article>`
    )
    .join("");

  els.resultsList.querySelectorAll("[data-open]").forEach((node) => {
    const open = () => openLightbox(Number(node.dataset.open));
    node.addEventListener("click", open);
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open();
      }
    });
  });
}

/** Where the incidents fall inside the recording, at a glance. */
function renderTimeline(job, clips) {
  const duration = Number(job.duration_s) ||
    clips.reduce((max, clip) => Math.max(max, Number(clip.end_time_s) || 0), 0);
  if (!duration) {
    els.timeline.hidden = true;
    return;
  }

  const covered = clips.reduce((sum, clip) => sum + (Number(clip.duration_s) || 0), 0);
  els.timelineTrack.innerHTML = clips
    .map((clip, index) => {
      const left = (Number(clip.start_time_s) / duration) * 100;
      const width = Math.max(0.6, (Number(clip.duration_s) / duration) * 100);
      return `<div class="timeline-seg" data-open="${index}" title="Incident ${clip.index} · ${formatClock(clip.start_time_s)}"
                   style="left:${left.toFixed(2)}%;width:${width.toFixed(2)}%"></div>`;
    })
    .join("");

  els.timelineTrack.querySelectorAll("[data-open]").forEach((node) => {
    node.addEventListener("click", () => openLightbox(Number(node.dataset.open)));
  });

  els.timelineLabel.textContent = `${Math.round((covered / duration) * 100)}% of footage flagged`;
  els.timelineEnd.textContent = formatClock(duration);
  els.timeline.hidden = false;
}

/* ------------------------------------------------------------- lightbox */

function openLightbox(index) {
  const clip = lastResults[index];
  if (!clip) return;
  lightboxIndex = index;

  els.lightboxTitle.textContent = clip.filename || `Incident ${clip.index}`;
  els.lightboxVideo.src = clip.media_url;
  els.lightboxDownload.href = clip.media_url;
  els.lightboxMeta.textContent =
    `starts ${formatClock(clip.start_time_s)} · ${formatSeconds(clip.duration_s)} · ${clip.max_objects} object(s)`;
  els.lightboxPrev.disabled = index <= 0;
  els.lightboxNext.disabled = index >= lastResults.length - 1;
  els.lightbox.hidden = false;
  els.lightboxVideo.play().catch(() => {});
}

function closeLightbox() {
  els.lightbox.hidden = true;
  els.lightboxVideo.pause();
  els.lightboxVideo.removeAttribute("src");
  els.lightboxVideo.load();
}

els.lightboxClose.addEventListener("click", closeLightbox);
els.lightbox.addEventListener("click", (event) => {
  if (event.target === els.lightbox) closeLightbox();
});
els.lightboxPrev.addEventListener("click", () => openLightbox(lightboxIndex - 1));
els.lightboxNext.addEventListener("click", () => openLightbox(lightboxIndex + 1));

document.addEventListener("keydown", (event) => {
  if (els.lightbox.hidden) return;
  if (event.key === "Escape") closeLightbox();
  if (event.key === "ArrowLeft" && lightboxIndex > 0) openLightbox(lightboxIndex - 1);
  if (event.key === "ArrowRight" && lightboxIndex < lastResults.length - 1) {
    openLightbox(lightboxIndex + 1);
  }
});

/* ------------------------------------------------------------------ live */

async function loadDevices(refresh) {
  try {
    const data = await requestJson(`/api/live/devices${refresh ? "?refresh=1" : ""}`);
    const devices = data.devices || [];
    if (!devices.length) {
      els.liveDevice.innerHTML = `<option value="0">No camera detected</option>`;
      if (refresh) toast("No camera detected — check it is connected and not in use", "error");
      return;
    }
    els.liveDevice.innerHTML = devices
      .map(
        (device) =>
          `<option value="${device.index}">Camera ${device.index} · ${device.width}×${device.height}</option>`
      )
      .join("");
    if (refresh) toast(`${devices.length} camera(s) found`, "ok");
  } catch (error) {
    els.liveDevice.innerHTML = `<option value="0">Camera 0</option>`;
    toast(error.message, "error");
  }
}

els.rescanBtn.addEventListener("click", () => {
  els.rescanBtn.disabled = true;
  toast("Scanning for cameras…", "info", 2000);
  loadDevices(true).finally(() => (els.rescanBtn.disabled = false));
});

els.startLiveBtn.addEventListener("click", async () => {
  els.startLiveBtn.disabled = true;
  setState("busy", "Starting", "Opening the camera…");
  try {
    const data = await requestJson("/api/live/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        device: Number(els.liveDevice.value || 0),
        sensitivity: els.liveSensitivity.value,
        tracking: els.liveTracking.checked,
        recording: els.liveRecording.checked,
        detection: els.liveClassifier.value,
        roi: liveZoneState.roi,
        ignore_zones: liveZoneState.ignore,
      }),
    });

    els.liveStream.src = `/api/live/feed?t=${Date.now()}`;
    els.stageWrap.hidden = false;
    els.liveInfo.hidden = false;
    els.stageBadge.hidden = false;
    els.snapshotBtn.hidden = false;
    els.stopLiveBtn.disabled = false;
    els.statResolution.textContent = data.resolution;
    setMonitor("Live camera", `device ${data.device} · ${data.backend}`);
    setState("ok", "Live", "Camera running — calibrating the background…");
    toast(`Camera ${data.device} started at ${data.resolution}`, "ok");
    pollLive();
  } catch (error) {
    setState("error", "Camera error", escapeHtml(error.message));
    toast(error.message, "error", 8000);
    els.startLiveBtn.disabled = false;
  }
});

els.stopLiveBtn.addEventListener("click", async () => {
  els.stopLiveBtn.disabled = true;
  clearInterval(livePoller);
  try {
    const data = await requestJson("/api/live/stop", { method: "POST" });
    const stats = data.stats || {};
    setState("idle", "Idle", `Camera stopped — ${stats.incidents || 0} incident(s) recorded.`);
    toast(`Camera stopped · ${stats.incidents || 0} incident(s) recorded`, "info");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    els.liveStream.src = "";
    els.stageWrap.hidden = true;
    els.stageBadge.hidden = true;
    els.snapshotBtn.hidden = true;
    els.startLiveBtn.disabled = false;
    setMonitor("Monitor", "Nothing running");
    incidentsLoaded = false;
    refreshIncidentCount();
    loadIncidents();
  }
});

function pollLive() {
  clearInterval(livePoller);
  // Fetch once straight away: waiting a full second before the first update
  // leaves the tiles showing placeholders after the camera is already live.
  updateLiveStats();
  livePoller = setInterval(updateLiveStats, 1000);
}

async function updateLiveStats() {
    try {
      const data = await requestJson("/api/live/status");
      const stats = data.stats || {};

      if (!stats.active) {
        clearInterval(livePoller);
        setState("idle", "Idle", stats.error ? `Stopped: ${escapeHtml(stats.error)}` : "Camera stopped.");
        els.startLiveBtn.disabled = false;
        els.stopLiveBtn.disabled = true;
        els.stageWrap.hidden = true;
        els.stageBadge.hidden = true;
        if (stats.error) toast(stats.error, "error", 8000);
        return;
      }

      els.statFps.textContent = stats.measured_fps;
      els.statFrames.textContent = stats.frames;
      els.statIncidents.textContent = stats.incidents;
      els.statTracked.textContent = stats.tracked_objects;
      els.statUptime.textContent = formatSeconds(stats.uptime_s);
      els.statResolution.textContent = `${stats.width}×${stats.height}`;
      els.statIncidentsTile.classList.toggle("is-alert", stats.incidents > 0);

      if (stats.warming_up) {
        setState("busy", "Calibrating", "Learning the background before alarms can be raised…");
        els.stageBadgeText.textContent = "CALIBRATING";
      } else if (stats.recording) {
        setState("alert", "Recording", `Recording an incident — ${stats.tracked_objects} object(s) tracked.`);
        els.stageBadgeText.textContent = "RECORDING";
      } else if (stats.alert) {
        setState("alert", "Motion", `Motion detected — ${stats.tracked_objects} object(s) tracked.`);
        els.stageBadgeText.textContent = "MOTION";
      } else {
        setState("ok", "Live", `Monitoring at ${stats.measured_fps} fps — no incident.`);
        els.stageBadgeText.textContent = "LIVE";
      }
    } catch (error) {
      clearInterval(livePoller);
      setState("error", "Error", escapeHtml(error.message));
    }
}

/* ------------------------------------------------------------- incidents */

function incidentSkeletons(count = 6) {
  return Array.from({ length: count }, () => `<div class="skeleton skeleton-card"></div>`).join("");
}

async function loadIncidents() {
  incidentsLoaded = true;
  els.incidentsStatus.textContent = "Loading…";
  els.incidentsGrid.innerHTML = incidentSkeletons();

  try {
    const source = els.incidentFilter.value;
    const data = await requestJson(`/api/incidents${source ? `?source=${source}` : ""}`);
    const incidents = data.incidents || [];
    updateIncidentCount(data.stats?.total ?? incidents.length);

    if (!incidents.length) {
      els.incidentsGrid.innerHTML = `
        <div class="empty-state" style="grid-column:1/-1">
          <svg><use href="#i-empty" /></svg>
          <span>No incidents recorded yet.</span>
          <span class="hint">Analyse a recording or run the live camera; clips appear here automatically.</span>
        </div>`;
      els.incidentsStatus.textContent = "0 incidents";
      return;
    }

    els.incidentsGrid.innerHTML = incidents
      .map(
        (incident) => `
          <article class="incident-card">
            <a class="thumb" href="${escapeHtml(incident.view_url)}" target="_blank" rel="noreferrer"
               aria-label="Open ${escapeHtml(incident.filename)}">
              ${incident.thumbnail_url
                ? `<img src="${escapeHtml(incident.thumbnail_url)}" alt="" loading="lazy" />`
                : `<div class="thumb-placeholder">no preview</div>`}
              <span class="play"><span>▶</span></span>
              ${incident.duration_s ? `<span class="dur">${formatSeconds(incident.duration_s)}</span>` : ""}
            </a>
            <div class="incident-body">
              <h3>${escapeHtml(incident.filename)}</h3>
              <div class="incident-meta">
                <span class="tag tag-${escapeHtml(incident.source)}">${escapeHtml(incident.source)}</span>
                <span class="num">${formatBytes(incident.size_bytes)}</span>
              </div>
              <div class="incident-meta dim">
                <span>${escapeHtml(incident.source_name || "—")}</span>
              </div>
              <div class="incident-meta dim">
                <span>${formatDateTime(incident.created_at)}</span>
              </div>
              <div class="incident-actions">
                <a class="link-btn" href="${escapeHtml(incident.view_url)}" target="_blank" rel="noreferrer">Play</a>
                <a class="link-btn" href="${escapeHtml(incident.media_url)}" download>Save</a>
                <button class="link-btn danger" type="button" data-delete="${incident.id}">Delete</button>
              </div>
            </div>
          </article>`
      )
      .join("");

    els.incidentsGrid.querySelectorAll("[data-delete]").forEach((button) => {
      button.addEventListener("click", async () => {
        if (!window.confirm("Delete this incident and its clip permanently?")) return;
        button.disabled = true;
        try {
          await requestJson(`/api/incidents/${button.dataset.delete}`, { method: "DELETE" });
          toast("Incident deleted", "ok");
          loadIncidents();
        } catch (error) {
          toast(error.message, "error");
          button.disabled = false;
        }
      });
    });

    const stats = data.stats || {};
    els.incidentsStatus.textContent =
      `${incidents.length} shown · ${stats.total || 0} stored · ${formatBytes(stats.size_bytes)}`;
  } catch (error) {
    els.incidentsGrid.innerHTML =
      `<div class="empty-state" style="grid-column:1/-1">${escapeHtml(error.message)}</div>`;
    els.incidentsStatus.textContent = "Could not load incidents";
  }
}

function updateIncidentCount(total) {
  els.incidentCount.hidden = !total;
  els.incidentCount.textContent = total;
}

async function refreshIncidentCount() {
  try {
    const data = await requestJson("/api/health");
    updateIncidentCount(data.incidents?.total ?? 0);
  } catch (error) {
    /* the health check reports connection problems */
  }
}

els.refreshIncidentsBtn.addEventListener("click", loadIncidents);
els.incidentFilter.addEventListener("change", loadIncidents);

/* ------------------------------------------------------------------ zones */

/* The editor draws on a still frame. For a local file the browser decodes the
 * frame itself, so nothing is uploaded until the analysis actually starts; the
 * server endpoint is the fallback for containers the browser cannot open. */

function ensureUploadEditor() {
  if (uploadZones) return uploadZones;
  uploadZones = new ZoneEditor({
    root: els.zoneEditor,
    image: els.zoneImage,
    canvas: els.zoneCanvas,
    onChange: syncZonesToInputs,
  });
  return uploadZones;
}

function syncZonesToInputs() {
  if (!uploadZones) return;
  const { roi, ignore } = uploadZones.getZones();
  els.roiInput.value = zoneUtils.formatZone(roi);
  els.ignoreInput.value = ignore.map(zoneUtils.formatZone).join("\n");
}

function syncInputsToZones() {
  if (!uploadZones) return;
  const roi = zoneUtils.parseZone(els.roiInput.value);
  const ignore = els.ignoreInput.value.split("\n").map(zoneUtils.parseZone).filter(Boolean);
  uploadZones.setZones(roi, ignore);
}

els.roiInput.addEventListener("change", syncInputsToZones);
els.ignoreInput.addEventListener("change", syncInputsToZones);

async function loadPreviewFrame(file) {
  els.zonePreviewStatus.textContent = "Reading a frame…";
  const editor = ensureUploadEditor();

  try {
    const dataUrl = await zoneUtils.grabFrameFromFile(file);
    await editor.setImage(dataUrl);
    els.zoneEditor.hidden = false;
    els.zonePreviewStatus.textContent = "Drag on the frame to draw";
    return;
  } catch (error) {
    els.zonePreviewStatus.textContent = "Asking the server for a frame…";
  }

  try {
    const body = new FormData();
    body.append("video_file", file);
    const response = await fetch("/api/preview", { method: "POST", body });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `Preview failed (HTTP ${response.status})`);
    }
    const blob = await response.blob();
    await editor.setImage(URL.createObjectURL(blob));
    els.zoneEditor.hidden = false;
    els.zonePreviewStatus.textContent = "Drag on the frame to draw";
  } catch (error) {
    els.zoneEditor.hidden = true;
    els.zonePreviewStatus.textContent = "Frame unavailable — type coordinates instead";
    toast(error.message, "error");
  }
}

/* A pasted URL can also be previewed, but only on request: fetching it costs a
 * download, so it waits for the field to lose focus. */
const urlField = els.form.querySelector('input[name="video_url"]');
urlField.addEventListener("change", async () => {
  const url = urlField.value.trim();
  if (!url || els.videoFile.files.length) return;

  els.zonePreviewStatus.textContent = "Fetching a frame from that URL…";
  try {
    const body = new FormData();
    body.append("video_url", url);
    const response = await fetch("/api/preview", { method: "POST", body });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `Preview failed (HTTP ${response.status})`);
    }
    const blob = await response.blob();
    await ensureUploadEditor().setImage(URL.createObjectURL(blob));
    els.zoneEditor.hidden = false;
    els.zonePreviewStatus.textContent = "Drag on the frame to draw";
  } catch (error) {
    els.zoneEditor.hidden = true;
    els.zonePreviewStatus.textContent = "Frame unavailable — type coordinates instead";
    toast(error.message, "error");
  }
});

/* ------------------------------------------------------- live camera zones */

els.liveZonesBtn.addEventListener("click", async () => {
  if (!els.liveZoneEditor.hidden) {
    els.liveZoneEditor.hidden = true;
    return;
  }

  els.liveZonesBtn.disabled = true;
  try {
    const device = Number(els.liveDevice.value || 0);
    const response = await fetch(`/api/live/preview?device=${device}&t=${Date.now()}`);
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `Could not get a frame (HTTP ${response.status})`);
    }
    const blob = await response.blob();

    if (!liveZoneEditor) {
      liveZoneEditor = new ZoneEditor({
        root: els.liveZoneEditor,
        image: els.liveZoneImage,
        canvas: els.liveZoneCanvas,
        onChange: (zones) => {
          liveZoneState = zones;
          describeLiveZones();
        },
      });
    }
    await liveZoneEditor.setImage(URL.createObjectURL(blob));
    liveZoneEditor.setZones(liveZoneState.roi, liveZoneState.ignore);
    els.liveZoneEditor.hidden = false;
    describeLiveZones();
  } catch (error) {
    toast(error.message, "error", 7000);
  } finally {
    els.liveZonesBtn.disabled = false;
  }
});

els.liveZonesDone.addEventListener("click", () => {
  els.liveZoneEditor.hidden = true;
  const active = !els.stopLiveBtn.disabled;
  toast(active ? "Zones saved — restart the camera to apply" : "Zones saved", "ok");
});

function describeLiveZones() {
  const parts = [];
  if (liveZoneState.roi) parts.push("1 monitored zone");
  if (liveZoneState.ignore.length) parts.push(`${liveZoneState.ignore.length} ignore zone(s)`);
  els.liveZoneHint.textContent = parts.length
    ? `${parts.join(", ")} — applied when the camera starts.`
    : "Drag to draw. Zones apply the next time the camera starts.";
}

/* ---------------------------------------------------------------- health */

async function checkHealth() {
  try {
    const data = await requestJson("/api/health");
    updateIncidentCount(data.incidents?.total ?? 0);
    if (!data.yolo_available) {
      // Say so on the options that need it rather than letting them fail later.
      document.querySelectorAll('option[value^="yolo"]').forEach((option) => {
        option.textContent = `${option.textContent} — needs ultralytics`;
        option.disabled = true;
      });
    }
    if (els.statePill.dataset.state === "idle") {
      setState("idle", "Idle",
        `Ready · <strong>${data.incidents?.total ?? 0}</strong> incident(s) stored. ` +
        "Choose a recording or start the live camera.");
    }
  } catch (error) {
    setState("error", "Offline", "The server is not responding. Is <code>app.py</code> still running?");
  }
}

function updateClassesVisibility() {
  // The class list only applies to the YOLO modes.
  els.classesField.hidden = !els.classifierSelect.value.startsWith("yolo");
}

els.classifierSelect.addEventListener("change", updateClassesVisibility);
updateClassesVisibility();

els.resultsList.innerHTML = `
  <div class="empty-state">
    <svg><use href="#i-empty" /></svg>
    <span>No analysis has run yet.</span>
    <span class="hint">Drop a recording on the left, or open the Live tab to monitor a camera.</span>
  </div>`;

/** Adopt a live session that is already running, so opening the dashboard on
 *  a second screen shows the feed instead of an idle Start button. */
async function attachToRunningSession() {
  try {
    const data = await requestJson("/api/live/status");
    const stats = data.stats || {};
    if (!stats.active) return;

    els.liveStream.src = `/api/live/feed?t=${Date.now()}`;
    els.stageWrap.hidden = false;
    els.stageBadge.hidden = false;
    els.liveInfo.hidden = false;
    els.snapshotBtn.hidden = false;
    els.startLiveBtn.disabled = true;
    els.stopLiveBtn.disabled = false;
    els.statResolution.textContent = `${stats.width}×${stats.height}`;
    setMonitor("Live camera", `device ${stats.device_index} · ${stats.backend}`);
    toast("Attached to the camera session already running", "info");
    pollLive();
  } catch (error) {
    /* checkHealth reports server problems */
  }
}

/* ------------------------------------------------------------- start-up */

const params = new URLSearchParams(location.search);

// A kiosk or a shared link can pin the theme; otherwise the saved choice wins.
const forcedTheme = params.get("theme");
if (forcedTheme === "light" || forcedTheme === "dark") applyTheme(forcedTheme);

activateTab(location.hash.slice(1));
checkHealth();
loadDevices(false);
attachToRunningSession();

// Reopen a run that is still going, or one that has just finished.
const resumeJob = params.get("job");
if (resumeJob) {
  currentJobId = resumeJob;
  setState("busy", "Loading", "Restoring the last analysis…");
  pollJob();
}
