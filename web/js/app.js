// Co-pilot web app: video or camera -> YOLO (in the browser) -> tracker -> crosswalk -> behaviour -> advisor.
// Two views: "Driver" (what a driver would get: a soft coloured glow on the hazard side and a short hint)
// and "How the system sees it" (lane, every tracked object, threats with arrows, crosswalk, risk bar).
// Analysis always takes the newest frame and never queues: it may run less often than the video, never behind it.
import { Advisor } from "./advisor.js";
import { BehaviorEngine } from "./behavior.js";
import { Crosswalk } from "./crosswalk.js";
import { Detector } from "./detector.js";
import { Sound } from "./sound.js";
import { Tracker } from "./tracker.js";

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("view"), ctx = canvas.getContext("2d");
const sound = new Sound();
let C, detector, tracker, engine, advisor, xwalk;
let state = null, running = false, analyzing = false, sourceKind = null, camT0 = 0, frameNo = 0, mode = "driver";
let hintUntil = 0;
const stats = { ms: 0, rate: 0, last: 0 };
const COLOR = ["200,200,200", "40,190,100", "255,200,0", "235,45,50"];          // clear, green, yellow, red
const SHORT = { danger: "DANGER", ttc: "BRAKE", closing: "CLOSING", cutin: "CUT-IN", crossing: "CROSSING", ped: "PEDESTRIAN",
  ped_xwalk: "PED", vru_near: "PED", drift: "DRIFT", hardbrake: "HARD BRAKING", crash: "CRASH", wrongway: "WRONG WAY",
  weave: "ERRATIC", close: "TOO CLOSE", skid: "SKID", aggressive: "AGGRESSIVE" };
const CLS = { 0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck" };

// ---------- page chrome: theme, stats, results tables ----------
$("theme").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch (e) { /* private mode */ }
});
fetch("data/results.json").then((r) => (r.ok ? r.json() : null)).then(fillResults).catch(() => {});
fetch("build.json").then((r) => r.json()).then((b) => { $("build").textContent = `build ${b.commit}`; }).catch(() => {});

function fillResults(R) {
  if (!R) return;
  const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`), sec = (v) => (v == null ? "–" : `${v.toFixed(1)} s`), num = (v) => (v == null ? "–" : v.toFixed(2));
  const set = (k, v) => document.querySelectorAll(`[data-stat="${k}"]`).forEach((e) => { e.textContent = v; });
  if (R.headline) {
    set("recall", pct(R.headline.recall_y)); set("lead", sec(R.headline.lead_y));
    set("false", R.headline.false_y_per_min == null ? "–" : `${num(R.headline.false_y_per_min)}`);
    set("speed", R.headline.ms_per_frame ? `${Math.round(R.headline.ms_per_frame)} ms` : "–");
  }
  if (R.rows) {
    $("results-table").tBodies[0].innerHTML = R.rows.map((r) => `<tr><td>${r.set}</td><td class="num">${r.n ?? "–"}</td><td class="num">${pct(r.recall_y)}</td>
      <td class="num">${sec(r.lead_y)}</td><td class="num">${pct(r.recall_r)}</td><td class="num">${num(r.false_y)}</td><td class="num">${num(r.false_r)}</td>
      <td class="num">${num(r.spoken)}</td></tr>`).join("");
  }
  if (R.detector) {
    $("detector-table").tBodies[0].innerHTML = R.detector.map((r) => `<tr><td>${r.model}</td><td class="num">${r.day_vehicle.toFixed(3)}</td>
      <td class="num">${r.night_vehicle.toFixed(3)}</td><td class="num">${r.day_person.toFixed(3)}</td><td class="num">${r.night_person.toFixed(3)}</td></tr>`).join("");
  }
}

// ---------- the demo ----------
async function init() {
  C = await (await fetch("js/config.json")).json();
  const params = new URLSearchParams(location.search);
  const phone = /(Android|iPhone|iPad)/i.test(navigator.userAgent);
  const size = Number(params.get("size") || (phone ? 320 : 416));
  const model = params.get("model") || (phone ? C.WEB_MODEL_PHONE : C.WEB_MODEL);
  detector = new Detector(C);
  xwalk = new Crosswalk();
  try {
    const backend = await detector.load(`models/${model}_${size}.onnx`, size);
    $("status").textContent = `Model ready · ${backend.toUpperCase()} · ${size} px`;
    $("backend").textContent = `${backend} · ${size} px`;
  } catch (e) {
    $("status").textContent = "This browser could not start the model: " + e.message;
    return;
  }
  document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () => startVideo(b.dataset.sample)));
  $("camera").addEventListener("click", startCamera);
  $("file").addEventListener("change", (e) => e.target.files[0] && startVideo(URL.createObjectURL(e.target.files[0])));
  $("voice").addEventListener("click", () => { sound.enabled = !sound.enabled; $("voice").textContent = `Voice: ${sound.enabled ? "on" : "off"}`; });
  for (const m of ["driver", "system"]) $(`mode-${m}`).addEventListener("click", () => setMode(m));
  $("stage").addEventListener("dblclick", () => { if (video.src) { $("intro").hidden = !$("intro").hidden; } });
  requestAnimationFrame(render);
}

function setMode(m) {
  mode = m;
  $("mode-driver").classList.toggle("on", m === "driver");
  $("mode-system").classList.toggle("on", m === "system");
}

function reset() {
  tracker = new Tracker(C);
  engine = new BehaviorEngine(C);
  advisor = new Advisor(C);
  state = null; frameNo = 0; hintUntil = 0;
  $("log").innerHTML = "";
}

async function startVideo(src) {
  sound.unlock();
  stopCamera();
  reset();
  sourceKind = "video";
  video.srcObject = null;
  video.src = src;
  video.loop = false;
  video.muted = true;
  begin();
  video.play().catch((e) => { $("status").textContent = "Press play: " + e.message; });
}

async function startCamera() {
  sound.unlock();
  reset();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    video.removeAttribute("src");
    video.srcObject = stream;
    await video.play();
    sourceKind = "camera";
    camT0 = performance.now();
    begin();
  } catch (e) {
    $("status").textContent = "Camera not available: " + e.message;
  }
}

function stopCamera() {
  if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}

function begin() {
  $("intro").hidden = true;
  running = true;
  if (!analyzing) analyze();
}

video.addEventListener("ended", () => { $("intro").hidden = false; });
const now = () => (sourceKind === "camera" ? (performance.now() - camT0) / 1000 : video.currentTime);

async function analyze() {
  analyzing = true;
  while (running) {
    if (video.readyState < 2 || video.paused || video.ended) { await new Promise((r) => setTimeout(r, 50)); continue; }
    const w = video.videoWidth, h = video.videoHeight, t = now();
    const tic = performance.now();
    const tracked = tracker.update(await detector.detect(video, w, h));
    frameNo++;
    const xw = C.CROSSWALK && frameNo % C.CROSSWALK_EVERY === 0 ? xwalk.detect(video, w, h, tracked.map((d) => d.box)) : undefined;
    const res = engine.update(t, w, h, tracked, xw);
    const adv = advisor.update(t, res, w);
    const ms = performance.now() - tic;
    stats.ms = stats.ms ? 0.9 * stats.ms + 0.1 * ms : ms;
    const dt = (performance.now() - stats.last) / 1000;
    stats.rate = stats.last ? (stats.rate ? 0.9 * stats.rate + 0.1 / dt : 1 / dt) : 0;
    stats.last = performance.now();
    state = { res, adv, w, h, t };
    if (adv.cue) {
      sound.play(adv.cue);
      addLog(t, adv.cue);
      $("hint").textContent = adv.cue.phrase;
      $("hint").className = "hint on";
      hintUntil = performance.now() + 3000;
    }
    await new Promise((r) => setTimeout(r, 0));
  }
  analyzing = false;
}

function addLog(t, cue) {
  const li = document.createElement("li");
  li.className = `lvl${cue.level}`;
  li.innerHTML = `<b>${t.toFixed(1)} s</b>${cue.phrase}`;
  if (sourceKind === "video") li.addEventListener("click", () => { video.currentTime = Math.max(0, t - 3); video.play(); });
  $("log").prepend(li);
}

function render() {
  requestAnimationFrame(render);
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw) return;
  if (canvas.width !== vw || canvas.height !== vh) { canvas.width = vw; canvas.height = vh; }
  ctx.drawImage(video, 0, 0, vw, vh);
  if (performance.now() > hintUntil) $("hint").className = "hint";
  if (!state) return;
  const { res, adv } = state, s = vw / 1280, lvl = res.level;
  if (mode === "system") drawSystem(res, vw, vh, s);
  drawGlow(adv, vw, vh);
  $("advice").textContent = res.top ? res.top.screen : "";
  $("advice").className = `advice lvl${lvl}`;
  $("hud").textContent = mode === "system"
    ? `${C.LEVEL_NAMES[lvl]} · ${stats.rate.toFixed(0)} analyses/s · ${stats.ms.toFixed(0)} ms` + (res.ttc != null && res.ttc < 6 ? ` · TTC ${res.ttc.toFixed(1)} s` : "")
    : "";
}

function drawGlow(adv, vw, vh) {
  if (!(adv.glow > 0)) return;
  const col = COLOR[adv.level], g = adv.glow * (adv.level >= 3 ? 0.85 + 0.15 * Math.sin(performance.now() / 90) : 1);
  const band = vw * 0.16;
  const grad = (x0, y0, x1, y1) => { const gr = ctx.createLinearGradient(x0, y0, x1, y1); gr.addColorStop(0, `rgba(${col},${0.62 * g})`); gr.addColorStop(1, `rgba(${col},0)`); return gr; };
  if (adv.side === "left" || adv.level >= 3) { ctx.fillStyle = grad(0, 0, band, 0); ctx.fillRect(0, 0, band, vh); }
  if (adv.side === "right" || adv.level >= 3) { ctx.fillStyle = grad(vw, 0, vw - band, 0); ctx.fillRect(vw - band, 0, band, vh); }
  if (adv.side === "ahead" || adv.level >= 3) { ctx.fillStyle = grad(0, 0, 0, vh * 0.22); ctx.fillRect(0, 0, vw, vh * 0.22); }
}

function drawSystem(res, vw, vh, s) {
  const lvl = res.level;
  // own lane (trapezoid from the calibrated horizon to the hood)
  const poly = engine.corridor(vw, vh);
  ctx.beginPath(); poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath();
  ctx.fillStyle = `rgba(${COLOR[lvl]},${lvl ? 0.22 : 0.1})`; ctx.fill();
  ctx.strokeStyle = `rgba(${COLOR[lvl]},.85)`; ctx.lineWidth = 2 * s; ctx.stroke();
  // crosswalk
  if (res.crosswalk) {
    const [x1, y1, x2, y2] = res.crosswalk;
    ctx.setLineDash([10 * s, 8 * s]); ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = 2 * s;
    ctx.strokeRect(x1 * vw, y1 * vh, (x2 - x1) * vw, (y2 - y1) * vh); ctx.setLineDash([]);
    label("CROSSWALK", x1 * vw, y1 * vh - 6 * s, "255,255,255", s);
  }
  // every tracked object
  const threatIds = new Set(res.threats.map((th) => th.det.tid));
  for (const [d, status] of res.tracks) {
    if (threatIds.has(d.tid)) continue;
    const [x1, y1, x2, y2] = d.box, lead = res.lead && res.lead.tid === d.tid;
    ctx.strokeStyle = lead ? "rgba(255,255,255,.95)" : "rgba(255,255,255,.45)"; ctx.lineWidth = (lead ? 2.5 : 1.5) * s;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.font = `500 ${12 * s}px "Geist Mono", ui-monospace, monospace`; ctx.fillStyle = "rgba(255,255,255,.85)";
    ctx.fillText(`${CLS[d.cls] || ""}${lead && res.distance ? ` ${res.distance.toFixed(0)} m` : ""} ${status === "hood" ? "" : status}`, x1, y2 + 14 * s);
  }
  // threats: most severe per object, tag, motion arrow
  const worst = new Map();
  for (const th of res.threats) if (!worst.has(th.det.tid) || th.level > worst.get(th.det.tid).level) worst.set(th.det.tid, th);
  for (const th of worst.values()) {
    const [x1, y1, x2, y2] = th.det.box, col = COLOR[th.level];
    ctx.strokeStyle = `rgba(${col},.95)`; ctx.lineWidth = (th.level >= 3 ? 4 : 3) * s;
    roundRect(x1, y1, x2 - x1, y2 - y1, 6 * s);
    label(SHORT[th.kind] + (th.count ? ` ×${th.count}` : "") + (th.dist != null && th.kind !== "crash" ? ` ${th.dist.toFixed(0)} m` : ""), x1, Math.max(y1 - 6 * s, 16 * s), col, s);
    if (th.path) {
      const p0 = [(x1 + x2) / 2, y2], p1 = [th.path[0] * vw, th.path[1] * vh];
      ctx.beginPath(); ctx.moveTo(...p0); ctx.lineTo(...p1); ctx.stroke();
    }
  }
  // risk bar
  const bx = vw - 46 * s, by = 50 * s, bh = 160 * s;
  ctx.fillStyle = "rgba(40,40,40,.75)"; ctx.fillRect(bx, by, 14 * s, bh);
  ctx.fillStyle = `rgb(${COLOR[Math.max(1, lvl)]})`; const f = Math.min(Math.max(res.risk, 0), 1) * bh; ctx.fillRect(bx, by + bh - f, 14 * s, f);
}

function label(text, x, y, col, s) {
  ctx.font = `600 ${14 * s}px "Geist", system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 10 * s, h = 19 * s;
  ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(x, y - h + 4 * s, w, h);
  ctx.fillStyle = `rgb(${col})`; ctx.fillText(text, x + 5 * s, y);
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); ctx.stroke();
}

init();
