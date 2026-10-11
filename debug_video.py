"""Render a video with everything the system sees and thinks - for checking by eye and for demos.

    python debug_video.py data/videos/krakow_peds.mp4                 # full debug view
    python debug_video.py data/videos/krakow_peds.mp4 --demo          # driver view only (for a demo)
    python debug_video.py data/videos --start 60 --dur 30             # every video in a folder

Output in debug/: <video>_debug.mp4 (or _demo.mp4) with sound and <video>_frames.csv (values per analysis).
Offline, frame by frame (not real time), reproducible. Every --stride-th frame is analysed (as on
the device); the last result is held in between. The pipeline is the same as in main.py.

Debug view:
  boxes   - every track: #id, class, u (lanes from our lane centre), lateral speed, fired threats;
  lanes   - own lane filled with the level colour (by markings; without them - white trapezoid);
  threats - box in the level colour, type, distance; arrow - where the object will be in a second;
  left    - time, lead car, TTC, distance, own speed, risk, raw conditions, confirmed alerts;
  bottom  - plots for the last --window seconds: TTC, learned risk; one row per alert type:
            thin - condition met (raw), thick - alert confirmed, coloured by its level.
Sound: what the driver hears - the level chime (beeps.py) + phrase from sounds/, with the same
budget and cooldowns as the app (advisor.py). Needs ffmpeg from imageio-ffmpeg.
"""
import argparse
import csv
import glob
import os
import subprocess
import wave
from collections import deque

import cv2
import numpy as np

import config as C
import ui
from behavior import corridor_polygon
from voice import chime_file, phrase_file

VIDEO_EXT = (".mov", ".mp4", ".avi", ".mkv", ".webm", ".ogv")
CLS = {**C.VEHICLE_CLASSES, C.PERSON_CLASS: "person", C.BICYCLE_CLASS: "bicycle"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
GRAY, WHITE = (150, 150, 150), (255, 255, 255)


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def text(img, s, org, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, s, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_debug(frame, res, t, fi, k=1.0):
    """k - how much the frame is scaled relative to the source (boxes are in source pixels)."""
    H, W = frame.shape[:2]
    if not ui.draw_lanes(frame, res.lanes, res.level):
        cv2.polylines(frame, [corridor_polygon(W, H, res.view)], True, WHITE, 1, cv2.LINE_AA)
    ui.draw_crosswalk(frame, res.crosswalk)
    cv2.line(frame, (0, int(res.view[1] * H)), (W, int(res.view[1] * H)), (255, 120, 0), 1, cv2.LINE_AA)
    lead_id = res.lead.tid if res.lead else None
    threat_ids = {th.det.tid for th in res.threats}
    for d, status in res.tracks:
        x1, y1, x2, y2 = (int(v * k) for v in d.box)
        color = (0, 200, 0) if d.tid == lead_id else GRAY
        if d.tid not in threat_ids:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if d.tid == lead_id else 1)
        tag = "LEAD " if d.tid == lead_id else ""
        text(frame, f"#{d.tid} {CLS.get(d.cls, d.cls)} {tag}{status}", (x1, y2 + 14), 0.4, color)
    ui.draw_threats(frame, res, k, 0.5)

    lines = [f"t {t:7.2f}s  frame {fi}"]
    if res.lead:
        ttc = f"{res.ttc:.2f}s" if res.ttc is not None else "-"
        lines.append(f"lead #{res.lead.tid}  TTC {ttc}" + ("  ONCOMING" if res.oncoming else ""))
        if res.distance is not None:
            lines.append(f"dist ~{res.distance:.1f} m")
    else:
        lines.append("lead: none")
    lines.append(f"own speed: {res.ego_speed * 3.6:.0f} km/h" if res.ego_speed is not None else "own speed: -")
    lines.append("lanes: " + ("yes" if res.lanes is not None else "no (trapezoid)") + f"   risk {res.risk:.2f}")
    lines.append("raw: " + (" ".join(sorted(res.raw)) or "-"))
    lines.append("ALERT: " + (" ".join(f"{a.kind}:{a.level}" for a in sorted(res.alerts, key=lambda a: -a.level)) or "-"))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (340, 18 * len(lines) + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for n, s in enumerate(lines):
        text(frame, s, (8, 20 + 18 * n), 0.48, ui.COLORS[res.level] if s.startswith("ALERT: ") and res.alerts else WHITE)
    if res.top:
        text(frame, res.top.screen, (W // 2 - 120, 50), 1.2, ui.COLORS[res.level], 3)
        cv2.rectangle(frame, (0, 0), (W - 1, H - 1), ui.COLORS[res.level], 6)


def draw_panel(W, hist, t, window, events):
    """Plots for the last window seconds. hist: [(t, ttc, risk, raw, {kind: level})]."""
    kinds = list(C.ALERTS)
    ph = 120 + 13 * len(kinds) + 30
    panel = np.full((ph, W, 3), 25, np.uint8)
    x0, x1 = 90, W - 10
    tx = lambda tt: int(x1 - (t - tt) / window * (x1 - x0))

    def plot(y0, h, label, idx, vmax, marks=()):
        cv2.rectangle(panel, (x0, y0), (x1, y0 + h), (60, 60, 60), 1)
        text(panel, label, (4, y0 + 14), 0.4, GRAY)
        ty = lambda v: int(y0 + h - min(v, vmax) / vmax * h)
        for m, col in marks:
            cv2.line(panel, (x0, ty(m)), (x1, ty(m)), col, 1)
        prev = None
        for row in hist:
            v = row[idx]
            p = (tx(row[0]), ty(v)) if v is not None else None
            if p and prev:
                cv2.line(panel, prev, p, WHITE, 1, cv2.LINE_AA)
            prev = p

    plot(8, 50, "TTC s", 1, 10, [(C.TTC_WARNING, ui.COLORS[2]), (C.TTC_CRITICAL, ui.COLORS[3])])
    plot(64, 46, "risk", 2, 1.0, [(0.5, GRAY)])
    y = 118
    for kind in kinds:
        text(panel, kind, (4, y + 9), 0.36, GRAY)
        for a, b in zip(hist, hist[1:]):
            if kind in a[3]:
                cv2.line(panel, (tx(a[0]), y + 6), (tx(b[0]), y + 6), GRAY, 1)
            if kind in a[4]:
                cv2.rectangle(panel, (tx(a[0]), y + 1), (tx(b[0]), y + 11), ui.COLORS[a[4][kind]], -1)
        y += 13
    for s in range(int(t - window) + 1, int(t) + 1):
        if s >= 0 and s % 5 == 0:
            cv2.line(panel, (tx(s), ph - 6), (tx(s), ph), GRAY, 1)
            text(panel, f"{s}", (tx(s) - 8, ph - 8), 0.33, GRAY)
    cv2.line(panel, (x1, 0), (x1, ph), (0, 255, 0), 1)
    ev = "  ".join(f"{e[0]:.1f}s {e[1]}" for e in events[-6:])
    text(panel, "spoken: " + (ev or "-"), (x0, ph - 14), 0.4, WHITE)
    return panel


def load_wav(path, rate=22050):
    """Mono float; resampled to `rate` if needed."""
    if not path or not os.path.exists(path):
        return np.zeros(0)
    with wave.open(path) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        a = a.reshape(-1, w.getnchannels()).mean(1)
        if w.getframerate() != rate:
            n = int(len(a) * rate / w.getframerate())
            a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)
        return a


class SoundTrack:
    """Audio track like voice.Voice: one cue at a time, stale cues dropped."""
    RATE = 22050

    def __init__(self, t0):
        self.t0, self.busy_until = t0, -1e9
        self.clips = []                                # (start, end, sound, text)

    def say(self, cue, t):
        if cue is None:
            return
        start = max(t, self.busy_until)
        if start - t > C.VOICE_MAX_DELAY:
            return
        snd = np.concatenate([load_wav(chime_file(cue["level"])) if C.BEEP else np.zeros(0),
                              load_wav(phrase_file(cue))])
        self.busy_until = start + len(snd) / self.RATE
        self.clips.append((start, self.busy_until, snd, cue))

    def playing(self, t):
        return next((c[3] for c in self.clips if c[0] <= t < c[1]), None)

    def write(self, path, duration):
        out = np.zeros(int(duration * self.RATE) + self.RATE)
        for start, _, snd, _ in self.clips:
            i = int((start - self.t0) * self.RATE)
            if 0 <= i < len(out):
                out[i:i + len(snd)] += snd[:len(out) - i]
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.RATE)
            w.writeframes((np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes())


def mux_audio(video, audio):
    """Merge video and sound with ffmpeg (from imageio-ffmpeg), re-encoding video to H.264."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        print("  ffmpeg not found (pip install imageio-ffmpeg) - video without sound, audio separately:", audio)
        return
    tmp = video[:-4] + "_av.mp4"
    r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", video, "-i", audio, "-map", "0:v", "-map", "1:a",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "128k", "-shortest", tmp])
    if r.returncode == 0:
        os.replace(tmp, video)
        os.remove(audio)


def run(path, analyzer, args):
    analyzer.det.reset()
    analyzer._new()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 1 <= fps <= 120:
        fps = 30.0
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or None
    fi = 0
    if args.start:
        fi = int(args.start * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    end = (args.start + args.dur) if args.dur else None
    stride = args.stride or max(1, round(fps / 10))

    os.makedirs(args.out, exist_ok=True)
    name = stem(path) + (f"_{args.start:g}s" if args.start else "")
    vid_path = os.path.join(args.out, f"{name}_{'demo' if args.demo else 'debug'}.mp4")
    csv_f = open(os.path.join(args.out, f"{name}_frames.csv"), "w", newline="", encoding="utf-8")
    log = csv.writer(csv_f)
    log.writerow(["time", "level", "alerts", "raw", "ttc", "distance", "risk", "ego_kmh", "spoken"])

    writer, res, advice = None, None, None
    sound = SoundTrack(fi / fps)
    hist, events = deque(), []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        if end and t > end:
            break
        if res is None or n % stride == 0:
            res, advice = analyzer.step(t, frame)
            sound.say(advice.cue, t)
            if advice.cue:
                events.append((t, advice.cue["phrase"]))
                print(f"  {t:7.2f} s  {C.LEVEL_NAMES[advice.cue['level']]:9s} {advice.cue['phrase']}")
            hist.append((t, res.ttc, res.risk, set(res.raw), {a.kind: a.level for a in res.alerts}))
            log.writerow([f"{t:.3f}", res.level, " ".join(f"{a.kind}:{a.level}" for a in res.alerts),
                          " ".join(sorted(res.raw)), f"{res.ttc:.2f}" if res.ttc is not None else "",
                          f"{res.distance:.1f}" if res.distance is not None else "", f"{res.risk:.3f}",
                          f"{res.ego_speed * 3.6:.0f}" if res.ego_speed is not None else "",
                          advice.cue["phrase"] if advice.cue else ""])
        while hist and t - hist[0][0] > args.window:
            hist.popleft()

        k = args.width / frame.shape[1]
        frame = cv2.resize(frame, (args.width, round(frame.shape[0] * k)), interpolation=cv2.INTER_AREA)
        if args.demo:
            # what the driver gets: the coloured glow and the warning, nothing else - no boxes, distances or lane
            s = args.width / 1280
            ui.draw_glow(frame, advice.side, advice.level, advice.glow)
            if res.top:
                ui._label(frame, res.top.screen, (int(40 * s), int(80 * s)), 1.5 * s, ui.COLORS[res.level], max(2, int(3 * s)))
            cue = sound.playing(t)
            if cue:
                ui._text(frame, f'"{cue["phrase"]}"', (int(40 * args.width / 1280), frame.shape[0] - int(70 * args.width / 1280)),
                         0.8 * args.width / 1280, ui.COLORS[cue["level"]], 2)
            out = frame
        else:
            draw_debug(frame, res, t, fi, k)
            ui.draw_glow(frame, advice.side, advice.level, advice.glow)
            cue = sound.playing(t)
            if cue:
                text(frame, f"SOUND: {cue['phrase']}", (frame.shape[1] - 420, 30), 0.6, (0, 255, 255), 2)
            out = np.vstack([frame, draw_panel(args.width, list(hist), t, args.window, events)])
        if writer is None:
            writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out.shape[1], out.shape[0]))
        writer.write(out)

        fi += 1
        n += 1
        if n % int(fps * 10) == 0:
            pct = f" ({fi / total:.0%})" if total else ""
            print(f"  ...{t:.0f} s{pct}", flush=True)

    cap.release()
    csv_f.close()
    if writer:
        writer.release()
        if not args.no_audio:
            wav_path = vid_path[:-4] + ".wav"
            sound.write(wav_path, fi / fps - sound.t0)
            mux_audio(vid_path, wav_path)
    print(f"{vid_path}: {len(events)} spoken cues")
    return vid_path


def _scaled(res, k):
    """A shallow copy of the result with boxes scaled to the output frame (for ui.draw)."""
    import copy
    from behavior import Detection
    r = copy.copy(res)
    sc = lambda d: Detection(tuple(v * k for v in d.box), d.cls, d.tid) if d is not None else None
    r.threats = [copy.copy(th) for th in res.threats]
    for th in r.threats:
        th.det = sc(th.det)
    r.lead = sc(res.lead)
    return r


def main():
    from main import Analyzer
    ap = argparse.ArgumentParser(description="Co-pilot debug / demo video")
    ap.add_argument("videos", help="video or folder of videos")
    ap.add_argument("--demo", action="store_true", help="driver view only (no debug panels)")
    ap.add_argument("--start", type=float, default=0, help="start at second N")
    ap.add_argument("--dur", type=float, help="process N seconds")
    ap.add_argument("--stride", type=int, help="analyse every N-th frame (default ~10 analyses/s)")
    ap.add_argument("--width", type=int, default=1280, help="output width")
    ap.add_argument("--window", type=float, default=15, help="seconds of history on the plots")
    ap.add_argument("--out", default="debug", help="output folder")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    if os.path.isfile(args.videos):
        paths = [args.videos]
    else:
        paths = sorted(p for p in glob.glob(os.path.join(args.videos, "*")) if p.lower().endswith(VIDEO_EXT))
    if not paths:
        raise SystemExit(f"No videos: {args.videos}")
    analyzer = Analyzer.__new__(Analyzer)          # the same pipeline as the app, without its thread
    from detector import Detector
    from lanes import LaneDetector
    analyzer.det = Detector(args.model)
    analyzer.lanes = LaneDetector() if os.path.exists(C.LANE_MODEL) else None
    analyzer.can_speed = None
    analyzer._new()
    for p in paths:
        print(p)
        run(p, analyzer, args)


if __name__ == "__main__":
    main()
