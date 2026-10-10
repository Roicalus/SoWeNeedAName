"""Отладочное видео: всё, что «видит и думает» система, поверх ролика — чтобы оценить глазами.

    python debug_video.py data/videos/krakow_cc_by.webm
    python debug_video.py data/videos/krakow_cc_by.webm --start 60 --dur 30
    python debug_video.py data/videos --dur 60            (все ролики папки)

На выходе в debug/: <ролик>_debug.mp4 и <ролик>_frames.csv (значения на каждый анализ).
Обработка без окна и звука, кадр за кадром (не в реальном времени), результат воспроизводимый.
Анализируется каждый --stride-й кадр (как на устройстве), между ними держится последний результат.

Что на видео:
  рамки  — все треки: #id, класс, статус. Машины: LEAD — машина впереди, cand — кандидат,
           lane — вне трапеции полосы, off N.Nw — смещение в своих ширинах больше порога,
           side — видна сбоку, edge — у края кадра, hood — капот. Пешеходы: PED, far, lane, off N.Nm.
  полосы — своя полоса залита цветом уровня опасности (по разметке; без неё — белая трапеция),
           HOOD_Y — синяя линия (ниже — капот).
  угрозы — рамка цвета уровня, тип и дистанция; стрелка — куда участник сместится через секунду.
  слева  — время, TTC, дистанция, скорость сближения, ширина рамки, виляние.
  снизу  — графики за последние --window секунд: TTC (пороги 3 и 1.5 с), ширина рамки,
           амплитуда виляния; полосы по типам предупреждений: тонкая — условие выполнено (сырое),
           толстая — предупреждение подтверждено (то, что услышит водитель).
  звук   — то, что услышит водитель: сигнал (beeps.py) + фраза из sounds/, с теми же паузами
           между повторами, что в voice.py. Пока звучит, справа сверху горит «SOUND».
           Нужен ffmpeg из пакета imageio-ffmpeg; без него видео будет без звука.
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
from behavior import BehaviorEngine, corridor_polygon
from detector import Detector
from lanes import LaneDetector
from evaluate import VIDEO_EXT, stem
from voice import beep_file

CLS = {**C.VEHICLE_CLASSES, C.PERSON_CLASS: "person"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
GRAY, WHITE = (150, 150, 150), (255, 255, 255)
KIND_COLOR = {"danger": (0, 90, 255), "ttc": (0, 0, 255), "crash": (0, 0, 200), "wrongway": (60, 0, 255), "cutin": (0, 180, 255),
              "crossing": (0, 120, 255), "closing": (0, 140, 255), "weave": (255, 0, 255),
              "ped": (0, 220, 255), "brakelight": (80, 80, 255), "close": (255, 200, 0)}


def text(img, s, org, scale=0.5, color=WHITE, thick=1):
    # тень того же начертания: у толстой обводки в OpenCV другой шаг букв, и она «съезжает»
    cv2.putText(img, s, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def dashed_hline(img, y, color, dash=12):
    for x in range(0, img.shape[1], 2 * dash):
        cv2.line(img, (x, y), (x + dash, y), color, 1, cv2.LINE_AA)


def draw_scene(frame, res, t, fi, k=1.0):
    """k — во сколько раз кадр уменьшен относительно исходного (рамки в исходных пикселях)."""
    H, W = frame.shape[:2]
    if not ui.draw_lanes(frame, res.lanes, res.level):
        cv2.polylines(frame, [corridor_polygon(W, H)], True, WHITE, 1, cv2.LINE_AA)
    cv2.line(frame, (0, int(C.HOOD_Y * H)), (W, int(C.HOOD_Y * H)), (255, 120, 0), 1, cv2.LINE_AA)

    lead_id = res.lead.tid if res.lead else None
    threat_ids = {th.det.tid for th in res.threats}
    for d, status in res.tracks:                  # все треки: #id, класс, u (полос), боковая скорость
        x1, y1, x2, y2 = (int(v * k) for v in d.box)
        color = (0, 200, 0) if d.tid == lead_id else GRAY
        if d.tid not in threat_ids:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if d.tid == lead_id else 1)
        tag = "LEAD " if d.tid == lead_id else ""
        text(frame, f"#{d.tid} {CLS.get(d.cls, d.cls)} {tag}{status}", (x1, y2 + 14), 0.4, color)
    ui.draw_threats(frame, res, k, 0.5)

    # левая панель с числами
    lines = [f"t {t:7.2f}s  frame {fi}"]
    if res.lead:
        w = (res.lead.box[2] - res.lead.box[0]) * k / W
        ttc = f"{res.ttc:.2f}s" if res.ttc is not None else "-"
        lines.append(f"lead #{res.lead.tid}  w {w:.3f}")
        lines.append(f"TTC {ttc}" + ("  ONCOMING" if res.oncoming else ""))
        if res.distance is not None:
            lines.append(f"dist ~{res.distance:.1f} m")
            if res.ttc:
                lines.append(f"closing {res.distance / res.ttc:.1f} m/s")
        if res.weave:
            lines.append(f"weave amp {res.weave[0]:.2f} (>{C.WEAVE_REL_STD})  rev {res.weave[1]} (>={C.WEAVE_MIN_REVERSALS})")
    else:
        lines.append("lead: none")
    lines.append("lanes: " + ("yes" if res.lanes is not None else "no (trapezoid)") + f"   risk {res.risk:.2f}")
    lines.append("raw: " + (" ".join(sorted(res.raw)) or "-"))
    lines.append("ALERT: " + (" ".join(sorted(a.kind for a in res.alerts)) or "-"))
    box_h = 18 * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (330, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for n, s in enumerate(lines):
        text(frame, s, (8, 20 + 18 * n), 0.48, (0, 0, 255) if s.startswith("ALERT: ") and res.alerts else WHITE)

    if res.top:                                   # то, что видит водитель
        text(frame, res.top.screen, (W // 2 - 120, 50), 1.2, ui.COLORS[res.level], 3)
        cv2.rectangle(frame, (0, 0), (W - 1, H - 1), ui.COLORS[res.level], 6)


def draw_panel(W, hist, t, window, events):
    """Графики за последние window секунд. hist: [(t, ttc, w, weave_amp, raw, active)]."""
    ph = 375
    panel = np.full((ph, W, 3), 25, np.uint8)
    x0, x1 = 90, W - 10
    tx = lambda tt: int(x1 - (t - tt) / window * (x1 - x0))

    def plot(y0, h, label, idx, vmax, marks=()):
        cv2.rectangle(panel, (x0, y0), (x1, y0 + h), (60, 60, 60), 1)
        text(panel, label, (4, y0 + 14), 0.4, GRAY)
        ty = lambda v: int(y0 + h - min(v, vmax) / vmax * h)
        for m, col in marks:
            cv2.line(panel, (x0, ty(m)), (x1, ty(m)), col, 1)
            text(panel, f"{m:g}", (x0 - 28, ty(m) + 4), 0.35, col)
        prev = None
        for row in hist:
            v = row[idx]
            p = (tx(row[0]), ty(v)) if v is not None else None
            if p and prev:
                cv2.line(panel, prev, p, WHITE, 1, cv2.LINE_AA)
            prev = p

    plot(8, 60, "TTC s", 1, 10, [(C.TTC_WARNING, (0, 140, 255)), (C.TTC_CRITICAL, (0, 0, 255))])
    plot(76, 40, "lead w", 2, 0.5, [(C.CLOSE_WIDTH, (255, 200, 0)), (C.TTC_MIN_WIDTH, GRAY)])
    plot(124, 30, "weave", 3, 0.5, [(C.WEAVE_REL_STD, (255, 0, 255))])

    y = 162
    for kind in C.ALERTS:
        col = KIND_COLOR[kind]
        text(panel, kind, (4, y + 9), 0.38, col)
        for a, b in zip(hist, hist[1:]):
            if kind in a[4]:
                cv2.line(panel, (tx(a[0]), y + 6), (tx(b[0]), y + 6), col, 2)
            if kind in a[5]:
                cv2.rectangle(panel, (tx(a[0]), y), (tx(b[0]), y + 12), col, -1)
        y += 15

    for s in range(int(t - window) + 1, int(t) + 1):          # секундные метки
        if s >= 0 and s % 5 == 0:
            cv2.line(panel, (tx(s), ph - 6), (tx(s), ph), GRAY, 1)
            text(panel, f"{s}", (tx(s) - 8, ph - 8), 0.33, GRAY)
    cv2.line(panel, (x1, 0), (x1, ph), (0, 255, 0), 1)
    ev = "  ".join(f"{e[0]:.1f}s {e[1]}" for e in events[-6:])
    text(panel, "events: " + (ev or "-"), (x0, ph - 22), 0.4, WHITE)
    return panel


def load_wav(path):
    """Моно int16 → float; все наши .wav — 22050 Гц (см. beeps.py, make_sounds.py)."""
    if not os.path.exists(path):
        return np.zeros(0)
    with wave.open(path) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        return a.reshape(-1, w.getnchannels()).mean(1)


class SoundTrack:
    """Звуковая дорожка как у voice.Voice: анти-спам по типу, очередь, выброс устаревших фраз."""
    RATE = 22050

    def __init__(self, t0):
        self.t0, self.last, self.busy_until = t0, {}, -1e9
        self.clips = []                                # (начало, конец, звук, тип)

    def say(self, alert, t):
        if alert is None or t - self.last.get(alert.kind, -1e9) < C.VOICE_COOLDOWN[alert.kind]:
            return
        self.last[alert.kind] = t
        start = max(t, self.busy_until)
        if start - t > C.VOICE_MAX_DELAY:
            return
        snd = np.concatenate([load_wav(beep_file(alert.level)) if C.BEEP else np.zeros(0),
                              load_wav(os.path.join("sounds", f"{alert.kind}.wav"))])
        self.busy_until = start + len(snd) / self.RATE
        self.clips.append((start, self.busy_until, snd, alert.kind))

    def playing(self, t):
        return next((c[3] for c in self.clips if c[0] <= t < c[1]), None)

    def write(self, path, duration):
        out = np.zeros(int(duration * self.RATE) + self.RATE)
        for start, _, snd, _ in self.clips:
            i = int((start - self.t0) * self.RATE)
            out[i:i + len(snd)] += snd[:len(out) - i]
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.RATE)
            w.writeframes((np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes())


def mux_audio(video, audio):
    """Склеить видео и звук через ffmpeg (из imageio-ffmpeg). Видео не перекодируется."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        print("  ffmpeg не найден (pip install imageio-ffmpeg) — видео без звука, звук отдельно:", audio)
        return
    tmp = video[:-4] + "_av.mp4"
    r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", video, "-i", audio, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", tmp])
    if r.returncode == 0:
        os.replace(tmp, video)
        os.remove(audio)


def open_writer(path, fps, size):
    for fourcc in ("avc1", "mp4v"):              # avc1 (H.264) открывается везде; mp4v — запасной
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w
    raise SystemExit(f"Не удалось создать {path}")


def run(path, det, args):
    det.reset()
    engine = BehaviorEngine()
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
    vid_path = os.path.join(args.out, f"{name}_debug.mp4")
    csv_f = open(os.path.join(args.out, f"{name}_frames.csv"), "w", newline="", encoding="utf-8")
    log = csv.writer(csv_f)
    log.writerow(["time", "lead_id", "lead_w", "ttc", "distance", "oncoming", "weave_amp", "weave_rev",
                  "n_tracks", "raw", "alerts"])

    writer, res = None, None
    sound = SoundTrack(fi / fps)
    hist, events, last_active = deque(), [], {}
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        if end and t > end:
            break
        if res is None or n % stride == 0:
            lanes = det.lanes.detect(frame) if det.lanes else None
            res = engine.update(t, frame.shape[1], frame.shape[0], det(frame), frame, lanes)
            w = (res.lead.box[2] - res.lead.box[0]) / frame.shape[1] if res.lead else None
            active = {a.kind for a in res.alerts}
            hist.append((t, res.ttc, w, res.weave[0] if res.weave else None, set(res.raw), active))
            sound.say(res.top, t)
            for k in active:
                if t - last_active.get(k, -1e9) > C.EVENT_GAP:
                    events.append((t, k))
                    print(f"  {t:7.2f} с  {k}")
                last_active[k] = t
            log.writerow([f"{t:.3f}", res.lead.tid if res.lead else "", f"{w:.4f}" if w else "",
                          f"{res.ttc:.2f}" if res.ttc is not None else "",
                          f"{res.distance:.1f}" if res.distance is not None else "", int(res.oncoming),
                          f"{res.weave[0]:.3f}" if res.weave else "", res.weave[1] if res.weave else "",
                          len(res.tracks), " ".join(sorted(res.raw)), " ".join(sorted(active))])
        while hist and t - hist[0][0] > args.window:
            hist.popleft()

        k = args.width / frame.shape[1]
        frame = cv2.resize(frame, (args.width, round(frame.shape[0] * k)), interpolation=cv2.INTER_AREA)
        draw_scene(frame, res, t, fi, k)
        playing = sound.playing(t)
        if playing:
            text(frame, f"SOUND: {playing}", (frame.shape[1] - 230, 30), 0.7, (0, 255, 255), 2)
        out = np.vstack([frame, draw_panel(args.width, list(hist), t, args.window, events)])
        if writer is None:
            writer = open_writer(vid_path, fps, (out.shape[1], out.shape[0]))
        writer.write(out)

        fi += 1
        n += 1
        if n % int(fps * 10) == 0:
            pct = f" ({fi / total:.0%})" if total else ""
            print(f"  ...{t:.0f} с{pct}", flush=True)

    cap.release()
    csv_f.close()
    if writer:
        writer.release()
        if not args.no_audio:
            wav_path = vid_path[:-4] + ".wav"
            sound.write(wav_path, fi / fps - sound.t0)
            mux_audio(vid_path, wav_path)
    print(f"{vid_path}: {len(events)} событий")
    return vid_path


def main():
    ap = argparse.ArgumentParser(description="Отладочное видео «Второго пилота»")
    ap.add_argument("videos", help="ролик или папка с роликами")
    ap.add_argument("--start", type=float, default=0, help="начать с N-й секунды")
    ap.add_argument("--dur", type=float, help="обработать N секунд")
    ap.add_argument("--stride", type=int, help="анализировать каждый N-й кадр (по умолчанию ~10 анализов/с)")
    ap.add_argument("--width", type=int, default=1280, help="ширина выходного видео")
    ap.add_argument("--window", type=float, default=15, help="сколько секунд истории на графиках")
    ap.add_argument("--out", default="debug", help="папка для результатов")
    ap.add_argument("--no-lanes", action="store_true", help="без распознавания разметки (трапеция)")
    ap.add_argument("--no-audio", action="store_true", help="без звуковой дорожки")
    ap.add_argument("--model", default=C.MODEL)
    args = ap.parse_args()

    if os.path.isfile(args.videos):
        paths = [args.videos]
    else:
        paths = sorted(p for p in glob.glob(os.path.join(args.videos, "*")) if p.lower().endswith(VIDEO_EXT))
    if not paths:
        raise SystemExit(f"Нет роликов: {args.videos}")
    det = Detector(args.model)
    det.lanes = None if args.no_lanes or not os.path.exists(C.LANE_MODEL) else LaneDetector()
    for p in paths:
        print(p)
        run(p, det, args)


if __name__ == "__main__":
    main()
