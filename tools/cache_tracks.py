"""Один раз прогнать детектор + трекер (+ полосы) по роликам и сохранить результат в кэш.

Логику угроз потом можно проверять и настраивать на сотнях роликов за секунды
(tools/replay.py), не запуская нейросети заново.

    python tools/cache_tracks.py data/dada/videos data/bdd data/videos
    python tools/cache_tracks.py data/bdd --no-lanes

Кэш: cache/<ролик>.npz
    t      — время анализа, с (частота ~10 раз/с, как на устройстве)
    det    — N x 8: индекс анализа, id трека, класс COCO, x1, y1, x2, y2 (пиксели), уверенность (-1 — нет)
    lanes  — T x 4 x R: x линий разметки (доли ширины) на строках lane_rows, NaN — линии нет
    W, H, fps, model
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from detector import Detector  # noqa: E402

VIDEO_EXT = (".mov", ".mp4", ".avi", ".mkv", ".webm", ".ogv")


def cache_path(video, out="cache"):
    return os.path.join(out, os.path.splitext(os.path.basename(video))[0] + ".npz")


def process(path, det, lanes, rate, out):
    det.reset()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 1 <= fps <= 120:
        fps = 30.0
    stride = max(1, round(fps / rate))
    ts, rows, lane_rows = [], [], []
    i = k = 0
    W = H = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            H, W = frame.shape[:2]
            t = i / fps
            for d in det(frame):
                rows.append((k, d.tid, d.cls, *d.box, -1))
            if lanes is not None:
                lane_rows.append(lanes.detect(frame))
            ts.append(t)
            k += 1
        i += 1
    cap.release()
    extra = {}
    if lanes is not None:
        extra = {"lanes": np.array(lane_rows, np.float32), "lane_rows": lanes.rows}
    np.savez_compressed(out, t=np.array(ts), det=np.array(rows, np.float32).reshape(-1, 8), W=W, H=H, fps=fps,
                        model=det.model_name, **extra)
    return k


def main():
    ap = argparse.ArgumentParser(description="Кэш детекций и полос для быстрых экспериментов")
    ap.add_argument("paths", nargs="+", help="ролики или папки")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--rate", type=float, default=10, help="анализов в секунду")
    ap.add_argument("--model", default=C.MODEL)
    ap.add_argument("--no-lanes", action="store_true")
    ap.add_argument("--force", action="store_true", help="пересчитать, даже если кэш есть")
    args = ap.parse_args()

    videos = []
    for p in args.paths:
        videos += [p] if os.path.isfile(p) else sorted(
            v for v in glob.glob(os.path.join(p, "*")) if v.lower().endswith(VIDEO_EXT))
    os.makedirs(args.out, exist_ok=True)
    det = Detector(args.model)
    det.model_name = os.path.basename(args.model)
    lanes = None
    if not args.no_lanes:
        from lanes import LaneDetector
        lanes = LaneDetector()
    for n, v in enumerate(videos, 1):
        out = cache_path(v, args.out)
        if os.path.exists(out) and not args.force:
            continue
        tic = time.perf_counter()
        k = process(v, det, lanes, args.rate, out)
        print(f"[{n}/{len(videos)}] {os.path.basename(v)}: {k} анализов, {k / (time.perf_counter() - tic):.0f}/с",
              flush=True)


if __name__ == "__main__":
    main()
