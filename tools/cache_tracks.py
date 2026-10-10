"""Run the detector + tracker (+ lanes, crosswalk, own motion) over videos once and cache the result.

The threat logic can then be checked and tuned on hundreds of videos in seconds (tools/replay.py,
tools/train_risk.py) without running the networks again.

    python tools/cache_tracks.py data/dada/videos data/bdd data/videos
    python tools/cache_tracks.py data/bdd --no-lanes
    python tools/cache_tracks.py data/dada/videos --workers 3      # several videos in parallel

Cache: cache/<video>.npz
    t      - analysis time, s (~10 per second, as on the device)
    det    - N x 8: analysis index, track id, COCO class, x1, y1, x2, y2 (pixels), unused (-1)
    lanes  - T x 4 x R: x of lane lines (frame fractions) on rows lane_rows, NaN - no line
    ego    - T x 3: own speed (m/s), yaw (frame widths/s), ok (0/1) from egomotion.py
    xwalk  - T x 4: crosswalk box (frame fractions), NaN - none
    W, H, fps, model
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

VIDEO_EXT = (".mov", ".mp4", ".avi", ".mkv", ".webm", ".ogv")


def cache_path(video, out="cache"):
    return os.path.join(out, os.path.splitext(os.path.basename(video))[0] + ".npz")


_state = {}


def _init(model, use_lanes):
    from detector import Detector
    _state["det"] = Detector(model)
    _state["lanes"] = None
    if use_lanes:
        from lanes import LaneDetector
        _state["lanes"] = LaneDetector()


def process(path, rate, out):
    import crosswalk
    from egomotion import EgoMotion
    det, lanes = _state["det"], _state["lanes"]
    det.reset()
    ego = EgoMotion()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 1 <= fps <= 120:
        fps = 30.0
    stride = max(1, round(fps / rate))
    ts, rows, lane_rows, ego_rows, xw_rows = [], [], [], [], []
    i = k = 0
    W = H = 0
    tic = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            H, W = frame.shape[:2]
            t = i / fps
            dets = det(frame)
            for d in dets:
                rows.append((k, d.tid, d.cls, *d.box, -1))
            boxes = [d.box for d in dets]
            if lanes is not None:
                lane_rows.append(lanes.detect(frame))
            ego_rows.append(ego.update(t, frame, boxes))
            xw = crosswalk.detect(frame, boxes)
            xw_rows.append(xw if xw is not None else (np.nan,) * 4)
            ts.append(t)
            k += 1
        i += 1
    cap.release()
    extra = {}
    if lanes is not None:
        extra = {"lanes": np.array(lane_rows, np.float32), "lane_rows": lanes.rows}
    np.savez_compressed(out, t=np.array(ts), det=np.array(rows, np.float32).reshape(-1, 8), W=W, H=H, fps=fps,
                        model=det.model_name, ego=np.array(ego_rows, np.float32).reshape(-1, 3),
                        xwalk=np.array(xw_rows, np.float32).reshape(-1, 4), **extra)
    return os.path.basename(path), k, k / max(time.perf_counter() - tic, 1e-6)


def main():
    ap = argparse.ArgumentParser(description="Cache detections, lanes, crosswalks and own motion")
    ap.add_argument("paths", nargs="+", help="videos or folders")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--rate", type=float, default=10, help="analyses per second")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-lanes", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute even if the cache exists")
    ap.add_argument("--workers", type=int, default=1, help="videos processed in parallel")
    args = ap.parse_args()

    videos = []
    for p in args.paths:
        videos += [p] if os.path.isfile(p) else sorted(
            v for v in glob.glob(os.path.join(p, "*")) if v.lower().endswith(VIDEO_EXT))
    os.makedirs(args.out, exist_ok=True)
    todo = [v for v in videos if args.force or not os.path.exists(cache_path(v, args.out))]
    print(f"videos: {len(videos)}, to process: {len(todo)}", flush=True)
    use_lanes = not args.no_lanes and os.path.exists(C.LANE_MODEL)
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.model, use_lanes)) as pool:
        futs = [pool.submit(process, v, args.rate, cache_path(v, args.out)) for v in todo]
        for n, f in enumerate(futs, 1):
            name, k, speed = f.result()
            print(f"[{n}/{len(todo)}] {name}: {k} analyses, {speed:.0f}/s", flush=True)


if __name__ == "__main__":
    main()
