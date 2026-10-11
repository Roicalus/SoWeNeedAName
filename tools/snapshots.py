"""Snapshots of alert onsets: the frame with the full overlay at the moment an alert of a kind starts.

    python tools/snapshots.py --cache cache --videos data/bdd --kinds cutin ped crash --level 3 --per-kind 6
The fastest way to see WHY a rule fires on normal driving (false alarms) or on crashes.
Writes a contact sheet per kind to snaps/<kind>.jpg.
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
import ui  # noqa: E402
from tools.replay import load, steps  # noqa: E402

VIDEO_EXT = (".mov", ".mp4", ".avi", ".mkv", ".webm", ".ogv")


def main():
    ap = argparse.ArgumentParser(description="Snapshots of alert onsets")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--videos", nargs="+", default=["data/bdd"])
    ap.add_argument("--kinds", nargs="+", default=list(C.ALERTS))
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--per-kind", type=int, default=6)
    ap.add_argument("--out", default="snaps")
    args = ap.parse_args()
    vids = {}
    for d in args.videos:
        for p in glob.glob(os.path.join(d, "*")):
            if p.lower().endswith(VIDEO_EXT):
                vids[os.path.splitext(os.path.basename(p))[0]] = p
    shots = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.cache, "*.npz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name not in vids or all(len(shots[k]) >= args.per_kind for k in args.kinds):
            continue
        last = {}
        cap = None
        for t, res in steps(load(path)):
            for a in res.alerts:
                if a.kind in args.kinds and a.level >= args.level and t - last.get(a.kind, -1e9) > C.EVENT_GAP \
                        and len(shots[a.kind]) < args.per_kind:
                    cap = cap or cv2.VideoCapture(vids[name])
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    ok, frame = cap.read()
                    if ok:
                        ui.draw(frame, res)
                        f = cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1])))
                        cv2.putText(f, f"{name[:12]} {t:.1f}s", (5, f.shape[0] - 8), 0, 0.5, (255, 255, 255), 1)
                        shots[a.kind].append(f)
                last.setdefault(a.kind, -1e9)
            for a in res.alerts:
                last[a.kind] = t
    os.makedirs(args.out, exist_ok=True)
    for k, fs in shots.items():
        h = max(f.shape[0] for f in fs)
        fs = [cv2.copyMakeBorder(f, 0, h - f.shape[0], 0, 0, cv2.BORDER_CONSTANT) for f in fs]
        while len(fs) % 3:
            fs.append(np.zeros_like(fs[0]))
        cv2.imwrite(os.path.join(args.out, f"{k}.jpg"), np.vstack([np.hstack(fs[i:i + 3]) for i in range(0, len(fs), 3)]))
        print(k, len(fs))


if __name__ == "__main__":
    main()
