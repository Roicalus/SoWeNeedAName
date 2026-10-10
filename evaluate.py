"""Evaluate the system on a folder of videos: alerts per colour per minute and, with labels, how many
dangers were caught and how early.

    python evaluate.py data/videos
    python evaluate.py data/nexar --labels data/nexar/labels.csv
    python evaluate.py data/bdd --tags data/bdd/tags.csv

labels.csv: file,start,end[,...] - dangerous moments (seconds); videos without rows are "normal driving"
(every yellow/red alert there is false). --tags: CSV (file,weather,timeofday,scene) for a breakdown.
Videos are first cached (tools/cache_tracks.py: detector, lanes, crosswalk, own motion), then the logic
runs from the cache, so a second run with other thresholds takes seconds.
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

import config as C
from tools.replay import events, run

VIDEO_EXT = (".mov", ".mp4", ".avi", ".mkv", ".webm", ".ogv")


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    ap = argparse.ArgumentParser(description="Co-pilot evaluation on a folder of videos")
    ap.add_argument("videos", help="folder with videos or one file")
    ap.add_argument("--labels", help="CSV with dangerous moments")
    ap.add_argument("--tags", help="conditions: CSV file,weather,timeofday,scene")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="eval_events.csv")
    args = ap.parse_args()

    paths = [args.videos] if os.path.isfile(args.videos) else sorted(
        p for p in glob.glob(os.path.join(args.videos, "*")) if p.lower().endswith(VIDEO_EXT))
    if not paths:
        raise SystemExit(f"No videos in {args.videos}")
    cmd = [sys.executable, "tools/cache_tracks.py", *paths, "--out", args.cache]
    if args.model:
        cmd += ["--model", args.model]
    subprocess.run(cmd, check=True)

    labels = defaultdict(list)
    if args.labels:
        with open(args.labels, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                labels[stem(r["file"])].append((float(r["start"]), float(r["end"])))
    tags = {}
    if args.tags:
        with open(args.tags, encoding="utf-8") as f:
            tags = {stem(r["file"]): r for r in csv.DictReader(f)}

    rows, per = [], {}
    for p in paths:
        name = stem(p)
        seq = run(os.path.join(args.cache, name + ".npz"))
        dur = (seq[-1][0] - seq[0][0]) / 60 if seq else 0
        m = {"min": dur}
        for lv in (1, 2, 3):
            ev = events(seq, lv)
            m[f"ev{lv}"] = len(ev)
            danger = labels.get(name, [])
            m[f"false{lv}"] = sum(not any(s - C.EVAL_EARLY <= t <= e for s, e in danger) for t, _ in ev)
            m[f"caught{lv}"] = sum(any(s - C.EVAL_EARLY <= t <= e for t, _ in ev) for s, e in danger)
            m[f"lead{lv}"] = [e - min(t for t, _ in ev if s - C.EVAL_EARLY <= t <= e) for s, e in danger
                              if any(s - C.EVAL_EARLY <= t <= e for t, _ in ev)]
        m["dangers"] = len(labels.get(name, []))
        level_at = {(t, k): lv for t, kinds in seq for k, lv in kinds.items()}
        for t, k in events(seq, 1):
            rows.append({"file": name, "time": round(t, 2), "kind": k, "level": level_at[(t, k)]})
        per[name] = m
        print(f"{name}: {dur:.1f} min, green+ {m['ev1']}, yellow+ {m['ev2']}, red {m['ev3']}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "time", "kind", "level"])
        w.writeheader()
        w.writerows(rows)

    def report(title, names):
        mins = sum(per[n]["min"] for n in names)
        line = f"{title:20s} videos {len(names):4d}  min {mins:6.1f}"
        for lv, nm in ((1, "green+"), (2, "yellow+"), (3, "red")):
            line += f" | {nm} {sum(per[n][f'ev{lv}'] for n in names) / max(mins, 1e-6):5.2f}/min"
        print(line)
        dangers = sum(per[n]["dangers"] for n in names)
        if dangers:
            for lv, nm in ((2, "yellow+"), (3, "red")):
                leads = [x for n in names for x in per[n][f"lead{lv}"]]
                print(f"{'':20s} {nm:8s} caught {sum(per[n][f'caught{lv}'] for n in names)}/{dangers}"
                      f"  median lead {np.median(leads) if leads else float('nan'):.1f} s"
                      f"  false {sum(per[n][f'false{lv}'] for n in names) / max(mins, 1e-6):.2f}/min")

    print()
    report("ALL", list(per))
    for attr in ("timeofday", "weather", "scene"):
        groups = defaultdict(list)
        for n in per:
            groups[str(tags.get(n, {}).get(attr, "unknown"))].append(n)
        if tags and len(groups) > 1:
            print(f"--- {attr}")
            for v, ns in sorted(groups.items()):
                report(v, ns)
    print(f"\nevents: {args.out}")


if __name__ == "__main__":
    main()
