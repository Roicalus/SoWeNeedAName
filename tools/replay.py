"""Run the threat logic over the cache (tools/cache_tracks.py) and print metrics - seconds, no networks.

    python tools/replay.py                                    # DADA (crashes) + BDD (normal driving)
    python tools/replay.py --split test                       # only the held-out half of DADA
    python tools/replay.py --set CUTIN_MIN_SPEED=0.5 TTC_WARNING=2.5     # try other thresholds
    python tools/replay.py --details                          # per DADA video

Metrics (per colour level: "yellow+" = yellow or red, "red" = red only):
  caught     - share of DADA videos with an alert inside [danger start - EVAL_EARLY, crash];
  lead       - median time from the first such alert to the crash (larger is better: it is the
               opposite of "warning too late");
  false/min  - alerts per minute on BDD (normal driving: every yellow/red is a false alarm, green
               is allowed) and in DADA before the danger starts (minus EVAL_EARLY).
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402


def load(path):
    z = np.load(path, allow_pickle=True)
    by = defaultdict(list)
    for row in z["det"]:
        by[int(row[0])].append(row)
    T = len(z["t"])
    lanes = z["lanes"] if "lanes" in z.files else None
    ego = z["ego"] if "ego" in z.files else None
    xwalk = z["xwalk"] if "xwalk" in z.files else np.full((T, 4), np.nan)
    return {"t": z["t"], "by": by, "lanes": lanes, "ego": ego, "xwalk": xwalk, "W": int(z["W"]), "H": int(z["H"])}


def steps(cache, engine=None, use_lanes=True):
    """Yields (t, FrameResult) for every analysis step of a cached video."""
    from behavior import BehaviorEngine, Detection
    eng = engine or BehaviorEngine()
    for k, t in enumerate(cache["t"]):
        dets = [Detection(tuple(r[3:7]), int(r[2]), int(r[1])) for r in cache["by"].get(k, [])]
        lanes = cache["lanes"][k] if (use_lanes and cache["lanes"] is not None) else None
        ego = tuple(cache["ego"][k]) if cache["ego"] is not None else None
        xw = cache["xwalk"][k]
        res = eng.update(float(t), cache["W"], cache["H"], dets, None, lanes, ego,
                         None if np.isnan(xw[0]) else tuple(xw))
        yield float(t), res


def run(path, use_lanes=True):
    """[(time, {kind: level})] on every analysis step."""
    return [(t, {a.kind: a.level for a in res.alerts}) for t, res in steps(load(path), use_lanes=use_lanes)]


def events(seq, min_level):
    """Alert onsets of at least min_level (new if the same kind was absent for EVENT_GAP seconds)."""
    ev, last = [], {}
    for t, kinds in seq:
        for k, lvl in kinds.items():
            if lvl < min_level:
                continue
            if t - last.get(k, -1e9) > C.EVENT_GAP:
                ev.append((t, k))
            last[k] = t
    return ev


def is_bdd(name):
    return len(name) == 17 and name[8] == "-"


def load_labels(path, split="all"):
    labels = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if split == "all" or r["split"] == split:
                    labels[os.path.splitext(r["file"])[0]] = r
    return labels


def main():
    ap = argparse.ArgumentParser(description="Threat logic metrics from the cache")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--labels", default="data/dada/labels.csv")
    ap.add_argument("--split", choices=["tune", "test", "all"], default="all")
    ap.add_argument("--no-lanes", action="store_true", help="ignore lane markings")
    ap.add_argument("--set", nargs="*", default=[], help="override config values: NAME=value")
    ap.add_argument("--details", action="store_true", help="per DADA video")
    ap.add_argument("--limit", type=int, default=0, help="at most N videos of each source (quick runs)")
    args = ap.parse_args()
    for kv in args.set:
        k, v = kv.split("=", 1)
        setattr(C, k, type(getattr(C, k))(v) if hasattr(C, k) else float(v))

    labels = load_labels(args.labels, args.split)
    stats = defaultdict(lambda: {lv: {"n": 0, "hit": 0, "leads": [], "false": 0, "min": 0.0} for lv in (2, 3)})
    kinds_hit = defaultdict(int)
    bdd = {lv: {"false": 0, "min": 0.0, "n": 0, "kinds": defaultdict(int)} for lv in (1, 2, 3)}
    n_dada = n_bdd = 0
    for path in sorted(glob.glob(os.path.join(args.cache, "*.npz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        dada, bddv = name in labels, is_bdd(name)
        if not (dada or bddv):
            continue
        if args.limit and ((dada and n_dada >= args.limit) or (bddv and n_bdd >= args.limit)):
            continue
        n_dada += dada
        n_bdd += bddv
        seq = run(path, not args.no_lanes)
        if not seq:
            continue
        dur = seq[-1][0] - seq[0][0]
        if bddv:
            for lv in (1, 2, 3):
                ev = events(seq, lv)
                bdd[lv]["n"] += 1
                bdd[lv]["min"] += dur / 60
                bdd[lv]["false"] += len(ev)
                for _, k in ev:
                    bdd[lv]["kinds"][k] += 1
            continue
        L = labels[name]
        start, end = float(L["start"]), float(L["end"])
        line = []
        for lv in (2, 3):
            ev = events(seq, lv)
            inside = [(t, k) for t, k in ev if start - C.EVAL_EARLY <= t <= end]
            before = [(t, k) for t, k in ev if t < start - C.EVAL_EARLY]
            for key in (L["group"], f"{L['group']}/{L['light']}", "ALL DADA"):
                s = stats[key][lv]
                s["n"] += 1
                s["false"] += len(before)
                s["min"] += max(start - C.EVAL_EARLY - seq[0][0], 0) / 60
                if inside:
                    s["hit"] += 1
                    s["leads"].append(end - min(t for t, _ in inside))
            if lv == 2:
                for k in {k for _, k in inside}:
                    kinds_hit[(L["group"], k)] += 1
            line.append(f"{end - min(t for t, _ in inside):4.1f}s ({inside[0][1]})" if inside else "   miss   ")
        if args.details:
            print(f"  {name:18s} {L['group']:9s} {L['light']:5s} danger {start:5.1f}-{end:5.1f}: "
                  f"yellow+ {line[0]:18s} red {line[1]:18s} | {L['text'][:55]}")

    print(f"\n{'group':22s} {'videos':>6s} | {'yellow+ caught':>14s} {'lead':>6s} {'false/min':>9s} |"
          f" {'red caught':>10s} {'lead':>6s} {'false/min':>9s}")
    for key in sorted(stats, key=lambda k: (k == "ALL DADA", k)):
        cells = []
        for lv in (2, 3):
            s = stats[key][lv]
            lead = f"{np.median(s['leads']):.1f}s" if s["leads"] else "-"
            fpm = s["false"] / s["min"] if s["min"] else 0
            cells.append(f"{s['hit'] / max(s['n'], 1):{14 if lv == 2 else 10}.0%} {lead:>6s} {fpm:9.2f}")
        print(f"{key:22s} {stats[key][2]['n']:6d} | {cells[0]} | {cells[1]}")
    if bdd[1]["n"]:
        print(f"\nBDD normal driving: {bdd[1]['n']} videos, {bdd[1]['min']:.1f} min")
        for lv, title in ((1, "any colour"), (2, "yellow+"), (3, "red")):
            b = bdd[lv]
            print(f"  {title:10s} {b['false'] / b['min']:5.2f}/min  " +
                  ", ".join(f"{k}: {v}" for k, v in sorted(b["kinds"].items(), key=lambda x: -x[1])))
    if kinds_hit:
        print("\nWhich yellow+ alerts fired in the danger window (group: kind x videos):")
        for g in sorted({g for g, _ in kinds_hit}):
            print(f"  {g:9s} " + ", ".join(f"{k} {v}" for (gg, k), v in sorted(kinds_hit.items(), key=lambda x: -x[1])
                                           if gg == g))


if __name__ == "__main__":
    main()
