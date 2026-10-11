"""Automatic threshold tuning of the rules (config.py) on the cached "tune" half of the data.

    python tools/tune.py --cache cache --iters 80 --workers 8
    python tools/tune.py --cache cache --iters 40 --start tune_best.json      # continue from a result

Objective (higher is better), measured on DADA-tune crashes and BDD-tune normal driving:
    recall(yellow+) + 0.5 * recall(red) + 0.2 * min(median lead, 3 s) / 3
    - 1.0 * max(0, false yellow+/min on BDD - YELLOW_BUDGET) - 3.0 * max(0, false red/min - RED_BUDGET)
Random search over SPACE, then local refinement around the best point. Writes tune_best.json and
prints the config lines to paste. The held-out "test" half is never used here.
"""
import argparse
import glob
import json
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

YELLOW_BUDGET, RED_BUDGET = 1.0, 0.2
SPACE = {   # name: (low, high, type)
    "TTC_WARNING": (2.0, 3.5, float), "TTC_CRITICAL": (1.0, 1.8, float), "CRITICAL_MIN_WIDTH": (0.08, 0.16, float),
    "IN_LANE": (0.3, 0.5, float),
    "CUTIN_MIN_SPEED": (0.5, 1.2, float), "CUTIN_HORIZON": (0.6, 1.5, float), "CUTIN_TTC": (2.0, 4.0, float),
    "CUTIN_MAX_DIST": (15.0, 30.0, float), "CUTIN_MIN_AGE": (0.3, 1.2, float), "STATIC_TOL": (0.2, 0.5, float),
    "VRU_NEAR": (8.0, 15.0, float), "VRU_MIN_SPEED": (0.4, 1.0, float), "VRU_HORIZON": (0.8, 2.0, float),
    "VRU_DIST": (15.0, 30.0, float), "VRU_SUDDEN_DIST": (8.0, 20.0, float), "VRU_SUDDEN_U": (0.6, 1.4, float),
    "VRU_STEP_U": (1.0, 2.0, float),
    "BRAKE_RATE": (0.12, 0.45, float), "BRAKE_JERK": (0.15, 0.6, float), "BRAKE_MAX_DIST": (15.0, 40.0, float),
    "CRASH_RISE": (0.6, 1.5, float), "CRASH_ASPECT": (0.4, 1.2, float),
    "WEAVE_REL_STD": (0.15, 0.35, float), "WEAVE_MIN_REVERSALS": (2, 4, int),
    "LEAD_MAX_CLOSING": (9.0, 16.0, float), "WRONGWAY_TTC": (2.5, 5.0, float),
    "CONFIRM_cutin": (0.1, 0.5, float), "CONFIRM_ped": (0.1, 0.5, float), "CONFIRM_closing": (0.1, 0.6, float),
    "CONFIRM_hardbrake": (0.0, 0.4, float), "CONFIRM_crossing": (0.1, 0.5, float), "CONFIRM_crash": (0.0, 0.4, float),
}

_cache = {}


def apply(params):
    for k, v in params.items():
        if k.startswith("CONFIRM_"):
            C.CONFIRM_SEC[k[8:]] = v
        else:
            setattr(C, k, v)


def eval_video(job):
    """(params, path) -> (name, duration s, [(t, max level)] onsets per level 2/3)."""
    params, path = job
    from tools.replay import events, load, steps
    from behavior import BehaviorEngine
    from risk import RiskModel
    apply(params)
    if path not in _cache:
        _cache[path] = load(path)
    eng = BehaviorEngine()
    eng.risk = RiskModel(path="")
    seq = [(t, {a.kind: a.level for a in res.alerts}) for t, res in steps(_cache[path], eng)]
    dur = seq[-1][0] - seq[0][0] if seq else 0.0
    return os.path.basename(path)[:-4], dur, {lv: events(seq, lv) for lv in (2, 3)}


def score(results, labels):
    s = {"n": 0, "y": 0, "r": 0, "leads": [], "bmin": 0.0, "by": 0, "br": 0}
    for name, dur, ev in results:
        if name in labels:
            L = labels[name]
            start, end = float(L["start"]), float(L["end"])
            s["n"] += 1
            iny = [t for t, _ in ev[2] if start - C.EVAL_EARLY <= t <= end]
            inr = [t for t, _ in ev[3] if start - C.EVAL_EARLY <= t <= end]
            s["y"] += bool(iny)
            s["r"] += bool(inr)
            if iny:
                s["leads"].append(end - min(iny))
        else:
            s["bmin"] += dur / 60
            s["by"] += len(ev[2])
            s["br"] += len(ev[3])
    ry, rr = s["y"] / max(s["n"], 1), s["r"] / max(s["n"], 1)
    lead = float(np.median(s["leads"])) if s["leads"] else 0.0
    fy, fr = s["by"] / max(s["bmin"], 1e-6), s["br"] / max(s["bmin"], 1e-6)
    obj = ry + 0.5 * rr + 0.2 * min(lead, 3) / 3 - 1.0 * max(0, fy - YELLOW_BUDGET) - 3.0 * max(0, fr - RED_BUDGET)
    return obj, {"recall_y": ry, "recall_r": rr, "lead": lead, "false_y": fy, "false_r": fr, "n": s["n"],
                 "bdd_min": s["bmin"]}


def current():
    out = {}
    for k in SPACE:
        out[k] = C.CONFIRM_SEC[k[8:]] if k.startswith("CONFIRM_") else getattr(C, k)
    return out


def sample(rnd, center=None, scale=1.0):
    p = {}
    for k, (lo, hi, typ) in SPACE.items():
        if center is None:
            v = rnd.uniform(lo, hi)
        else:
            v = center[k] + rnd.gauss(0, 0.15 * scale * (hi - lo))
        v = min(max(v, lo), hi)
        p[k] = int(round(v)) if typ is int else round(v, 3)
    return p


def main():
    from tools.replay import is_bdd, load_labels
    from tools.train_risk import LABEL_FILES, bdd_split
    ap = argparse.ArgumentParser(description="Tune rule thresholds")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start", help="json with a starting point")
    ap.add_argument("--out", default="tune_best.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    labels = {}
    for f in LABEL_FILES:
        labels.update(load_labels(f, "tune"))
    paths = []
    for p in sorted(glob.glob(os.path.join(args.cache, "*.npz"))):
        n = os.path.basename(p)[:-4]
        if n in labels or (is_bdd(n) and bdd_split(n) == "tune"):
            paths.append(p)
    print(f"tune videos: crash {sum(os.path.basename(p)[:-4] in labels for p in paths)}, "
          f"BDD {sum(is_bdd(os.path.basename(p)[:-4]) for p in paths)}", flush=True)
    rnd = random.Random(args.seed)
    best_p = json.load(open(args.start))["params"] if args.start else current()
    with Pool(args.workers) as pool:
        def evaluate(params):
            res = pool.map(eval_video, [(params, p) for p in paths], chunksize=4)
            return score(res, labels)
        best, info = evaluate(best_p)
        print(f"start: {best:.3f} {info}", flush=True)
        for it in range(args.iters):
            local = it >= args.iters // 3
            p = sample(rnd, best_p if local else None, scale=1.0 if it < 2 * args.iters // 3 else 0.5)
            if not local and rnd.random() < 0.5:          # half of the global samples start from the best
                p = sample(rnd, best_p, scale=2.0)
            sc, inf = evaluate(p)
            mark = ""
            if sc > best:
                best, best_p, info, mark = sc, p, inf, "  *"
                with open(args.out, "w") as f:
                    json.dump({"score": best, "info": info, "params": best_p}, f, indent=1)
            print(f"[{it + 1}/{args.iters}] {sc:.3f} y {inf['recall_y']:.0%} r {inf['recall_r']:.0%} "
                  f"lead {inf['lead']:.1f}s fy {inf['false_y']:.2f} fr {inf['false_r']:.2f}{mark}", flush=True)
    print(f"\nbest {best:.3f} {info}")
    for k, v in best_p.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
