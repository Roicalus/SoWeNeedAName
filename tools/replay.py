"""Прогон логики угроз по кэшу (tools/cache_tracks.py) и метрики — за секунды, без нейросетей.

    python tools/replay.py                                    # DADA (опасные) + BDD (обычная езда)
    python tools/replay.py --split test                       # только контрольная половина DADA
    python tools/replay.py --set CUTIN_MIN_SPEED=0.5 TTC_WARNING=2.5     # попробовать другие пороги

Метрики:
  поймано    — доля роликов DADA, где предупреждение было в окне [начало опасности − EVAL_EARLY, авария];
  до аварии  — медиана времени от первого такого предупреждения до момента аварии (больше — лучше:
               это и есть «запаздывание» наоборот);
  ложные/мин — предупреждения на BDD (обычная езда) и в DADA до начала опасности (минус EVAL_EARLY).
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
    det = z["det"]
    by = defaultdict(list)
    for row in det:
        by[int(row[0])].append(row)
    lanes = z["lanes"] if "lanes" in z.files else None
    return z["t"], by, lanes, int(z["W"]), int(z["H"])


def run(path, use_lanes=True):
    """Список (время, множество активных типов, уровень) на каждом шаге анализа."""
    from behavior import BehaviorEngine, Detection
    ts, by, lanes, W, H = load(path)
    eng = BehaviorEngine()
    out = []
    for k, t in enumerate(ts):
        dets = [Detection(tuple(r[3:7]), int(r[2]), int(r[1])) for r in by.get(k, [])]
        res = eng.update(float(t), W, H, dets, None, lanes[k] if (use_lanes and lanes is not None) else None)
        out.append((float(t), {a.kind for a in res.alerts}, res.level))
    return out


def events(steps, min_level=1):
    """Начала предупреждений (новое, если того же типа не было EVENT_GAP секунд)."""
    ev, last = [], {}
    for t, kinds, level in steps:
        for k in kinds:
            if C.ALERTS[k][0] >= min_level or level >= min_level:
                if t - last.get(k, -1e9) > C.EVENT_GAP:
                    ev.append((t, k))
                last[k] = t
    return ev


def main():
    ap = argparse.ArgumentParser(description="Метрики логики угроз по кэшу")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--labels", default="data/dada/labels.csv")
    ap.add_argument("--split", choices=["tune", "test", "all"], default="all")
    ap.add_argument("--no-lanes", action="store_true", help="не использовать разметку (как раньше)")
    ap.add_argument("--set", nargs="*", default=[], help="переопределить параметры config: ИМЯ=значение")
    ap.add_argument("--min-level", type=int, default=1, help="считать только предупреждения от этого уровня")
    ap.add_argument("--details", action="store_true", help="построчно по роликам DADA")
    args = ap.parse_args()
    for kv in args.set:
        k, v = kv.split("=", 1)
        setattr(C, k, type(getattr(C, k))(v) if hasattr(C, k) else float(v))

    labels = {}
    if os.path.exists(args.labels):
        with open(args.labels, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if args.split == "all" or r["split"] == args.split:
                    labels[os.path.splitext(r["file"])[0]] = r

    stats = defaultdict(lambda: {"n": 0, "hit": 0, "leads": [], "false": 0, "min": 0.0})
    kinds_hit = defaultdict(int)
    bdd = {"false": 0, "min": 0.0, "n": 0, "kinds": defaultdict(int)}
    for path in sorted(glob.glob(os.path.join(args.cache, "*.npz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        is_dada = name in labels
        is_bdd = not name.startswith("dada_") and len(name) == 17 and "-" in name
        if not (is_dada or is_bdd):
            continue
        steps = run(path, not args.no_lanes)
        ev = events(steps, args.min_level)
        dur = steps[-1][0] - steps[0][0] if steps else 0
        if is_bdd:
            bdd["n"] += 1
            bdd["min"] += dur / 60
            bdd["false"] += len(ev)
            for _, k in ev:
                bdd["kinds"][k] += 1
            continue
        L = labels[name]
        start, end = float(L["start"]), float(L["end"])
        inside = [(t, k) for t, k in ev if start - C.EVAL_EARLY <= t <= end]
        before = [(t, k) for t, k in ev if t < start - C.EVAL_EARLY]
        for key in (L["group"], f"{L['group']}/{L['light']}", "ВСЕ DADA"):
            s = stats[key]
            s["n"] += 1
            s["false"] += len(before)
            s["min"] += max(start - C.EVAL_EARLY - steps[0][0], 0) / 60
            if inside:
                s["hit"] += 1
                s["leads"].append(end - min(t for t, _ in inside))
        for _, k in inside:
            kinds_hit[(L["group"], k)] += 1
        if args.details:
            first = f"{end - min(t for t, _ in inside):4.1f} с до аварии ({inside[0][1]})" if inside else "пропуск"
            print(f"  {name:18s} {L['group']:9s} {L['light']:5s} опасность {start:5.1f}–{end:5.1f}: {first}"
                  f"  | ложных до: {len(before)}  | {L['text'][:60]}")

    print(f"\n{'группа':22s} {'роликов':>7s} {'поймано':>8s} {'до аварии (медиана)':>20s} {'ложные/мин до опасности':>24s}")
    for key in sorted(stats, key=lambda k: (k == "ВСЕ DADA", k)):
        s = stats[key]
        lead = f"{np.median(s['leads']):.1f} с" if s["leads"] else "-"
        fpm = s["false"] / s["min"] if s["min"] else 0
        print(f"{key:22s} {s['n']:7d} {s['hit'] / max(s['n'], 1):8.0%} {lead:>20s} {fpm:24.2f}")
    if bdd["n"]:
        print(f"\nBDD (обычная езда): роликов {bdd['n']}, {bdd['min']:.1f} мин, предупреждений {bdd['false']}"
              f" = {bdd['false'] / bdd['min']:.2f}/мин  " + ", ".join(f"{k}: {v}" for k, v in sorted(bdd["kinds"].items())))
    if kinds_hit:
        print("\nКакие предупреждения срабатывали в опасном окне (группа: тип × роликов):")
        groups = sorted({g for g, _ in kinds_hit})
        for g in groups:
            print(f"  {g:9s} " + ", ".join(f"{k} {v}" for (gg, k), v in sorted(kinds_hit.items()) if gg == g))


if __name__ == "__main__":
    main()
