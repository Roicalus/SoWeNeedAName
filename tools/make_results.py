"""Honest numbers for the README and the site: the full pipeline (rules + advisor) replayed from the cache
on the held-out "test" half of DADA-2000 crashes / labelled reels, on BDD100K normal driving and on long
highway drives, broken down by condition.

    python tools/make_results.py --cache cache_final --ms 45 --detector logs/detector_ap.json
Counts:
  caught      - a yellow (or red) situation started between EVAL_EARLY s before the labelled danger and the crash;
  lead        - median time from that first hint to the crash;
  situations  - starts of a yellow+ (red) level after at least EVENT_GAP s without one (one situation may
                involve several rules at once - closing + hard braking - it is counted once);
  spoken      - hints actually spoken by the advisor (advisor.py budget), per minute.
Writes web/data/results.json and prints Markdown tables for the README.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from advisor import Advisor  # noqa: E402
from tools.replay import is_bdd, load, load_labels, steps  # noqa: E402
from tools.train_risk import LABEL_FILES  # noqa: E402


def bdd_timeofday():
    """BDD video name -> daytime / night / dawn or dusk (key frames of the videos share their names)."""
    tod = {}
    for f in glob.glob("data/bdd_det/meta/period_*.jsonl"):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            tod[d["file_name"][:-4]] = d["label"]
    return tod


def replay(path):
    """[(t, level)] per step and [(t, level, phrase)] spoken cues."""
    cache = load(path)
    adv = Advisor()
    seq, cues = [], []
    for t, res in steps(cache):
        seq.append((t, res.level))
        a = adv.update(t, res, cache["W"])
        if a.cue:
            cues.append((t, a.cue["level"], a.cue["phrase"]))
    return seq, cues


def situations(seq, min_level):
    out, last = [], -1e9
    for t, lv in seq:
        if lv >= min_level:
            if t - last > C.EVENT_GAP:
                out.append(t)
            last = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ms", type=float, help="measured ms per analysis (detector + lanes + crosswalk + logic)")
    ap.add_argument("--detector", help="json with detector AP rows")
    ap.add_argument("--out", default="web/data/results.json")
    args = ap.parse_args()
    labels = {}
    for f in LABEL_FILES:
        labels.update(load_labels(f, args.split))
    tod = bdd_timeofday()
    crash = defaultdict(list)                       # group -> [(caught_y, lead_y, caught_r)]
    calm = defaultdict(lambda: [0, 0, 0, 0.0])      # group -> [yellow+ situations, red situations, spoken, minutes]
    per_video = {}
    for p in sorted(glob.glob(os.path.join(args.cache, "*.npz"))):
        name = os.path.basename(p)[:-4]
        if name in labels:
            L = labels[name]
            seq, cues = replay(p)
            start, end = float(L["start"]), float(L["end"])
            win = lambda lv: [t for t in situations(seq, lv) if start - C.EVAL_EARLY <= t <= end]
            y, r = win(2), win(3)
            row = (bool(y), end - min(y) if y else None, bool(r))
            src = "reels" if name.startswith("reel_") else "dada"
            keys = [f"{src}:all"]
            if src == "dada":
                keys += [f"dada:{L['light']}", f"dada:group:{L['group']}"]
                if L.get("weather") in ("rainy", "snowy", "foggy"):
                    keys.append("dada:bad weather")
            for k in keys:
                crash[k].append(row)
        elif is_bdd(name) or not name.startswith(("dada_", "reel_")):
            seq, cues = replay(p)
            if len(seq) < 2:
                continue
            mins = (seq[-1][0] - seq[0][0]) / 60
            if is_bdd(name):
                keys = ["bdd:all", f"bdd:{tod.get(name, 'unknown')}"]
            else:
                keys = [f"clip:{name}"]
            for k in keys:
                c = calm[k]
                c[0] += len(situations(seq, 2))
                c[1] += len(situations(seq, 3))
                c[2] += sum(1 for _, lv, _ in cues if lv >= 2)
                c[3] += mins
            per_video[name] = {"minutes": mins, "cues": [(round(t, 1), lv, ph) for t, lv, ph in cues]}

    def crash_row(title, rows):
        leads = [x[1] for x in rows if x[1] is not None]
        return {"set": title, "n": len(rows), "recall_y": sum(x[0] for x in rows) / max(len(rows), 1),
                "lead_y": float(np.median(leads)) if leads else None, "recall_r": sum(x[2] for x in rows) / max(len(rows), 1),
                "false_y": None, "false_r": None, "spoken": None}
    names = {"dada:all": "Crashes (DADA-2000 test)", "dada:day": "Crashes · day", "dada:night": "Crashes · night",
             "dada:bad weather": "Crashes · rain / snow / fog", "reels:all": "Aggressive-driving clips (social media, test half)"}
    rows = [crash_row(names[k], crash[k]) for k in names if crash.get(k)]
    for k in sorted(k for k in crash if k.startswith("dada:group:")):
        rows.append(crash_row("Crashes · " + k.split(":")[-1].replace("_", " "), crash[k]))
    calm_titles = {"bdd:all": "Normal driving (BDD100K)", "bdd:daytime": "Normal driving · day", "bdd:night": "Normal driving · night",
                   "bdd:dawn or dusk": "Normal driving · dawn/dusk",
                   "clip:tehran_chamran_ccby": "Highway in dense traffic (Tehran)", "clip:vw_highway_ccby": "Highway (VW POV)",
                   "clip:austria_slovenia_ccby": "Small town + motorway (Austria)", "clip:urban": "Calm city drive (Germany)",
                   "clip:krakow_peds": "City with pedestrians (Kraków)"}
    for k, title in calm_titles.items():
        if k in calm and calm[k][3] > 0:
            y, r, sp, m = calm[k]
            rows.append({"set": f"{title} · {m:.0f} min", "n": None, "recall_y": None, "lead_y": None, "recall_r": None,
                         "false_y": y / m, "false_r": r / m, "spoken": sp / m})
    head = rows[0] if rows else {}
    b = calm.get("bdd:all", [0, 0, 0, 0])
    out = {"headline": {"recall_y": head.get("recall_y"), "lead_y": head.get("lead_y"),
                        "false_y_per_min": b[0] / b[3] if b[3] else None, "ms_per_frame": args.ms},
           "rows": rows, "split": args.split, "eval_early_s": C.EVAL_EARLY, "per_video": per_video}
    if args.detector and os.path.exists(args.detector):
        out["detector"] = json.load(open(args.detector, encoding="utf-8"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    pct = lambda v: "–" if v is None else f"{v:.0%}"
    num = lambda v: "–" if v is None else f"{v:.2f}"
    print("| Set | Clips | Yellow+ in advance | Median lead | Red in advance | Yellow+ situations / min | Red / min | Spoken yellow+ / min |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lead = "–" if r["lead_y"] is None else f"{r['lead_y']:.1f} s"
        print(f"| {r['set']} | {r['n'] or '–'} | {pct(r['recall_y'])} | {lead} | {pct(r['recall_r'])} | {num(r['false_y'])} | "
              f"{num(r['false_r'])} | {num(r['spoken'])} |")
    print("\nwritten", args.out)


if __name__ == "__main__":
    main()
