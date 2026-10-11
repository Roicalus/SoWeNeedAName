"""Train the learned part of the system (risk.py) on cached videos: DADA-2000 crashes, labelled reels and
BDD100K normal driving.

    python tools/train_risk.py --cache cache
    python tools/train_risk.py --cache cache --yellow-budget 0.8 --red-budget 0.15

Two models:
  threat scorer - scores every rule threat from its own features (behavior.THREAT_FEATURES). A threat keeps
                  its yellow / red level only if the score >= gate_yellow / gate_red, else it is shown green.
                  This is the main false-alarm filter.
  scene model   - scores the whole scene (pooled behavior.FEATURES) and raises "DANGER" when no single rule
                  fired but the scene looks like the seconds before a crash.
Labels per analysis step (10 per second):
  1 - from the moment the danger starts to the crash (DADA) / closest moment (reels);
  0 - earlier than 2 s before the danger in the same clip, and all of BDD (normal driving);
  the transition is not used.
Honesty: videos are split as in the labels (split=tune / test; BDD by a hash of the name). Model choice,
gates and thresholds use only "tune" - with out-of-fold predictions (4 folds by video), so a model is
never judged on videos it was trained on. "test" is used once, for the final numbers.
Writes models/risk.json (+ LightGBM text models next to it).
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from tools.replay import is_bdd, load, load_labels, steps  # noqa: E402

LABEL_FILES = ["data/dada/labels.csv", "data/reels/labels.csv"]
K_FOLDS = 4


def collect(path):
    """Per step: time, pooled scene features, rule level (+10 = physically certain);
    per rule threat: step index, level, certain flag, THREAT_FEATURES."""
    from behavior import BehaviorEngine
    from risk import FeaturePool, RiskModel
    eng = BehaviorEngine()
    eng.risk = RiskModel(path="")
    pool = FeaturePool()
    ts, X, rules, t_idx, t_lvl, t_cert, t_X, t_tid = [], [], [], [], [], [], [], []
    cache = load(path)
    for k, (t, res) in enumerate(steps(cache, eng)):
        ts.append(t)
        X.append(pool.vector(t, res.features))
        certain = any(eng._certain(th, t) for th in res.threats)
        rules.append(res.level + (10 if certain else 0))
        for th in res.threats:
            if th.feat is None:
                continue
            t_idx.append(k)
            t_lvl.append(th.level)
            t_cert.append(eng._certain(th, t))
            t_X.append(th.feat)
            t_tid.append(th.det.tid)
    nf = len(t_X[0]) if t_X else 0
    return {"t": np.array(ts, float), "X": np.array(X, np.float32), "rules": np.array(rules, np.int8),
            "t_idx": np.array(t_idx, np.int32), "t_lvl": np.array(t_lvl, np.int8), "t_cert": np.array(t_cert, bool),
            "t_tid": np.array(t_tid, np.int32), "boxes": {k: [(int(r[1]), *r[3:7]) for r in rows]
                                                          for k, rows in cache["by"].items()}, "W": cache["W"],
            "t_X": np.array(t_X, np.float32).reshape(-1, nf) if nf else np.zeros((0, 0), np.float32)}


def hazard_tids(d, end):
    """Track ids of the object that most likely takes part in the crash: the largest box in the central
    80% of the frame around the crash moment (DADA labels do not say which object it is)."""
    ts = d["t"]
    out = set()
    for k in np.nonzero(np.abs(ts - end) <= 0.35)[0]:
        rows = [r for r in d["boxes"].get(int(k), []) if 0.1 * d["W"] < (r[1] + r[3]) / 2 < 0.9 * d["W"]]
        if rows:
            out.add(max(rows, key=lambda r: (r[3] - r[1]) * (r[4] - r[2]))[0])
    return out


def bdd_split(name):
    return "test" if int(hashlib.md5(name.encode()).hexdigest(), 16) % 2 else "tune"


def split_of(name, labels):
    return bdd_split(name) if is_bdd(name) else labels[name]["split"]


def step_labels(name, ts, labels):
    if is_bdd(name):
        return np.zeros(len(ts))
    L = labels[name]
    start, end = float(L["start"]), float(L["end"])
    y = np.full(len(ts), -1.0)
    y[(ts >= start) & (ts <= end)] = 1
    y[ts < start - 2] = 0
    return y


def onsets(ts, levels, min_level):
    """Alert onsets of at least min_level (new after a gap of EVENT_GAP seconds)."""
    ev, last = [], -1e9
    for t, lv in zip(ts, levels):
        if lv >= min_level:
            if t - last > C.EVENT_GAP:
                ev.append(t)
            last = t
    return ev


def model_levels(ts, p, thr_y, thr_r):
    """Scene-model levels: the probability must stay above a threshold RISK_CONFIRM seconds (vectorised)."""
    out = np.zeros(len(p), np.int8)
    for lv, thr in ((2, thr_y), (3, thr_r)):
        on = p >= thr
        if not on.any():
            continue
        start = on & ~np.concatenate([[False], on[:-1]])
        run_start = ts[np.maximum.accumulate(np.where(start, np.arange(len(ts)), 0))]
        out[on & (ts - run_start >= C.RISK_CONFIRM)] = lv
    return out


def step_levels(d, score, gy, gr, p=None, thr_y=1.01, thr_r=1.01):
    """Per-step level after gating every threat with its score, plus the scene model."""
    lev = np.zeros(len(d["t"]), np.int8)
    if len(d["t_idx"]):
        gl = d["t_lvl"].copy()
        if score is not None:
            gl[(gl >= 3) & (score < gr) & ~d["t_cert"]] = 2
            gl[(gl >= 2) & (score < gy) & ~d["t_cert"]] = 1
        np.maximum.at(lev, d["t_idx"], gl)
    if p is not None:
        lev = np.maximum(lev, model_levels(d["t"], p, thr_y, thr_r))
    return lev


def evaluate(data, labels, names, scores, gy, gr, probs=None, thr_y=1.01, thr_r=1.01):
    """{level: (recall, median lead, false/min BDD, false/min crash clips before danger, n, BDD min)}."""
    out = {}
    levs = {n: step_levels(data[n], scores.get(n) if scores else None, gy, gr,
                           probs.get(n) if probs else None, thr_y, thr_r) for n in names}
    for lv in (2, 3):
        hits, leads, n_, fa, fmin, bfa, bmin = 0, [], 0, 0, 0.0, 0, 0.0
        for name in names:
            ts = data[name]["t"]
            if not len(ts):
                continue
            ev = onsets(ts, levs[name], lv)
            if is_bdd(name):
                bfa += len(ev)
                bmin += (ts[-1] - ts[0]) / 60
                continue
            L = labels[name]
            start, end = float(L["start"]), float(L["end"])
            n_ += 1
            inside = [t for t in ev if start - C.EVAL_EARLY <= t <= end]
            fa += len([t for t in ev if t < start - C.EVAL_EARLY])
            fmin += max(start - C.EVAL_EARLY - ts[0], 0) / 60
            if inside:
                hits += 1
                leads.append(end - min(inside))
        out[lv] = (hits / max(n_, 1), float(np.median(leads)) if leads else float("nan"),
                   bfa / max(bmin, 1e-6), fa / max(fmin, 1e-6), n_, bmin)
    return out


def lgbm(seed=0, n=500):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              verbose=-1, random_state=seed)


def fit(X, y, seed=0):
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
    m, s = X.mean(0), np.where(X.std(0) > 1e-6, X.std(0), 1.0)
    return lgbm(seed).fit((X - m) / s, y, sample_weight=w), m, s


def main():
    ap = argparse.ArgumentParser(description="Train the threat scorer and the scene model")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--yellow-budget", type=float, default=0.8, help="false yellow+ events per minute on BDD-tune")
    ap.add_argument("--red-budget", type=float, default=0.15, help="false red events per minute on BDD-tune")
    ap.add_argument("--out", default=C.RISK_MODEL)
    ap.add_argument("--features-cache", default=None, help="collected features (default <cache>/risk_features.npz)")
    ap.add_argument("--recollect", action="store_true")
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    from behavior import THREAT_FEATURES
    from risk import drop_mask, feature_names
    fcache = args.features_cache or os.path.join(args.cache, "risk_features.npz")

    labels = {}
    for f in LABEL_FILES:
        labels.update(load_labels(f))
    names = [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(args.cache, "*.npz")))]
    names = [n for n in names if (n in labels or is_bdd(n)) and n != "risk_features"]
    data = {}
    if os.path.exists(fcache) and not args.recollect:
        data = np.load(fcache, allow_pickle=True)["data"].item()
        if any(not isinstance(v, dict) or "t_tid" not in v for v in data.values()):
            data = {}
    todo = [n for n in names if n not in data]
    tic = time.time()
    for k, n in enumerate(todo, 1):
        data[n] = collect(os.path.join(args.cache, n + ".npz"))
        if k % 200 == 0:
            print(f"  features {k}/{len(todo)} ({time.time() - tic:.0f} s)", flush=True)
    if todo:
        np.savez_compressed(fcache, data=np.array(data, dtype=object))
    data = {n: data[n] for n in names}
    drop = drop_mask()
    for n in names:
        if len(data[n]["X"]):
            data[n]["X"][:, drop] = 0.0
    groups = {s: [n for n in names if split_of(n, labels) == s] for s in ("tune", "test")}
    print(f"videos: crash / near-miss {sum(n in labels for n in names)}, normal driving (BDD) "
          f"{sum(is_bdd(n) for n in names)}; tune {len(groups['tune'])}, test {len(groups['test'])}", flush=True)

    def threat_xy(group):
        Xs, ys = [], []
        for n in group:
            d = data[n]
            if not len(d["t_idx"]):
                continue
            y = step_labels(n, d["t"], labels)[d["t_idx"]]
            if n in labels:          # in the danger window only the crash participant is a positive example
                y[(y == 1) & ~np.isin(d["t_tid"], list(hazard_tids(d, float(labels[n]["end"]))))] = -1
            Xs.append(d["t_X"][y >= 0])
            ys.append(y[y >= 0])
        return np.concatenate(Xs), np.concatenate(ys)

    def scene_xy(group):
        Xs, ys = [], []
        for n in group:
            if is_bdd(n):          # scene model: negatives from the crash clips only (see risk.DROP)
                continue
            d = data[n]
            y = step_labels(n, d["t"], labels)
            Xs.append(d["X"][y >= 0])
            ys.append(y[y >= 0])
        return np.concatenate(Xs), np.concatenate(ys)

    # ---- out-of-fold predictions on tune
    tune = list(groups["tune"])
    np.random.default_rng(0).shuffle(tune)
    folds = [tune[i::K_FOLDS] for i in range(K_FOLDS)]
    oof_t, oof_s = {}, {}
    for i, fold in enumerate(folds):
        rest = [n for f in folds[:i] + folds[i + 1:] for n in f]
        mt, m_, s_ = fit(*threat_xy(rest), seed=i)
        ms, sm, ss = fit(*scene_xy(rest), seed=i)
        for n in fold:
            d = data[n]
            oof_t[n] = mt.predict_proba((d["t_X"] - m_) / s_)[:, 1] if len(d["t_idx"]) else np.zeros(0)
            oof_s[n] = ms.predict_proba((d["X"] - sm) / ss)[:, 1] if len(d["X"]) else np.zeros(0)
        print(f"  fold {i + 1}/{K_FOLDS}", flush=True)

    def auc_threat(sc, group):
        ys, ps = [], []
        for n in group:
            d = data[n]
            if not len(d["t_idx"]):
                continue
            y = step_labels(n, d["t"], labels)[d["t_idx"]]
            ys.append(y[y >= 0])
            ps.append(sc[n][y >= 0])
        return roc_auc_score(np.concatenate(ys), np.concatenate(ps))
    print(f"threat scorer out-of-fold AUC on tune: {auc_threat(oof_t, tune):.3f}")

    # ---- gates on tune (out-of-fold): best recall within the false-alarm budget on BDD-tune
    grid = np.round(np.concatenate([np.linspace(0.0, 0.9, 19), [0.93, 0.95, 0.97, 0.98, 0.99]]), 3)
    best = None
    for gy in grid:
        for gr in grid[grid >= gy]:
            r = evaluate(data, labels, tune, oof_t, gy, gr)
            if r[2][2] <= args.yellow_budget and r[3][2] <= args.red_budget:
                sc = r[2][0] + 0.5 * r[3][0] + 0.1 * min(np.nan_to_num(r[2][1]), 3) / 3
                if best is None or sc > best[0]:
                    best = (sc, gy, gr)
    gate_y, gate_r = (best[1], best[2]) if best else (0.99, 0.99)

    def pick(lv, low, thr_y=1.01):
        for thr in np.linspace(max(low, 0.5), 0.995, 50):
            ty, tr = (thr, 1.01) if lv == 2 else (thr_y, thr)
            r = evaluate(data, labels, tune, oof_t, gate_y, gate_r, oof_s, ty, tr)
            if r[2][2] <= args.yellow_budget and r[3][2] <= args.red_budget:
                return float(thr)
        return 1.01
    thr_y = pick(2, 0.5)
    thr_r = pick(3, thr_y, thr_y) if thr_y <= 1 else 1.01
    print(f"gates: yellow {gate_y:.2f}, red {gate_r:.2f}; scene DANGER thresholds: yellow {thr_y:.3f}, red {thr_r:.3f}")

    # ---- final models on all of tune (these ship); test is scored with them
    mt, t_mean, t_scale = fit(*threat_xy(groups["tune"]))
    ms, mean, scale = fit(*scene_xy(groups["tune"]))
    sc_test = {n: mt.predict_proba((data[n]["t_X"] - t_mean) / t_scale)[:, 1] if len(data[n]["t_idx"]) else np.zeros(0)
               for n in groups["test"]}
    p_test = {n: ms.predict_proba((data[n]["X"] - mean) / scale)[:, 1] if len(data[n]["X"]) else np.zeros(0)
              for n in groups["test"]}
    print(f"threat scorer AUC on test: {auc_threat(sc_test, groups['test']):.3f}")

    print(f"\n{'':6s} {'':22s} | {'yellow+ caught':>14s} {'lead':>6s} {'false/min BDD':>13s} {'crash-clip pre':>14s} |"
          f" {'red caught':>10s} {'lead':>6s} {'false/min BDD':>13s}")
    for split, sc, pr in (("tune", oof_t, oof_s), ("test", sc_test, p_test)):
        for title, use_sc, use_p in (("rules only", False, False), ("rules + threat scorer", True, False),
                                     ("+ scene model", True, True)):
            r = evaluate(data, labels, groups[split], sc if use_sc else None, gate_y, gate_r,
                         pr if use_p else None, thr_y, thr_r)
            y_, r_ = r[2], r[3]
            print(f"{split:6s} {title:22s} | {y_[0]:14.0%} {y_[1]:5.1f}s {y_[2]:13.2f} {y_[3]:14.2f} |"
                  f" {r_[0]:10.0%} {r_[1]:5.1f}s {r_[2]:13.2f}")
    r = evaluate(data, labels, groups["test"], sc_test, gate_y, gate_r, p_test, thr_y, thr_r)
    print(f"(test: {r[2][4]} crash / near-miss clips, {r[2][5]:.0f} min of normal driving; step-level simulation -"
          f" run tools/replay.py --split test for the exact numbers of the full pipeline)")

    base = os.path.splitext(os.path.basename(args.out))[0]
    out_dir = os.path.dirname(args.out) or "."
    mt.booster_.save_model(os.path.join(out_dir, base + "_threat.txt"))
    ms.booster_.save_model(os.path.join(out_dir, base + "_scene.txt"))
    model = {"kind": "lgbm", "model_file": base + "_scene.txt", "features": feature_names(), "drop": drop.tolist(),
             "mean": mean.tolist(), "scale": scale.tolist(), "threshold": thr_y, "threshold_red": thr_r,
             "gate_yellow": float(gate_y), "gate_red": float(gate_r),
             "threat": {"model_file": base + "_threat.txt", "features": THREAT_FEATURES,
                        "mean": t_mean.tolist(), "scale": t_scale.tolist()},
             "budget": {"yellow_per_min": args.yellow_budget, "red_per_min": args.red_budget},
             "trained_on": {"crash_tune": sum(1 for n in groups["tune"] if n in labels),
                            "bdd_tune": sum(1 for n in groups["tune"] if is_bdd(n))}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=1)
    print(f"\nmodel: {args.out}")
    imp = mt.booster_.feature_importance("gain")
    print("threat scorer, most important:", ", ".join(THREAT_FEATURES[i] for i in np.argsort(-imp)[:10]))


if __name__ == "__main__":
    main()
