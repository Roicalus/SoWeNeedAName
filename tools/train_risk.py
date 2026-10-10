"""Train the risk model (risk.py) on cached videos: DADA-2000 crashes, labelled reels, BDD100K normal driving.

    python tools/train_risk.py
    python tools/train_risk.py --yellow-budget 0.6 --red-budget 0.1     # false alarms per minute on BDD

Step labels (10 analyses per second):
  1 - from the moment the danger starts to the crash (DADA) / closest moment (reels);
  0 - earlier than 2 s before the danger, and all of BDD (normal driving);
  the transition is not used.
Videos are split as in the labels (split=tune / test; BDD by a hash of the name): everything below -
model choice, thresholds - uses only "tune"; "test" is touched once for the final honest numbers.
Model choice: logistic regression, a small neural network (MLP) and gradient boosting (LightGBM) are
compared by AUC on a held-out 25% of tune videos; the winner is refit on all of tune.
Thresholds: yellow - the lowest one with at most --yellow-budget false yellow+ events per minute on BDD-tune
(the rules' own false alarms included), red - at most --red-budget false red events per minute.
Writes models/risk.json (+ models/risk_lgbm.txt for LightGBM) and prints recall, lead time and false
alarms for rules only, model only and rules + model.
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


def collect(path):
    """Pooled feature vectors, times and rule levels for every analysis step (risk model off)."""
    from behavior import BehaviorEngine
    from risk import FeaturePool, RiskModel
    eng = BehaviorEngine()
    eng.risk = RiskModel(path="")
    pool = FeaturePool()
    ts, X, rules = [], [], []
    for t, res in steps(load(path), eng):
        ts.append(t)
        X.append(pool.vector(t, res.features))
        rules.append(res.level)
    return np.array(ts, float), np.array(X, np.float32), np.array(rules, np.int8)


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
    """Like the app: the probability must stay above a threshold RISK_CONFIRM seconds."""
    out = np.zeros(len(p), np.int8)
    for lv, thr in ((2, thr_y), (3, thr_r)):
        on = p >= thr
        if not on.any():
            continue
        start = on & ~np.concatenate([[False], on[:-1]])           # first step of every run above threshold
        run_start = ts[np.maximum.accumulate(np.where(start, np.arange(len(ts)), 0))]
        out[on & (ts - run_start >= C.RISK_CONFIRM)] = lv
    return out


def evaluate(data, labels, names, prob, thr_y, thr_r, use_rules, use_model):
    """{level: (recall, median lead, false/min BDD, false/min DADA before danger)} over `names`."""
    out = {}
    for lv in (2, 3):
        hits, leads, n, fa, fmin, bfa, bmin = 0, [], 0, 0, 0.0, 0, 0.0
        for name in names:
            ts, _, rules = data[name]
            lev = np.zeros(len(ts), np.int8)
            if use_rules:
                lev = np.maximum(lev, rules)
            if use_model:
                lev = np.maximum(lev, model_levels(ts, prob[name], thr_y, thr_r))
            ev = onsets(ts, lev, lv)
            if is_bdd(name):
                bfa += len(ev)
                bmin += (ts[-1] - ts[0]) / 60
                continue
            L = labels[name]
            start, end = float(L["start"]), float(L["end"])
            n += 1
            inside = [t for t in ev if start - C.EVAL_EARLY <= t <= end]
            fa += len([t for t in ev if t < start - C.EVAL_EARLY])
            fmin += max(start - C.EVAL_EARLY - ts[0], 0) / 60
            if inside:
                hits += 1
                leads.append(end - min(inside))
        out[lv] = (hits / max(n, 1), float(np.median(leads)) if leads else float("nan"),
                   bfa / max(bmin, 1e-6), fa / max(fmin, 1e-6), n, bmin)
    return out


def fit_models(Xtr, ytr, Xva, yva, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    import lightgbm as lgb
    res = {}
    lr = LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000).fit(Xtr, ytr)
    res["logreg"] = (lr, roc_auc_score(yva, lr.predict_proba(Xva)[:, 1]))
    w = np.where(ytr == 1, (ytr == 0).sum() / max((ytr == 1).sum(), 1), 1.0)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=512, learning_rate_init=1e-3,
                        max_iter=60, early_stopping=True, random_state=seed)
    # MLPClassifier has no sample weights: balance by oversampling positives
    pos = np.where(ytr == 1)[0]
    extra = np.random.default_rng(seed).choice(pos, max(0, int((ytr == 0).sum() - len(pos))), replace=True)
    idx = np.concatenate([np.arange(len(ytr)), extra])
    mlp.fit(Xtr[idx], ytr[idx])
    res["mlp"] = (mlp, roc_auc_score(yva, mlp.predict_proba(Xva)[:, 1]))
    gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                             verbose=-1, random_state=seed)
    gbm.fit(Xtr, ytr, sample_weight=w)
    res["lgbm"] = (gbm, roc_auc_score(yva, gbm.predict_proba(Xva)[:, 1]))
    return res


def main():
    ap = argparse.ArgumentParser(description="Train the risk model")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--yellow-budget", type=float, default=0.6, help="false yellow+ events per minute on BDD-tune")
    ap.add_argument("--red-budget", type=float, default=0.1, help="false red events per minute on BDD-tune")
    ap.add_argument("--kind", choices=["auto", "logreg", "mlp", "lgbm"], default="auto")
    ap.add_argument("--out", default=C.RISK_MODEL)
    ap.add_argument("--features-cache", default="cache/risk_features.npz", help="reuse collected features")
    ap.add_argument("--recollect", action="store_true")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    from risk import feature_names

    labels = {}
    for f in LABEL_FILES:
        labels.update(load_labels(f))
    names = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(args.cache, "*.npz")))]
    names = [n for n in names if n in labels or is_bdd(n)]
    data = {}
    if os.path.exists(args.features_cache) and not args.recollect:
        z = np.load(args.features_cache, allow_pickle=True)
        data = z["data"].item()
    todo = [n for n in names if n not in data]
    tic = time.time()
    for k, n in enumerate(todo, 1):
        data[n] = collect(os.path.join(args.cache, n + ".npz"))
        if k % 100 == 0:
            print(f"  features {k}/{len(todo)} ({time.time() - tic:.0f} s)", flush=True)
    if todo:
        np.savez_compressed(args.features_cache, data=np.array(data, dtype=object))
    data = {n: data[n] for n in names}
    groups = {s: [n for n in names if split_of(n, labels) == s] for s in ("tune", "test")}
    print(f"videos: crash/near-miss {sum(n in labels for n in names)} (DADA + reels), normal BDD "
          f"{sum(is_bdd(n) for n in names)}; tune {len(groups['tune'])}, test {len(groups['test'])}")

    def xy(group):
        Xs, ys = [], []
        for n in group:
            ts, X, _ = data[n]
            y = step_labels(n, ts, labels)
            Xs.append(X[y >= 0])
            ys.append(y[y >= 0])
        return np.concatenate(Xs), np.concatenate(ys)

    rnd = np.random.default_rng(0)
    tune = list(groups["tune"])
    rnd.shuffle(tune)
    cut = len(tune) // 4
    Xa, ya = xy(tune[cut:])
    Xv, yv = xy(tune[:cut])
    mean = Xa.mean(0)
    scale = np.where(Xa.std(0) > 1e-6, Xa.std(0), 1.0)
    norm = lambda X: (X - mean) / scale
    print(f"steps: fit {len(ya)} (danger {int(ya.sum())}), validation {len(yv)} (danger {int(yv.sum())})")
    cands = fit_models(norm(Xa), ya, norm(Xv), yv)
    for k, (_, auc) in cands.items():
        print(f"  {k:7s} validation AUC {auc:.3f}")
    kind = max(cands, key=lambda k: cands[k][1]) if args.kind == "auto" else args.kind
    print(f"chosen: {kind}")

    # refit the chosen kind on all of tune
    Xt, yt = xy(groups["tune"])
    mean = Xt.mean(0)
    scale = np.where(Xt.std(0) > 1e-6, Xt.std(0), 1.0)
    final = fit_models(norm(Xt), yt, norm(Xv), yv)[kind][0]
    prob = {n: final.predict_proba(norm(data[n][1]))[:, 1] if len(data[n][1]) else np.zeros(0) for n in names}
    Xte, yte = xy(groups["test"])
    print(f"test AUC (steps): {roc_auc_score(yte, final.predict_proba(norm(Xte))[:, 1]):.3f}")

    # thresholds on tune: lowest with the false-alarm budget on BDD-tune (rules included)
    def pick(lv, budget, other):
        for thr in np.linspace(0.3, 0.995, 140):
            ty, tr = (thr, other) if lv == 2 else (other, thr)
            if lv == 3 and thr < other:
                continue
            r = evaluate(data, labels, groups["tune"], prob, ty, tr, True, True)[lv]
            if r[2] <= budget:
                return float(thr)
        return 0.995
    thr_y = pick(2, args.yellow_budget, 0.999)
    thr_r = pick(3, args.red_budget, thr_y)
    print(f"thresholds: yellow {thr_y:.3f}, red {thr_r:.3f}")

    print(f"\n{'':6s} {'':18s} | {'yellow+ caught':>14s} {'lead':>6s} {'false/min BDD':>13s} {'DADA pre':>8s} |"
          f" {'red caught':>10s} {'lead':>6s} {'false/min BDD':>13s}")
    for split in ("tune", "test"):
        for title, ur, um in (("rules only", True, False), ("model only", False, True), ("rules + model", True, True)):
            r = evaluate(data, labels, groups[split], prob, thr_y, thr_r, ur, um)
            y_, r_ = r[2], r[3]
            print(f"{split:6s} {title:18s} | {y_[0]:14.0%} {y_[1]:5.1f}s {y_[2]:13.2f} {y_[3]:8.2f} |"
                  f" {r_[0]:10.0%} {r_[1]:5.1f}s {r_[2]:13.2f}")
    r = evaluate(data, labels, groups["test"], prob, thr_y, thr_r, True, True)
    print(f"(test: {r[2][4]} crash videos, {r[2][5]:.0f} min of BDD)")

    model = {"kind": kind, "features": feature_names(), "mean": mean.tolist(), "scale": scale.tolist(),
             "threshold": thr_y, "threshold_red": thr_r,
             "trained_on": {"crash_tune": sum(1 for n in groups["tune"] if n in labels),
                            "bdd_tune": sum(1 for n in groups["tune"] if is_bdd(n))},
             "test": {"yellow_recall": r[2][0], "yellow_lead_s": r[2][1], "yellow_false_per_min": r[2][2],
                      "red_recall": r[3][0], "red_lead_s": r[3][1], "red_false_per_min": r[3][2]}}
    if kind == "logreg":
        model["coef"], model["intercept"] = final.coef_[0].tolist(), float(final.intercept_[0])
    elif kind == "mlp":
        model["layers"] = [(W.tolist(), b.tolist()) for W, b in zip(final.coefs_, final.intercepts_)]
    else:
        mf = os.path.splitext(os.path.basename(args.out))[0] + "_lgbm.txt"
        final.booster_.save_model(os.path.join(os.path.dirname(args.out), mf))
        model["model_file"] = mf
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=1)
    print(f"\nmodel: {args.out}")
    if kind == "lgbm":
        imp = final.booster_.feature_importance("gain")
        names_f = feature_names()
        print("most important features:", ", ".join(names_f[i] for i in np.argsort(-imp)[:12]))


if __name__ == "__main__":
    main()
