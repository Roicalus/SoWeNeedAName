"""Обучение модели риска (risk.py) на кэше роликов: DADA-2000 — аварии, BDD100K — обычная езда.

    python tools/train_risk.py
    python tools/train_risk.py --fp-budget 0.5        # допустимо ложных «Опасно!» в минуту на BDD

Разметка кадров:
  1 — DADA, от начала опасной ситуации до аварии;
  0 — DADA раньше чем за 2 с до начала опасности, и весь BDD;
  остальное (переход) в обучение не идёт.
Обучаем на половине «tune» (DADA split=tune, половина BDD), проверяем на «test» — честная оценка.
Порог подбирается так, чтобы на BDD-tune было не больше --fp-budget срабатываний в минуту.
Пишет models/risk.json и печатает: AUC, поймано аварий, за сколько секунд до аварии, ложные/мин.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from tools.replay import load  # noqa: E402


def collect(path):
    """Признаки (текущие + максимум за pool секунд) и время на каждом шаге анализа + сработавшие правила."""
    from behavior import BehaviorEngine, Detection
    from risk import RiskModel
    ts, by, lanes, W, H = load(path)
    eng, pool = BehaviorEngine(), RiskModel(path="")
    pool.pool = C.RISK_POOL_SEC
    X, rules = [], []
    for k, t in enumerate(ts):
        dets = [Detection(tuple(r[3:7]), int(r[2]), int(r[1])) for r in by.get(k, [])]
        res = eng.update(float(t), W, H, dets, None, lanes[k] if lanes is not None else None)
        X.append(pool.vector(float(t), res.features))
        rules.append(res.level)
    return np.array(ts, float), np.array(X, np.float32), np.array(rules)


def is_bdd(name):
    return len(name) == 17 and name[8] == "-"


def bdd_split(name):
    return "test" if int(hashlib.md5(name.encode()).hexdigest(), 16) % 2 else "tune"


def alert_events(ts, on):
    """Начала срабатываний (новое, если перерыв больше EVENT_GAP)."""
    ev, last = [], -1e9
    for t, o in zip(ts, on):
        if o:
            if t - last > C.EVENT_GAP:
                ev.append(t)
            last = t
    return ev


def confirm(ts, p, thr):
    """Как в приложении: сигнал засчитывается, если держится RISK_CONFIRM секунд."""
    on, since = np.zeros(len(p), bool), None
    for i, (t, v) in enumerate(zip(ts, p)):
        if v >= thr:
            since = t if since is None else since
            on[i] = t - since >= C.RISK_CONFIRM
        else:
            since = None
    return on


def evaluate(data, labels, split, prob, thr, use_rules=False):
    hits, leads, n, fa, fmin = 0, [], 0, 0, 0.0
    bfa, bmin = 0, 0.0
    for name, (ts, X, rules) in data.items():
        on = confirm(ts, prob[name], thr)
        if use_rules:
            on = on | (rules >= 2)
        if is_bdd(name):
            if bdd_split(name) != split:
                continue
            bfa += len(alert_events(ts, on))
            bmin += (ts[-1] - ts[0]) / 60
            continue
        L = labels.get(name)
        if L is None or L["split"] != split:
            continue
        n += 1
        start, end = float(L["start"]), float(L["end"])
        ev = alert_events(ts, on)
        inside = [t for t in ev if start - C.EVAL_EARLY <= t <= end]
        fa += len([t for t in ev if t < start - C.EVAL_EARLY])
        fmin += max(start - C.EVAL_EARLY - ts[0], 0) / 60
        if inside:
            hits += 1
            leads.append(end - min(inside))
    return {"recall": hits / max(n, 1), "n": n, "lead": float(np.median(leads)) if leads else None,
            "fp_dada": fa / max(fmin, 1e-6), "fp_bdd": bfa / max(bmin, 1e-6), "bdd_min": bmin}


def main():
    ap = argparse.ArgumentParser(description="Обучение модели риска")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--labels", default="data/dada/labels.csv")
    ap.add_argument("--fp-budget", type=float, default=0.5, help="ложных срабатываний в минуту на BDD-tune")
    ap.add_argument("--out", default=C.RISK_MODEL)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from risk import feature_names

    with open(args.labels, encoding="utf-8") as f:
        labels = {os.path.splitext(r["file"])[0]: r for r in csv.DictReader(f)}
    data = {}
    for path in sorted(glob.glob(os.path.join(args.cache, "*.npz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in labels or is_bdd(name):
            data[name] = collect(path)
    print(f"Роликов: DADA {sum(n in labels for n in data)}, BDD {sum(is_bdd(n) for n in data)}")

    def xy(split):
        Xs, ys = [], []
        for name, (ts, X, _) in data.items():
            if is_bdd(name):
                if bdd_split(name) == split:
                    Xs.append(X)
                    ys.append(np.zeros(len(X)))
                continue
            L = labels[name]
            if L["split"] != split:
                continue
            start, end = float(L["start"]), float(L["end"])
            y = np.full(len(X), -1.0)
            y[(ts >= start) & (ts <= end)] = 1
            y[ts < start - 2] = 0
            Xs.append(X[y >= 0])
            ys.append(y[y >= 0])
        return np.concatenate(Xs), np.concatenate(ys)

    Xtr, ytr = xy("tune")
    Xte, yte = xy("test")
    print(f"Шагов анализа: обучение {len(ytr)} (опасных {int(ytr.sum())}), проверка {len(yte)} (опасных {int(yte.sum())})")
    scaler = StandardScaler().fit(Xtr)
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000)
    clf.fit((Xtr - scaler.mean_) / scale, ytr)
    p_te = clf.predict_proba((Xte - scaler.mean_) / scale)[:, 1]
    print(f"AUC на проверке: {roc_auc_score(yte, p_te):.3f}")

    prob = {name: clf.predict_proba((X - scaler.mean_) / scale)[:, 1] for name, (ts, X, _) in data.items()}
    # порог: самый низкий, при котором на BDD-tune ложных не больше бюджета
    best = None
    for thr in np.linspace(0.3, 0.99, 70):
        r = evaluate(data, labels, "tune", prob, thr)
        if r["fp_bdd"] <= args.fp_budget:
            best = thr
            break
    best = best if best is not None else 0.99
    print(f"Порог {best:.2f} (ложных на BDD-tune ≤ {args.fp_budget}/мин)\n")

    names = feature_names()
    order = np.argsort(-np.abs(clf.coef_[0]))
    print("Самые важные признаки (вес после нормировки):")
    for i in order[:10]:
        print(f"  {names[i]:22s} {clf.coef_[0][i]:+.2f}")

    print(f"\n{'':28s} {'поймано':>8s} {'до аварии':>10s} {'ложные/мин BDD':>15s} {'ложные/мин DADA до опасности':>29s}")
    for split in ("tune", "test"):
        for title, use_rules, p in (("только правила", True, {n: np.zeros(len(v[0])) for n, v in data.items()}),
                                     ("только модель", False, prob), ("правила + модель", True, prob)):
            r = evaluate(data, labels, split, p, best if p is prob else 2.0, use_rules)
            lead = f"{r['lead']:.1f} с" if r["lead"] is not None else "-"
            print(f"{split:5s} {title:22s} {r['recall']:8.0%} {lead:>10s} {r['fp_bdd']:15.2f} {r['fp_dada']:29.2f}")

    model = {"features": names, "mean": scaler.mean_.tolist(), "scale": scale.tolist(),
             "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]), "threshold": float(best),
             "pool_sec": C.RISK_POOL_SEC,
             "trained_on": {"dada_tune": sum(1 for n in data if n in labels and labels[n]["split"] == "tune"),
                            "bdd_tune": sum(1 for n in data if is_bdd(n) and bdd_split(n) == "tune")}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=1)
    print(f"\nМодель: {args.out}")


if __name__ == "__main__":
    main()
