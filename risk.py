"""Learned risk model: from scene features (behavior.FEATURES) to the probability of a pre-crash moment.

The rules in behavior.py say WHAT the threat is (cut-in, pedestrian...). The model, trained on real
crashes from DADA-2000 and normal driving from BDD100K (tools/train_risk.py), says HOW MUCH the scene
looks like the seconds before a crash - and raises "DANGER" when no single rule fired, but the
combination (several cars close, someone moved sharply, a pedestrian at the lane edge) is typical of
crashes. Two thresholds: yellow (possible risk) and red (danger).

Input vector = current features + max and mean over the last POOLS seconds (so short detections are
not lost and the trend is visible). Model kinds stored in models/risk.json:
    logreg - logistic regression (weights in JSON);
    mlp    - small neural network (weights in JSON, numpy only - runs anywhere);
    lgbm   - gradient boosting (LightGBM text model next to the JSON; needs the lightgbm package).
Without the file the model is off.
"""
import json
import os
from collections import deque

import numpy as np

import config as C
from behavior import FEATURES

POOLS = (0.5, 2.0)


def feature_names():
    names = list(FEATURES)
    for p in POOLS:
        names += [f"{k}_max{p:g}" for k in FEATURES]
    names += [f"{k}_mean{POOLS[-1]:g}" for k in FEATURES]
    return names


class FeaturePool:
    """Current features + max over each pool window + mean over the longest one."""

    def __init__(self):
        self.hist = deque()

    def vector(self, t, features):
        cur = np.array([features.get(k, 0.0) for k in FEATURES], np.float32)
        self.hist.append((t, cur))
        while t - self.hist[0][0] > max(POOLS):
            self.hist.popleft()
        parts = [cur]
        for p in POOLS:
            parts.append(np.max([v for tt, v in self.hist if t - tt <= p], axis=0))
        parts.append(np.mean([v for _, v in self.hist], axis=0))
        return np.concatenate(parts)


class RiskModel:
    def __init__(self, path=C.RISK_MODEL):
        self.ok = bool(path) and os.path.exists(path)
        self.pool = FeaturePool()
        self.threshold, self.threshold_red = 1.0, 1.0
        if not self.ok:
            return
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        if m.get("features") != feature_names():
            print(f"[risk] {path} was trained on other features - retrain with tools/train_risk.py; model off")
            self.ok = False
            return
        self.kind = m["kind"]
        self.mean, self.scale = np.array(m["mean"]), np.array(m["scale"])
        self.threshold = m["threshold"]
        self.threshold_red = m.get("threshold_red", 1.0)
        if self.kind == "logreg":
            self.coef, self.bias = np.array(m["coef"]), m["intercept"]
        elif self.kind == "mlp":
            self.layers = [(np.array(W), np.array(b)) for W, b in m["layers"]]
        elif self.kind == "lgbm":
            import lightgbm
            self.booster = lightgbm.Booster(model_file=os.path.join(os.path.dirname(path), m["model_file"]))
        else:
            raise ValueError(f"unknown risk model kind {self.kind}")

    def predict(self, X):
        """Probabilities for a batch of feature vectors (N x len(feature_names()))."""
        X = (np.atleast_2d(X) - self.mean) / self.scale
        if self.kind == "logreg":
            z = X @ self.coef + self.bias
        elif self.kind == "mlp":
            a = X
            for i, (W, b) in enumerate(self.layers):
                a = a @ W + b
                if i < len(self.layers) - 1:
                    a = np.maximum(a, 0)
            z = a[:, 0]
        else:
            return self.booster.predict(X)
        return 1 / (1 + np.exp(-z))

    def __call__(self, t, features):
        """Probability of a dangerous moment 0..1 (None - no model)."""
        if not self.ok:
            return None
        return float(self.predict(self.pool.vector(t, features))[0])
