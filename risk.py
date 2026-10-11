"""Learned risk model: from scene features (behavior.FEATURES) to the probability of a pre-crash moment.

The rules in behavior.py say WHAT the threat is (cut-in, pedestrian...). The model, trained on real
crashes from DADA-2000 and normal driving from BDD100K (tools/train_risk.py), says HOW MUCH the scene
looks like the seconds before a crash - and raises "DANGER" when no single rule fired, but the
combination (several cars close, someone moved sharply, a pedestrian at the lane edge) is typical of
crashes. Two thresholds: yellow (possible risk) and red (danger).
A second model - the threat scorer - gates the rules: every rule threat is scored from its own features
(behavior.THREAT_FEATURES: rule type, TTC, lateral position and speed, where it will be at impact, track
age and jitter...) and keeps its yellow / red level only if the score is at least gate_yellow / gate_red;
otherwise it is shown green. This removes most false alarms of single-camera geometry (parked cars,
people on the pavement, jumpy boxes) while keeping real cut-ins and crossings.

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
# features that mostly tell WHICH DATASET a clip is from (China crash clips vs US normal driving),
# not whether the scene is dangerous: the model must not see them (zeroed in training and at run time)
DROP = ("n_tracks", "veh_count_near", "vru_count_near", "lanes_ok", "ego_speed", "xwalk", "ego_yaw")


def drop_mask():
    """Boolean mask over feature_names(): True = column zeroed."""
    return np.array([any(n == d or n.startswith(d + "_") for d in DROP) for n in feature_names()])


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
        self.gate_yellow, self.gate_red = 0.0, 0.0       # rule threats need a score >= gate to stay yellow / red
        self.threat = None
        if not self.ok:
            return
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        if m.get("features") != feature_names():
            print(f"[risk] {path} was trained on other features - retrain with tools/train_risk.py; model off")
            self.ok = False
            return
        self.kind = m["kind"]
        self.drop = np.array(m.get("drop", [False] * len(m["features"])), bool)
        self.mean, self.scale = np.array(m["mean"]), np.array(m["scale"])
        self.threshold = m["threshold"]
        self.threshold_red = m.get("threshold_red", 1.0)
        self.gate_yellow, self.gate_red = m.get("gate_yellow", 0.0), m.get("gate_red", 0.0)
        t = m.get("threat")
        if t:
            from behavior import THREAT_FEATURES
            if t["features"] != THREAT_FEATURES:
                print(f"[risk] threat scorer in {path} was trained on other features - retrain; scorer off")
            else:
                import lightgbm
                self.threat = lightgbm.Booster(model_file=os.path.join(os.path.dirname(path), t["model_file"]))
                self.t_mean, self.t_scale = np.array(t["mean"]), np.array(t["scale"])
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
        X = np.array(np.atleast_2d(X), np.float64)
        X[:, self.drop] = 0.0
        X = (X - self.mean) / self.scale
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

    def threat_score(self, feat):
        """Probability that this rule threat is like the ones before real crashes (None - no scorer)."""
        if self.threat is None:
            return None
        x = (np.array(feat, np.float64)[None] - self.t_mean) / self.t_scale
        return float(self.threat.predict(x)[0])

    def __call__(self, t, features):
        """Probability of a dangerous moment 0..1 (None - no model)."""
        if not self.ok:
            return None
        return float(self.predict(self.pool.vector(t, features))[0])
