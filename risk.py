"""Обученная модель риска: по признакам ситуации (behavior.FEATURES) — вероятность опасного момента.

Правила в behavior.py говорят, ЧТО за угроза (подрезание, пешеход…). Модель, обученная на реальных
авариях DADA-2000 и обычной езде BDD100K (tools/train_risk.py), говорит, НАСКОЛЬКО ситуация похожа
на предаварийную, — и даёт общий сигнал «Опасно!», когда ни одно правило не сработало, но сочетание
признаков (несколько машин рядом, кто-то резко сместился, пешеход у полосы) типично для аварий.

Модель — логистическая регрессия на признаках, взятых максимумом за последние POOL_SEC секунд:
models/risk.json (средние, масштабы, веса, порог). Без файла модель выключена.
"""
import json
import os
from collections import deque

import numpy as np

import config as C
from behavior import FEATURES


class RiskModel:
    def __init__(self, path=C.RISK_MODEL):
        self.ok = os.path.exists(path)
        self.hist = deque()
        if self.ok:
            with open(path, encoding="utf-8") as f:
                m = json.load(f)
            self.names = m["features"]
            self.mean, self.scale = np.array(m["mean"]), np.array(m["scale"])
            self.coef, self.bias = np.array(m["coef"]), m["intercept"]
            self.threshold = m["threshold"]
            self.pool = m.get("pool_sec", 0.5)

    def vector(self, t, features):
        """Признаки кадра + максимум каждого за последние pool секунд (мигающие детекции не теряются)."""
        cur = np.array([features.get(k, 0.0) for k in FEATURES], np.float32)
        self.hist.append((t, cur))
        while t - self.hist[0][0] > self.pool:
            self.hist.popleft()
        pooled = np.max([v for _, v in self.hist], axis=0)
        return np.concatenate([cur, pooled])

    def __call__(self, t, features):
        """Вероятность опасного момента 0..1 (None — модели нет)."""
        if not self.ok:
            return None
        x = (self.vector(t, features) - self.mean) / self.scale
        return float(1 / (1 + np.exp(-(x @ self.coef + self.bias))))


def feature_names():
    return FEATURES + [f"{k}_max" for k in FEATURES]
