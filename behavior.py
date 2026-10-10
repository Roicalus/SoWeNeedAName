"""Анализ дорожной ситуации: из детекций кадра получаем предупреждения.

Модуль не знает ни про окно, ни про звук, поэтому его используют и main.py, и evaluate.py.
"""
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

import config as C


@dataclass
class Detection:
    box: tuple   # x1, y1, x2, y2 в пикселях
    cls: int
    tid: int     # id трека


@dataclass
class Alert:
    kind: str
    level: int
    screen: str
    voice: str


@dataclass
class FrameResult:
    alerts: list = field(default_factory=list)
    lead: Detection | None = None
    ttc: float | None = None
    distance: float | None = None   # примерная дистанция до машины впереди, м
    oncoming: bool = False          # машина впереди похожа на встречную
    peds: list = field(default_factory=list)

    @property
    def top(self):
        return max(self.alerts, key=lambda a: a.level, default=None)

    @property
    def level(self):
        return self.top.level if self.top else 0


# ---------- своя полоса ----------
def _half_width(y):
    k = (y - C.CORRIDOR_TOP_Y) / (1 - C.CORRIDOR_TOP_Y)
    return C.CORRIDOR_TOP_HALF + k * (C.CORRIDOR_BOTTOM_HALF - C.CORRIDOR_TOP_HALF)


def in_corridor(x, y):
    """x, y в долях кадра; y — нижний край рамки."""
    return y >= C.CORRIDOR_TOP_Y and abs(x - C.CORRIDOR_CENTER_X) <= _half_width(y)


def on_path(cx, width):
    """Боковое смещение машины в её же ширинах: (cx - центр) / ширина = X / реальная_ширина.

    Не зависит от расстояния. Машина на нашей траектории — смещение < ~0.7,
    припаркованная у обочины — 1.5 и больше (она быстро растёт в кадре, но мы её объезжаем).
    """
    return abs(cx - C.CORRIDOR_CENTER_X) / max(width, 1e-6) <= C.LEAD_MAX_OFFSET


def corridor_polygon(W, H):
    cx, top = C.CORRIDOR_CENTER_X, C.CORRIDOR_TOP_Y
    pts = [(cx - C.CORRIDOR_TOP_HALF, top), (cx + C.CORRIDOR_TOP_HALF, top),
           (cx + C.CORRIDOR_BOTTOM_HALF, 1.0), (cx - C.CORRIDOR_BOTTOM_HALF, 1.0)]
    return np.array([(int(x * W), int(y * H)) for x, y in pts], np.int32)


class BehaviorEngine:
    def __init__(self):
        self.hist = defaultdict(deque)      # id трека -> [(t, ширина, центр_x)] в долях кадра
        self.last_seen = {}
        self.since = {}                     # с какого момента непрерывно держится условие
        self.red_base = {}                  # обычная доля красного у машины (для стоп-сигналов)

    def update(self, t, W, H, detections, frame=None):
        """t — время видео в секундах; frame — исходный кадр (нужен только для стоп-сигналов)."""
        res = FrameResult()
        raw = set()
        lead = None

        for d in detections:
            x1, y1, x2, y2 = d.box
            cx, by = (x1 + x2) / 2 / W, y2 / H
            if (y1 + y2) / 2 / H > C.HOOD_Y:
                continue                     # капот своей машины
            if d.cls in C.VEHICLE_CLASSES:
                h = self.hist[d.tid]
                h.append((t, (x2 - x1) / W, cx))
                while t - h[0][0] > C.HISTORY_SEC:
                    h.popleft()
                self.last_seen[d.tid] = t
                # машина впереди — ближайшая (самый низкий край рамки) на нашей траектории;
                # рамка намного шире высоты — машина/трамвай видны сбоку, это не «машина впереди»
                rear_view = (x2 - x1) <= C.LEAD_MAX_ASPECT * (y2 - y1)
                # рамка обрезана боковым краем кадра — её ширина (и TTC) недостоверны
                inside = x1 > C.EDGE_MARGIN * W and x2 < (1 - C.EDGE_MARGIN) * W
                if in_corridor(cx, by) and on_path(cx, (x2 - x1) / W) and rear_view and inside \
                        and (lead is None or y2 > lead.box[3]):
                    lead = d
            elif d.cls == C.PERSON_CLASS and by >= C.PED_MIN_Y and in_corridor(cx, by):
                # боковое смещение в метрах по росту человека — не зависит от расстояния
                offset_m = abs((x1 + x2) / 2 - C.CORRIDOR_CENTER_X * W) / max(y2 - y1, 1) * C.PERSON_HEIGHT_M
                if offset_m <= C.PED_MAX_OFFSET_M:
                    raw.add("ped")
                    res.peds.append(d)

        self._forget(t)

        if lead:
            res.lead = lead
            h = self.hist[lead.tid]
            w = h[-1][1]
            res.distance = C.VEHICLE_WIDTH_M / (2 * math.tan(math.radians(C.CAMERA_HFOV_DEG) / 2) * w)
            ttc = self._ttc(h, t)
            # встречная машина сближается с суммой скоростей — для машины в нашей полосе это нереально
            if ttc is not None and res.distance / ttc > C.LEAD_MAX_CLOSING:
                res.oncoming, ttc = True, None
            res.ttc = ttc
            if res.ttc is not None and res.ttc < C.TTC_WARNING:
                raw.add("closing")
                if res.ttc < C.TTC_CRITICAL and w >= C.CRITICAL_MIN_WIDTH:
                    raw.add("ttc")
            if w >= C.CLOSE_WIDTH:
                raw.add("close")
            if self._weaving(h, t):
                raw.add("weave")
            if C.BRAKE_LIGHTS and frame is not None and self._brake_lights(frame, lead):
                raw.add("brakelight")

        for kind in C.ALERTS:
            if kind in raw:
                self.since.setdefault(kind, t)
            else:
                self.since.pop(kind, None)
        active = {k for k in raw if t - self.since[k] >= C.CONFIRM_SEC[k]}
        if "ttc" in active:
            active.discard("closing")   # критическое заменяет обычное сближение

        res.alerts = [Alert(k, *C.ALERTS[k]) for k in active]
        return res

    # ---------- время до столкновения ----------
    def _ttc(self, h, t):
        """TTC по скорости роста рамки: d ln(ширина)/dt = 1/TTC. Расстояние знать не нужно."""
        pts = [(pt, w) for pt, w, _ in h if t - pt <= C.TTC_WINDOW]
        if len(pts) < 4 or pts[-1][0] - pts[0][0] < C.TTC_MIN_SPAN or pts[-1][1] < C.TTC_MIN_WIDTH:
            return None
        ts, ws = np.array(pts).T
        rate = np.polyfit(ts, np.log(ws), 1)[0]
        if rate <= C.TTC_MIN_RATE:
            return None
        # прямая даёт TTC для середины окна; к текущему моменту он меньше на полокна
        return max(1.0 / rate - (ts[-1] - ts.mean()), 0.1)

    # ---------- виляние ----------
    def _weaving(self, h, t):
        pts = [p for p in h if t - p[0] <= C.WEAVE_WINDOW]
        if len(pts) < 10 or pts[-1][0] - pts[0][0] < C.WEAVE_MIN_SPAN:
            return False
        ts, ws, xs = np.array(pts).T
        width = ws.mean()
        if width < C.WEAVE_MIN_WIDTH:
            return False
        # 1) амплитуда: убираем общий тренд (поворот дороги), смотрим размах колебаний
        resid = xs - np.polyval(np.polyfit(ts, xs, 1), ts)
        if resid.std() / width < C.WEAVE_REL_STD:
            return False
        # 2) характер: виляние — это многократная смена направления боковой скорости;
        #    у перестроения направление одно, у обгона с возвратом — одна смена
        fps = (len(ts) - 1) / (ts[-1] - ts[0])
        k = max(3, int(0.3 * fps))
        smooth = np.convolve(xs, np.ones(k) / k, mode="valid")
        speed = np.diff(smooth) * fps / width                  # ширин машины в секунду
        moving = np.sign(speed[np.abs(speed) > C.WEAVE_MIN_SPEED])
        return np.count_nonzero(np.diff(moving)) >= C.WEAVE_MIN_REVERSALS

    # ---------- стоп-сигналы (экспериментально) ----------
    def _brake_lights(self, frame, d):
        x1, y1, x2, y2 = (max(int(v), 0) for v in d.box)
        roi = frame[(y1 + y2) // 2:y2, x1:x2]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        ratio = (((hue < 10) | (hue > 170)) & (sat > 120) & (val > 150)).mean()
        base = self.red_base.get(d.tid, ratio)
        on = ratio > C.BRAKE_RED_MIN and ratio > base * C.BRAKE_RED_JUMP
        if not on:
            self.red_base[d.tid] = 0.9 * base + 0.1 * ratio
        return on

    def _forget(self, t):
        for tid in [k for k, seen in self.last_seen.items() if t - seen > C.TRACK_TTL]:
            del self.last_seen[tid]
            self.hist.pop(tid, None)
            self.red_base.pop(tid, None)
