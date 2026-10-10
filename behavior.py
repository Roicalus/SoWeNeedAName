"""Анализ дорожной ситуации: из детекций кадра получаем предупреждения.

Модуль не знает ни про окно, ни про звук, поэтому его используют и main.py, и evaluate.py.
"""
from __future__ import annotations

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
    crashed: list = field(default_factory=list)    # машины с признаками аварии (любые, не только впереди)
    # для отладки (debug_video.py): статус каждой рамки, сырые условия до подтверждения, метрики виляния
    tracks: list = field(default_factory=list)   # [(Detection, статус)]
    raw: set = field(default_factory=set)
    weave: tuple | None = None                   # (амплитуда в ширинах, смен направления)

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
        self.boxes = defaultdict(deque)     # id трека -> [(t, x1, y1, x2, y2)] в долях кадра
        self.crash_at = {}                  # id трека -> когда последний раз были признаки аварии

    def update(self, t, W, H, detections, frame=None):
        """t — время видео в секундах; frame — исходный кадр (нужен только для стоп-сигналов)."""
        res = FrameResult()
        raw = set()
        lead = None

        for d in detections:
            x1, y1, x2, y2 = d.box
            cx, by = (x1 + x2) / 2 / W, y2 / H
            if (y1 + y2) / 2 / H > C.HOOD_Y:
                res.tracks.append((d, "hood"))
                continue                     # капот своей машины
            if d.cls in C.VEHICLE_CLASSES:
                h = self.hist[d.tid]
                h.append((t, (x2 - x1) / W, cx))
                while t - h[0][0] > C.HISTORY_SEC:
                    h.popleft()
                self.last_seen[d.tid] = t
                b = self.boxes[d.tid]
                b.append((t, x1 / W, y1 / H, x2 / W, y2 / H))
                while t - b[0][0] > C.CRASH_WINDOW:
                    b.popleft()
                # машина впереди — ближайшая (самый низкий край рамки) на нашей траектории;
                # рамка намного шире высоты — машина/трамвай видны сбоку, это не «машина впереди»
                rear_view = (x2 - x1) <= C.LEAD_MAX_ASPECT * (y2 - y1)
                # рамка обрезана боковым краем кадра — её ширина (и TTC) недостоверны
                inside = x1 > C.EDGE_MARGIN * W and x2 < (1 - C.EDGE_MARGIN) * W
                offset = abs(cx - C.CORRIDOR_CENTER_X) / max((x2 - x1) / W, 1e-6)
                status = ("lane" if not in_corridor(cx, by) else f"off {offset:.1f}w" if not on_path(cx, (x2 - x1) / W)
                          else "side" if not rear_view else "edge" if not inside else "cand")
                res.tracks.append((d, status))
                if status == "cand" and (lead is None or y2 > lead.box[3]):
                    lead = d
            elif d.cls == C.PERSON_CLASS:
                # боковое смещение в метрах по росту человека — не зависит от расстояния
                offset_m = abs((x1 + x2) / 2 - C.CORRIDOR_CENTER_X * W) / max(y2 - y1, 1) * C.PERSON_HEIGHT_M
                status = ("far" if by < C.PED_MIN_Y else "lane" if not in_corridor(cx, by)
                          else f"off {offset_m:.1f}m" if offset_m > C.PED_MAX_OFFSET_M else "PED")
                res.tracks.append((d, status))
                if status == "PED":
                    raw.add("ped")
                    res.peds.append(d)

        self._forget(t)

        # авария у любой машины в кадре: держим тревогу CRASH_HOLD секунд, даже если трек потерян
        for tid in self._crashes(t):
            self.crash_at[tid] = t
        self.crash_at = {k: v for k, v in self.crash_at.items() if t - v <= C.CRASH_HOLD}
        if self.crash_at:
            raw.add("crash")
            res.crashed = [d for d in detections if d.tid in self.crash_at]

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
            weaving, res.weave = self._weaving(h, t, stats=True)
            if weaving:
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

        res.raw = raw
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
    def _weaving(self, h, t, stats=False):
        """stats=True — вернуть ещё (амплитуда, смен направления) для отладки."""
        out = (lambda ok, st=None: (ok, st)) if stats else (lambda ok, st=None: ok)
        pts = [p for p in h if t - p[0] <= C.WEAVE_WINDOW]
        if len(pts) < 10 or pts[-1][0] - pts[0][0] < C.WEAVE_MIN_SPAN:
            return out(False)
        ts, ws, xs = np.array(pts).T
        width = ws.mean()
        if width < C.WEAVE_MIN_WIDTH:
            return out(False)
        # 1) амплитуда: убираем общий тренд (поворот дороги), смотрим размах колебаний
        resid = xs - np.polyval(np.polyfit(ts, xs, 1), ts)
        amp = resid.std() / width
        # 2) характер: виляние — это многократная смена направления боковой скорости;
        #    у перестроения направление одно, у обгона с возвратом — одна смена
        fps = (len(ts) - 1) / (ts[-1] - ts[0])
        k = max(3, int(0.3 * fps))
        smooth = np.convolve(xs, np.ones(k) / k, mode="valid")
        speed = np.diff(smooth) * fps / width                  # ширин машины в секунду
        moving = np.sign(speed[np.abs(speed) > C.WEAVE_MIN_SPEED])
        rev = int(np.count_nonzero(np.diff(moving)))
        return out(amp >= C.WEAVE_REL_STD and rev >= C.WEAVE_MIN_REVERSALS, (amp, rev))

    # ---------- авария у любой машины ----------
    def _crashes(self, t):
        """Машина «взлетает»: рамка резко уходит вверх и становится выше относительно ширины,
        а ширина почти не меняется (машина не удаляется). У обычной машины так не бывает:
        при удалении вместе с подъёмом низа рамки сужается и сама рамка.
        Качание своей машины (кочки) сдвигает все рамки одинаково — вычитаем медианный сдвиг."""
        pairs = {}
        for tid, b in self.boxes.items():
            if self.last_seen.get(tid) != t or len(b) < 2 or t - b[0][0] < C.CRASH_WINDOW / 2:
                continue
            pairs[tid] = (b[0], b[-1])
        if not pairs:
            return []
        shifts = [((n[2] + n[4]) - (o[2] + o[4])) / 2 for o, n in pairs.values()]
        ego = float(np.median(shifts)) if len(shifts) >= 3 else 0.0
        out = []
        for tid, (o, n) in pairs.items():
            dt = n[0] - o[0]
            w, h, w0, h0 = n[3] - n[1], n[4] - n[2], o[3] - o[1], o[4] - o[2]
            cut = min(o[1], n[1]) < C.EDGE_MARGIN or max(o[3], n[3]) > 1 - C.EDGE_MARGIN \
                or (n[2] + n[4]) / 2 > C.HOOD_Y
            if cut or w < C.CRASH_MIN_WIDTH or h <= 0 or h0 <= 0:
                continue
            rise = -(((n[2] + n[4]) - (o[2] + o[4])) / 2 - ego) / h / dt       # высот рамки в секунду
            taller = math.log((h / w) / (h0 / w0)) / dt
            scale = math.log(w / w0) / dt
            if rise >= C.CRASH_RISE and taller >= C.CRASH_ASPECT and abs(scale) < C.CRASH_MAX_SCALE:
                out.append(tid)
        return out

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
            self.boxes.pop(tid, None)
