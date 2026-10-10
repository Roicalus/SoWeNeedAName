"""Анализ дорожной ситуации: из детекций и разметки кадра получаем угрозы и предупреждения.

Модуль не знает ни про окно, ни про звук, поэтому его используют main.py, evaluate.py,
debug_video.py и tools/replay.py.

Каждая машина, пешеход, велосипед и мотоцикл в кадре получает:
  u   — боковое положение в ширинах полосы: 0 — центр нашей полосы, ±1 — центр соседней.
        По разметке (lanes.py), если она видна; иначе по геометрии рамки (смещение в ширинах машины
        или в росте человека — не зависит от расстояния);
  du  — скорость по u (полос в секунду), с поправкой на поворот своей машины;
  TTC — время до столкновения по росту рамки, дистанция — по ширине (машина) или росту (человек).
Из этого — угрозы: сближение в нашей полосе, подрезание, пересечение пути, встречная по нашей полосе,
пешеход/двухколёсный на пути, авария (рамка «взлетает»), виляние машины впереди.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

import config as C
from lanes import LaneModel

VRU_CLASSES = {C.PERSON_CLASS, 1, 3}          # человек, велосипед, мотоцикл — «уязвимые участники»
# признаки ситуации для модели риска (risk.py); «near» = 1 / дистанция, «toward» — скорость к нашей полосе
FEATURES = ["veh_in_near", "veh_in_ttc", "veh_side_toward", "veh_side_near", "veh_side_ttc", "veh_count_near",
            "veh_lat_speed", "vru_in_near", "vru_side_toward", "vru_side_near", "crash", "oncoming", "weave",
            "lanes_ok"]


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
class Threat:
    det: Detection
    kind: str           # тип угрозы (ключ C.ALERTS)
    level: int          # 1–3
    u: float | None     # боковое положение, полос
    du: float | None    # боковая скорость, полос/с
    ttc: float | None
    dist: float | None
    path: tuple | None = None   # (x, y) куда машина будет через C.PREDICT_SEC — для стрелки на экране


@dataclass
class FrameResult:
    alerts: list = field(default_factory=list)
    lead: Detection | None = None
    ttc: float | None = None
    distance: float | None = None   # примерная дистанция до машины впереди, м
    oncoming: bool = False          # машина впереди похожа на встречную
    peds: list = field(default_factory=list)
    crashed: list = field(default_factory=list)    # машины с признаками аварии (любые, не только впереди)
    threats: list = field(default_factory=list)    # Threat по всем участникам
    lanes: LaneModel | None = None                 # разметка (для рисования), если видна
    risk: float = 0.0                              # общий уровень опасности 0..1
    features: dict = field(default_factory=dict)   # признаки ситуации для модели риска (risk.py)
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


# ---------- своя полоса без разметки: трапеция перед машиной ----------
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


def focal():
    """Фокусное расстояние в долях ширины кадра (по углу обзора камеры)."""
    return 0.5 / math.tan(math.radians(C.CAMERA_HFOV_DEG) / 2)


class BehaviorEngine:
    def __init__(self):
        # id трека -> [(t, x1, y1, x2, y2, u)]: рамка в долях кадра и боковое положение в полосах
        self.hist = defaultdict(deque)
        self.cls = {}
        self.last_seen = {}
        self.since = {}                     # тип -> когда условие появилось
        self.last_raw = {}                  # тип -> когда условие было последний раз
        self.red_base = {}                  # обычная доля красного у машины (для стоп-сигналов)
        self.crash_at = {}                  # id трека -> когда последний раз были признаки аварии
        self.lanes = LaneModel()
        from risk import RiskModel
        self.risk = RiskModel()
        self.yaw = deque()                  # (t, медианный сдвиг рамок по x) — поворот своей машины

    # ================================================================ главный шаг
    def update(self, t, W, H, detections, frame=None, lanes=None):
        """t — время видео, с; frame — кадр (для стоп-сигналов); lanes — линии от LaneDetector.detect."""
        res = FrameResult()
        self.lanes.update(t, lanes)
        lanes_ok = self.lanes.valid()
        res.lanes = self.lanes if lanes_ok else None
        aspect = W / H

        live = []
        for d in detections:
            x1, y1, x2, y2 = d.box
            if (y1 + y2) / 2 / H > C.HOOD_Y:
                res.tracks.append((d, "hood"))
                continue                     # капот своей машины
            if d.cls not in C.VEHICLE_CLASSES and d.cls not in VRU_CLASSES:
                continue
            box = (x1 / W, y1 / H, x2 / W, y2 / H)
            u = self._lateral(d.cls, box, aspect, lanes_ok)
            h = self.hist[d.tid]
            h.append((t, *box, u))
            while t - h[0][0] > C.HISTORY_SEC:
                h.popleft()
            self.cls[d.tid] = d.cls
            self.last_seen[d.tid] = t
            live.append(d)
        self._forget(t)
        yaw = self._yaw_rate(t, live)

        raw, threats, lead = set(), [], None
        F = dict.fromkeys(FEATURES, 0.0)
        F["lanes_ok"] = float(lanes_ok)
        for d in live:
            h = self.hist[d.tid]
            _, x1, y1, x2, y2, u = h[-1]
            w, by = x2 - x1, y2
            vehicle = d.cls in C.VEHICLE_CLASSES
            side_view = (x2 - x1) * aspect > C.CROSS_MIN_ASPECT * (y2 - y1)     # машина видна сбоку
            dist = self._distance(d.cls, x2 - x1, y2 - y1, aspect, side_view)
            # TTC по ширине (точнее для машины сзади) или по высоте (не зависит от поворота — для машин
            # сбоку и людей); у человека — чтобы понять, приближаемся ли мы к нему
            ttc = self._ttc(h, t, by_height=side_view or not vehicle)
            du = self._lat_speed(h, t, yaw, lanes_ok, vehicle, aspect)
            inside = x1 > C.EDGE_MARGIN and x2 < 1 - C.EDGE_MARGIN
            ahead = by >= C.CORRIDOR_TOP_Y
            in_lane = u is not None and abs(u) <= C.IN_LANE and ahead
            status = f"u{u:+.1f}" if u is not None else "-"
            if du is not None and abs(du) >= 0.2:
                status += f" {du:+.1f}/s"

            found = []
            # --- признаки для модели риска: «насколько опасен самый опасный участник» по каждому аспекту
            if dist is not None and ahead:
                near = 1 / max(dist, 1.0)
                toward = -math.copysign(1, u) * du if (u is not None and du is not None) else 0.0
                if vehicle:
                    if in_lane and inside:
                        F["veh_in_near"] = max(F["veh_in_near"], near)
                        if ttc is not None:
                            F["veh_in_ttc"] = max(F["veh_in_ttc"], 1 / ttc)
                    elif inside and u is not None and abs(u) <= C.CUTIN_MAX_U:
                        F["veh_side_toward"] = max(F["veh_side_toward"], toward * near * 10)
                        F["veh_side_near"] = max(F["veh_side_near"], near)
                        if ttc is not None:
                            F["veh_side_ttc"] = max(F["veh_side_ttc"], 1 / ttc)
                    F["veh_count_near"] += float(dist < 20)
                    F["veh_lat_speed"] = max(F["veh_lat_speed"], abs(du or 0) * near * 10)
                else:
                    if in_lane:
                        F["vru_in_near"] = max(F["vru_in_near"], near)
                    elif u is not None and abs(u) <= 2.5:
                        F["vru_side_toward"] = max(F["vru_side_toward"], toward * near * 10)
                        F["vru_side_near"] = max(F["vru_side_near"], near)
            if vehicle and in_lane and inside and (lead is None or y2 > self.hist[lead.tid][-1][4]):
                lead = d
            # --- подрезание / пересечение пути: машина сбоку быстро идёт в нашу полосу
            if u is not None and du is not None and ahead and inside and not in_lane and abs(u) <= C.CUTIN_MAX_U:
                toward = -math.copysign(1, u) * du                      # полос/с к нашей полосе
                t_in = (abs(u) - C.IN_LANE) / toward if toward > 0 else None
                # опасно, только если мы к нему приближаемся (TTC) или он уже рядом: на светофоре
                # поперечный поток и перестроения далеко впереди — не угроза
                closing_in = (ttc is not None and ttc <= C.CUTIN_TTC) or (dist is not None and dist <= C.CRIT_DIST)
                if vehicle and closing_in and dist <= C.CUTIN_MAX_DIST and toward >= C.CUTIN_MIN_SPEED \
                        and t_in is not None and t_in <= C.CUTIN_HORIZON:
                    # видна сбоку (едет поперёк — перекрёсток, разворот) — «пересекает путь»; иначе перестроение
                    kind = "crossing" if side_view else "cutin"
                    found.append((kind, 3 if ttc is not None and ttc < C.CUTIN_CRIT_TTC else 2))
                if not vehicle and closing_in and dist <= C.VRU_DIST and toward >= C.VRU_MIN_SPEED \
                        and t_in is not None and t_in <= C.VRU_HORIZON:
                    found.append(("ped", 3 if ttc is not None and ttc < C.CUTIN_CRIT_TTC else 2))
            # --- пешеход / велосипед / мотоцикл на нашем пути
            if not vehicle and in_lane and dist is not None and dist <= C.VRU_NEAR:
                found.append(("ped", 3 if dist <= C.CRIT_DIST else 2))
                res.peds.append(d)
            # --- машина в нашей полосе быстро сближается
            if vehicle and in_lane and inside and not side_view and ttc is not None and w >= C.TTC_MIN_WIDTH:
                closing_ms = dist / ttc if dist else 0
                if closing_ms > C.LEAD_MAX_CLOSING:
                    # сближение быстрее ~43 км/ч: встречная. По разметке видно, что она в нашей полосе —
                    # это опасность; без разметки (поворот, трапеция) — скорее ложная «машина впереди»
                    if lanes_ok and ttc < C.WRONGWAY_TTC:
                        found.append(("wrongway", 3))
                    elif d is lead:
                        res.oncoming = True
                elif ttc < C.TTC_WARNING:
                    crit = ttc < C.TTC_CRITICAL and w >= C.CRITICAL_MIN_WIDTH
                    found.append(("ttc", 3) if crit else ("closing", 2))

            for kind, level in found:
                threats.append(Threat(d, kind, level, u, du, ttc, dist, self._predict(h, u, du, yaw)))
                raw.add(kind)
            res.tracks.append((d, status + "".join(f" {k.upper()}" for k, _ in found)))

        # --- авария у любой машины в кадре: держим тревогу CRASH_HOLD секунд, даже если трек потерян
        for tid in self._crashes(t):
            self.crash_at[tid] = t
        self.crash_at = {k: v for k, v in self.crash_at.items() if t - v <= C.CRASH_HOLD}
        if self.crash_at:
            raw.add("crash")
            res.crashed = [d for d in live if d.tid in self.crash_at]
            threats += [Threat(d, "crash", 3, None, None, None, None) for d in res.crashed]

        # --- машина впереди: дистанция, TTC, «слишком близко», виляние, стоп-сигналы
        if lead:
            res.lead = lead
            h = self.hist[lead.tid]
            w = h[-1][3] - h[-1][1]
            res.distance = self._distance(lead.cls, w, h[-1][4] - h[-1][2], aspect)
            res.ttc = None if res.oncoming else self._ttc(h, t)
            if w * aspect > C.CROSS_MIN_ASPECT * (h[-1][4] - h[-1][2]):
                res.ttc = None                       # впереди машина боком — её TTC по ширине неверен
            # «слишком близко» — только если ещё и сближаемся: в пробке вплотную стоять нормально
            if w >= C.CLOSE_WIDTH and res.ttc is not None and res.ttc < C.CLOSE_MAX_TTC:
                raw.add("close")
                threats.append(Threat(lead, "close", 1, h[-1][5], None, res.ttc, res.distance))
            # по разметке виляние не считаем: её дрожание больше порога амплитуды (0.15 ширины машины)
            weaving, res.weave = self._weaving(h, t, stats=True)
            if weaving:
                raw.add("weave")
                threats.append(Threat(lead, "weave", 2, h[-1][5], None, res.ttc, res.distance))
            if C.BRAKE_LIGHTS and frame is not None and self._brake_lights(frame, lead):
                raw.add("brakelight")

        F["crash"] = float(bool(self.crash_at))
        F["oncoming"] = float(res.oncoming or "wrongway" in raw)
        F["weave"] = float("weave" in raw)
        res.features = F
        p = self.risk(t, F)                  # обученная модель (risk.py): похоже ли на предаварийную ситуацию
        if p is not None and p >= self.risk.threshold:
            raw.add("danger")
        res.raw, res.threats = raw, threats
        res.alerts = self._confirm(t, raw, threats)
        res.risk = p if p is not None else max((th.level for th in threats), default=0) / 3
        return res

    # ================================================================ подтверждение
    def _confirm(self, t, raw, threats):
        """Условие должно продержаться CONFIRM_SEC (защита от шума), но короткие провалы до RAW_GAP
        не сбрасывают его — так тревога не мигает и не запаздывает из-за одного пропущенного кадра."""
        for kind in raw:
            self.since.setdefault(kind, t)
            self.last_raw[kind] = t
        for kind in list(self.since):
            if t - self.last_raw[kind] > C.RAW_GAP:
                del self.since[kind]
        active = {k for k in self.since
                  if t - self.last_raw[k] <= C.ALERT_HOLD and self.last_raw[k] - self.since[k] >= C.CONFIRM_SEC[k]}
        if "ttc" in active:
            active.discard("closing")   # критическое заменяет обычное сближение
        alerts = []
        for k in active:
            level, screen, voice = C.ALERTS[k]
            level = max([level] + [th.level for th in threats if th.kind == k])
            alerts.append(Alert(k, level, screen, voice))
        return alerts

    # ================================================================ геометрия
    def _lateral(self, cls, box, aspect, lanes_ok):
        """Боковое положение в ширинах полосы: 0 — центр нашей полосы, ±1 — соседняя."""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        if lanes_ok:
            u = self.lanes.lane_pos(cx, y2)
            if u is not None:
                return u
        if y2 < C.CORRIDOR_TOP_Y:
            return None
        if cls in C.VEHICLE_CLASSES:
            meters = (cx - C.CORRIDOR_CENTER_X) / max(x2 - x1, 1e-6) * C.VEHICLE_WIDTH_M
        else:   # смещение в метрах по росту человека (у велосипеда/мотоцикла — по высоте рамки)
            meters = (cx - C.CORRIDOR_CENTER_X) * aspect / max(y2 - y1, 1e-6) * C.PERSON_HEIGHT_M
        return meters / C.LANE_WIDTH_M

    def _distance(self, cls, w, h, aspect, side_view=False):
        """Дистанция, м: по ширине машины сзади; у машины сбоку ширина рамки — это её длина,
        поэтому по высоте; у людей и двухколёсных — по росту."""
        if cls in C.VEHICLE_CLASSES and cls != 3:
            if not side_view:
                return C.VEHICLE_WIDTH_M * focal() / max(w, 1e-6)
            real_h = C.TRUCK_HEIGHT_M if cls in (5, 7) else C.CAR_HEIGHT_M
            return real_h * focal() * aspect / max(h, 1e-6)
        return C.PERSON_HEIGHT_M * focal() * aspect / max(h, 1e-6)

    def _yaw_rate(self, t, live):
        """Поворот своей машины: все рамки разом сдвигаются по x. Медиана скорости сдвига, доли кадра/с."""
        speeds = []
        for d in live:
            h = self.hist[d.tid]
            if len(h) >= 2 and h[-1][0] - h[-2][0] <= 0.3:
                dt = h[-1][0] - h[-2][0]
                speeds.append(((h[-1][1] + h[-1][3]) - (h[-2][1] + h[-2][3])) / 2 / max(dt, 1e-3))
        self.yaw.append((t, float(np.median(speeds)) if len(speeds) >= C.YAW_MIN_TRACKS else 0.0))
        while t - self.yaw[0][0] > C.KIN_WINDOW:
            self.yaw.popleft()
        return float(np.mean([v for _, v in self.yaw]))

    def _lat_speed(self, h, t, yaw, lanes_ok, vehicle, aspect):
        """du/dt, полос/с. По разметке — напрямую (она поворачивается вместе с кадром);
        без разметки — из сдвига рамки за вычетом поворота своей машины и «разъезда» перспективы."""
        pts = [p for p in h if t - p[0] <= C.KIN_WINDOW and p[5] is not None]
        if len(pts) < 4 or pts[-1][0] - pts[0][0] < C.KIN_MIN_SPAN:
            return None
        a = np.array([(p[0], (p[1] + p[3]) / 2, p[3] - p[1], p[5]) for p in pts])
        ts, cx, w, u = a.T
        if lanes_ok:
            return float(np.polyfit(ts, u, 1)[0])
        if not vehicle:      # человек: u по росту; поворот своей машины сдвигает его на yaw / высоту рамки
            hh = pts[-1][4] - pts[-1][2]
            return float(np.polyfit(ts, u, 1)[0] - yaw * aspect / max(hh, 1e-6) * C.PERSON_HEIGHT_M / C.LANE_WIDTH_M)
        # X = (cx - c) / w;  dX/dt = (dcx/dt - yaw) / w - X * dln(w)/dt  (у машины с постоянным
        # боковым смещением, которая приближается, dX/dt = 0 — её рамка «уезжает» вбок только из-за перспективы)
        dcx = np.polyfit(ts, cx, 1)[0]
        g = np.polyfit(ts, np.log(w), 1)[0]
        X = (cx[-1] - C.CORRIDOR_CENTER_X) / w[-1]
        dX = (dcx - yaw) / w[-1] - X * g
        return float(dX * C.VEHICLE_WIDTH_M / C.LANE_WIDTH_M)

    def _predict(self, h, u, du, yaw):
        """Куда сместится рамка через PREDICT_SEC (для стрелки на экране), доли кадра."""
        _, x1, y1, x2, y2, _ = h[-1]
        if du is None or u is None:
            return None
        shift = du * C.PREDICT_SEC                          # в полосах
        lane_w = (x2 - x1) * C.LANE_WIDTH_M / C.VEHICLE_WIDTH_M
        return ((x1 + x2) / 2 + shift * lane_w, y2)

    # ---------- время до столкновения ----------
    def _ttc(self, h, t, by_height=False):
        """TTC по скорости роста рамки: d ln(размер)/dt = 1/TTC. Расстояние знать не нужно."""
        pts = [(p[0], (p[4] - p[2]) if by_height else (p[3] - p[1])) for p in h if t - p[0] <= C.TTC_WINDOW]
        if len(pts) < 4 or pts[-1][0] - pts[0][0] < C.TTC_MIN_SPAN:
            return None
        if not by_height and pts[-1][1] < C.TTC_MIN_WIDTH:
            return None
        ts, ws = np.array(pts).T
        rate = np.polyfit(ts, np.log(ws), 1)[0]
        if rate <= C.TTC_MIN_RATE:
            return None
        # прямая даёт TTC для середины окна; к текущему моменту он меньше на полокна
        return max(1.0 / rate - (ts[-1] - ts.mean()), 0.1)

    # ---------- виляние ----------
    def _weaving(self, h, t, stats=False, lanes_ok=False):
        """stats=True — вернуть ещё (амплитуда, смен направления) для отладки.
        С разметкой положение берём относительно полосы (u в ширинах машины): наши повороты и
        перестроения не выглядят как виляние машины впереди."""
        out = (lambda ok, st=None: (ok, st)) if stats else (lambda ok, st=None: ok)
        if lanes_ok:
            k = C.LANE_WIDTH_M / C.VEHICLE_WIDTH_M
            pts = [(p[0], 1.0, p[5] * k) for p in h if t - p[0] <= C.WEAVE_WINDOW and p[5] is not None]
        else:
            pts = [(p[0], p[3] - p[1], (p[1] + p[3]) / 2) for p in h if t - p[0] <= C.WEAVE_WINDOW]
        if len(pts) < 10 or pts[-1][0] - pts[0][0] < C.WEAVE_MIN_SPAN:
            return out(False)
        ts, ws, xs = np.array(pts).T
        width = ws.mean()
        if not lanes_ok and width < C.WEAVE_MIN_WIDTH or lanes_ok and h[-1][3] - h[-1][1] < C.WEAVE_MIN_WIDTH:
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
        for tid, h in self.hist.items():
            if self.last_seen.get(tid) != t or self.cls.get(tid) not in C.VEHICLE_CLASSES:
                continue
            old = next((p for p in h if t - p[0] <= C.CRASH_WINDOW), None)
            if old is None or old is h[-1] or t - old[0] < C.CRASH_WINDOW / 2:
                continue
            pairs[tid] = (old, h[-1])
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
            self.cls.pop(tid, None)
            self.red_base.pop(tid, None)
