"""Road situation analysis: from detections, lane markings, crosswalks and own motion to threats and alerts.

The module knows nothing about windows or sound, so it is shared by main.py, evaluate.py,
debug_video.py, tools/replay.py and tools/train_risk.py.

Every car, pedestrian, bicycle and motorcycle in view gets:
  u   - lateral position in lane widths: 0 = centre of our lane, +-1 = centre of the next lane.
        From lane markings (lanes.py) when visible; otherwise from box geometry (offset in car
        widths or in person heights - independent of distance);
  du  - lateral speed in lanes per second, corrected for our own turning;
  TTC - time to collision from box growth; distance from width (car) or height (person, side view).
Threats derived from that:
  closing / ttc  - car in our lane closing in fast;
  hardbrake      - car ahead starts braking hard (growth rate jumps);
  cutin          - car from the next lane moving into ours in front of us;
  drift          - car slowly drifting towards our lane (attention);
  crossing       - car seen from the side crossing our path;
  wrongway       - oncoming car in our lane (needs visible markings);
  ped            - pedestrian / two-wheeler on our path, heading into it, or suddenly appearing close;
  ped_xwalk      - pedestrian at a crosswalk ahead (crosswalk.py);
  vru_near       - pedestrian / cyclist close to the road edge ahead;
  crash          - any car "takes off" (box jumps up and gets taller) - a collision in view;
  weave          - car ahead oscillates in its lane;
  close          - too close to the car ahead while closing in;
  danger         - learned risk model (risk.py) says the scene looks like the seconds before a crash.
Levels: 1 green (attention), 2 yellow (possible risk), 3 red (danger).
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

import config as C
from lanes import LaneModel

VRU_CLASSES = {C.PERSON_CLASS, C.BICYCLE_CLASS, 3}       # person, bicycle, motorcycle: vulnerable road users
# scene features for the risk model (risk.py); "near" = 1 / distance, "toward" = speed towards our lane
FEATURES = ["veh_in_near", "veh_in_ttc", "veh_in_rate", "veh_in_jerk", "veh_side_toward", "veh_side_near",
            "veh_side_ttc", "veh_cross_near", "veh_count_near", "veh_lat_speed", "vru_in_near", "vru_side_toward",
            "vru_side_near", "vru_new_near", "vru_count_near", "xwalk", "crash", "crash_rise", "oncoming", "weave",
            "lanes_ok", "ego_speed", "ego_yaw", "n_tracks", "any_ttc"]
_SKIP = object()      # "crosswalk not analysed on this step"


@dataclass
class Detection:
    box: tuple   # x1, y1, x2, y2 in pixels
    cls: int
    tid: int     # track id


@dataclass
class Alert:
    kind: str
    level: int
    screen: str
    voice: str


@dataclass
class Threat:
    det: Detection
    kind: str           # threat type (key of C.ALERTS)
    level: int          # 1-3
    u: float | None     # lateral position, lanes
    du: float | None    # lateral speed, lanes/s
    ttc: float | None
    dist: float | None
    path: tuple | None = None   # (x, y) where the object will be in C.PREDICT_SEC - for the on-screen arrow


@dataclass
class FrameResult:
    alerts: list = field(default_factory=list)
    lead: Detection | None = None
    ttc: float | None = None
    distance: float | None = None   # approximate distance to the car ahead, m
    oncoming: bool = False          # the car ahead looks like an oncoming one
    peds: list = field(default_factory=list)
    crashed: list = field(default_factory=list)    # cars with crash signs (any, not only ahead)
    threats: list = field(default_factory=list)    # Threat for all objects
    lanes: LaneModel | None = None                 # lane markings (for drawing) when visible
    crosswalk: tuple | None = None                 # crosswalk box (frame fractions) when visible
    ego_speed: float | None = None                 # m/s, when known
    risk: float = 0.0                              # overall danger 0..1
    features: dict = field(default_factory=dict)   # scene features for the risk model (risk.py)
    # for debugging (debug_video.py): status of every box, raw conditions before confirmation, weave metrics
    tracks: list = field(default_factory=list)     # [(Detection, status)]
    raw: set = field(default_factory=set)
    weave: tuple | None = None                     # (amplitude in widths, direction changes)

    @property
    def top(self):
        return max(self.alerts, key=lambda a: a.level, default=None)

    @property
    def level(self):
        return self.top.level if self.top else 0


# ---------- own lane without markings: trapezoid in front of the car ----------
def _half_width(y):
    k = (y - C.CORRIDOR_TOP_Y) / (1 - C.CORRIDOR_TOP_Y)
    return C.CORRIDOR_TOP_HALF + k * (C.CORRIDOR_BOTTOM_HALF - C.CORRIDOR_TOP_HALF)


def in_corridor(x, y):
    """x, y as frame fractions; y is the bottom edge of the box."""
    return y >= C.CORRIDOR_TOP_Y and abs(x - C.CORRIDOR_CENTER_X) <= _half_width(y)


def on_path(cx, width):
    """Lateral offset of a car in its own widths: (cx - centre) / width = X / real_width.

    Independent of distance. A car on our path has an offset < ~0.7,
    a car parked at the kerb 1.5 and more (it grows fast in the frame, but we pass it).
    """
    return abs(cx - C.CORRIDOR_CENTER_X) / max(width, 1e-6) <= C.LEAD_MAX_OFFSET


def corridor_polygon(W, H):
    cx, top = C.CORRIDOR_CENTER_X, C.CORRIDOR_TOP_Y
    pts = [(cx - C.CORRIDOR_TOP_HALF, top), (cx + C.CORRIDOR_TOP_HALF, top),
           (cx + C.CORRIDOR_BOTTOM_HALF, 1.0), (cx - C.CORRIDOR_BOTTOM_HALF, 1.0)]
    return np.array([(int(x * W), int(y * H)) for x, y in pts], np.int32)


def focal():
    """Focal length as a fraction of frame width (from the camera field of view)."""
    return 0.5 / math.tan(math.radians(C.CAMERA_HFOV_DEG) / 2)


class BehaviorEngine:
    def __init__(self):
        # track id -> [(t, x1, y1, x2, y2, u)]: box as frame fractions and lateral position in lanes
        self.hist = defaultdict(deque)
        self.cls = {}
        self.last_seen = {}
        self.first_seen = {}
        self.rates = defaultdict(deque)     # track id -> [(t, growth rate)] for hard-braking detection
        self.since = {}                     # kind -> when the condition appeared
        self.last_raw = {}                  # kind -> when the condition was last seen
        self.red_base = {}                  # usual share of red pixels on a car (brake lights)
        self.crash_at = {}                  # track id -> last time crash signs were seen
        self.lanes = LaneModel()
        from crosswalk import CrosswalkModel
        self.xwalk = CrosswalkModel()
        from risk import RiskModel
        self.risk = RiskModel()
        self.yaw = deque()                  # (t, median box shift in x) - own turning
        self.crash_rise = 0.0

    # ================================================================ main step
    def update(self, t, W, H, detections, frame=None, lanes=None, ego=None, crosswalk=_SKIP):
        """t - video time, s; frame - image (brake lights); lanes - lines from LaneDetector.detect;
        ego - (speed m/s, yaw frac/s, ok) from EgoMotion; crosswalk - crosswalk.detect() result
        (box or None) when it was analysed on this step."""
        res = FrameResult()
        self.lanes.update(t, lanes)
        lanes_ok = self.lanes.valid()
        res.lanes = self.lanes if lanes_ok else None
        if crosswalk is not _SKIP:
            self.xwalk.update(t, crosswalk)
        elif self.xwalk.box is not None and t - self.xwalk.seen > 1.0:
            self.xwalk.box = None
        res.crosswalk = self.xwalk.box
        ego_ok = ego is not None and ego[2]
        res.ego_speed = ego[0] if ego_ok else None
        stopped = ego_ok and ego[0] < 1.0
        aspect = W / H

        live = []
        for d in detections:
            x1, y1, x2, y2 = d.box
            if (y1 + y2) / 2 / H > C.HOOD_Y:
                res.tracks.append((d, "hood"))
                continue                     # our own hood
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
            self.first_seen.setdefault(d.tid, t)
            live.append(d)
        self._forget(t)
        yaw = ego[1] if ego_ok else self._yaw_rate(t, live)
        if ego_ok:
            self._yaw_rate(t, live)          # keep the history consistent

        raw, threats, lead = set(), [], None
        F = dict.fromkeys(FEATURES, 0.0)
        F["lanes_ok"] = float(lanes_ok)
        F["xwalk"] = float(res.crosswalk is not None)
        F["ego_speed"] = ego[0] if ego_ok else -1.0
        F["ego_yaw"] = abs(yaw)
        F["n_tracks"] = float(len(live))
        for d in live:
            h = self.hist[d.tid]
            _, x1, y1, x2, y2, u = h[-1]
            w, by = x2 - x1, y2
            vehicle = d.cls in C.VEHICLE_CLASSES
            side_view = (x2 - x1) * aspect > C.CROSS_MIN_ASPECT * (y2 - y1)     # car seen from the side
            dist = self._distance(d.cls, x2 - x1, y2 - y1, aspect, side_view)
            # TTC from width (more precise for a car seen from behind) or from height (independent of
            # turning - for side views and people); for people it tells whether we approach them
            ttc = self._ttc(h, t, by_height=side_view or not vehicle)
            du = self._lat_speed(h, t, yaw, lanes_ok, vehicle, aspect)
            inside = x1 > C.EDGE_MARGIN and x2 < 1 - C.EDGE_MARGIN
            ahead = by >= C.CORRIDOR_TOP_Y
            in_lane = u is not None and abs(u) <= C.IN_LANE and ahead
            toward = -math.copysign(1, u) * du if (u is not None and du is not None and u != 0) else 0.0
            age = t - self.first_seen[d.tid]
            status = f"u{u:+.1f}" if u is not None else "-"
            if du is not None and abs(du) >= 0.2:
                status += f" {du:+.1f}/s"

            found = []
            # --- features for the risk model: "how dangerous is the most dangerous object" per aspect
            if dist is not None and ahead:
                near = 1 / max(dist, 1.0)
                if ttc is not None and u is not None and abs(u) <= 1.5:
                    F["any_ttc"] = max(F["any_ttc"], 1 / ttc)
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
                    if side_view and u is not None and abs(u) <= 2.0:
                        F["veh_cross_near"] = max(F["veh_cross_near"], near)
                    F["veh_count_near"] += float(dist < 20)
                    F["veh_lat_speed"] = max(F["veh_lat_speed"], abs(du or 0) * near * 10)
                else:
                    F["vru_count_near"] += float(dist < 20)
                    if in_lane:
                        F["vru_in_near"] = max(F["vru_in_near"], near)
                    elif u is not None and abs(u) <= 2.5:
                        F["vru_side_toward"] = max(F["vru_side_toward"], toward * near * 10)
                        F["vru_side_near"] = max(F["vru_side_near"], near)
                    if age <= C.VRU_SUDDEN_AGE:
                        F["vru_new_near"] = max(F["vru_new_near"], near)
            if vehicle and in_lane and inside and (lead is None or y2 > self.hist[lead.tid][-1][4]):
                lead = d

            # --- cut-in / crossing / drift: a car from the side moving into our lane
            if u is not None and du is not None and ahead and inside and not in_lane and abs(u) <= C.CUTIN_MAX_U:
                t_in = (abs(u) - C.IN_LANE) / toward if toward > 0 else None
                # dangerous only if we approach it (TTC) or it is already close: at traffic lights the
                # cross traffic and lane changes far ahead are not a threat
                closing_in = (ttc is not None and ttc <= C.CUTIN_TTC) or (dist is not None and dist <= C.CRIT_DIST)
                if vehicle and closing_in and dist <= C.CUTIN_MAX_DIST and toward >= C.CUTIN_MIN_SPEED \
                        and t_in is not None and t_in <= C.CUTIN_HORIZON:
                    # seen from the side (driving across - junction, U-turn): "crossing"; else a lane change
                    kind = "crossing" if side_view else "cutin"
                    found.append((kind, 3 if ttc is not None and ttc < C.CUTIN_CRIT_TTC else 2))
                elif vehicle and not side_view and dist <= C.DRIFT_MAX_DIST and toward >= C.DRIFT_MIN_SPEED \
                        and t_in is not None and t_in <= C.DRIFT_HORIZON and not stopped:
                    found.append(("drift", 1))
            # --- pedestrians and two-wheelers
            if not vehicle and dist is not None and ahead:
                lvl = 0
                kind = "ped"
                if in_lane and dist <= C.VRU_NEAR:                          # on our path
                    lvl = 3 if dist <= C.CRIT_DIST else 2
                elif u is not None and du is not None and not in_lane and toward >= C.VRU_MIN_SPEED:
                    t_in = (abs(u) - C.IN_LANE) / toward
                    closing_in = (ttc is not None and ttc <= C.CUTIN_TTC) or dist <= C.CRIT_DIST
                    if closing_in and dist <= C.VRU_DIST and t_in <= C.VRU_HORIZON:   # stepping into our lane
                        lvl = 3 if ttc is not None and ttc < C.CUTIN_CRIT_TTC else 2
                if not lvl and age <= C.VRU_SUDDEN_AGE and dist <= C.VRU_SUDDEN_DIST \
                        and u is not None and abs(u) <= C.VRU_SUDDEN_U:            # appeared suddenly close
                    lvl = 2
                if not lvl and C.CROSSWALK and self.xwalk.near((x1 + x2) / 2, by) and dist <= C.CROSSWALK_DIST:
                    kind, lvl = "ped_xwalk", 1                                   # at a crosswalk ahead
                if not lvl and u is not None and abs(u) <= C.VRU_EDGE_U and dist <= C.VRU_ATTENTION_DIST:
                    kind, lvl = "vru_near", 1                                    # close to the road edge
                if lvl and stopped and kind == "ped":
                    lvl = 1                  # we stand still: people crossing in front are not a risk yet
                if lvl:
                    found.append((kind, lvl))
                    if kind == "ped":
                        res.peds.append(d)
            # --- car in our lane closing in fast
            if vehicle and in_lane and inside and not side_view and ttc is not None and w >= C.TTC_MIN_WIDTH:
                closing_ms = dist / ttc if dist else 0
                if closing_ms > C.LEAD_MAX_CLOSING:
                    # closing faster than ~43 km/h: oncoming. Lane markings show it is in our lane - danger;
                    # without markings (curve, trapezoid) it is more likely a false "car ahead"
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

        # --- crash of any car in view: keep the alarm CRASH_HOLD seconds even if the track is lost
        for tid in self._crashes(t):
            self.crash_at[tid] = t
        self.crash_at = {k: v for k, v in self.crash_at.items() if t - v <= C.CRASH_HOLD}
        if self.crash_at:
            raw.add("crash")
            res.crashed = [d for d in live if d.tid in self.crash_at]
            threats += [Threat(d, "crash", 3, None, None, None, None) for d in res.crashed]

        # --- car ahead: distance, TTC, hard braking, "too close", weaving, brake lights
        if lead:
            res.lead = lead
            h = self.hist[lead.tid]
            w = h[-1][3] - h[-1][1]
            res.distance = self._distance(lead.cls, w, h[-1][4] - h[-1][2], aspect)
            res.ttc = None if res.oncoming else self._ttc(h, t)
            side = w * aspect > C.CROSS_MIN_ASPECT * (h[-1][4] - h[-1][2])
            if side:
                res.ttc = None                       # side view ahead: TTC from width is wrong
            rate, jerk = self._brake(lead.tid, h, t) if not side else (0.0, 0.0)
            F["veh_in_rate"], F["veh_in_jerk"] = max(rate, 0.0), max(jerk, 0.0)
            if not res.oncoming and res.distance is not None and res.distance <= C.BRAKE_MAX_DIST \
                    and rate >= C.BRAKE_RATE and jerk >= C.BRAKE_JERK:
                raw.add("hardbrake")
                crit = res.ttc is not None and res.ttc < C.BRAKE_CRIT_TTC
                threats.append(Threat(lead, "hardbrake", 3 if crit else 2, h[-1][5], None, res.ttc, res.distance))
            # "too close" only if also closing in: standing bumper to bumper in a jam is fine
            if w >= C.CLOSE_WIDTH and res.ttc is not None and res.ttc < C.CLOSE_MAX_TTC:
                raw.add("close")
                threats.append(Threat(lead, "close", 1, h[-1][5], None, res.ttc, res.distance))
            # no weaving from lane markings: their jitter is above the amplitude threshold
            weaving, res.weave = self._weaving(h, t, stats=True)
            if weaving:
                raw.add("weave")
                threats.append(Threat(lead, "weave", 2, h[-1][5], None, res.ttc, res.distance))
            if C.BRAKE_LIGHTS and frame is not None and self._brake_lights(frame, lead):
                raw.add("brakelight")
                threats.append(Threat(lead, "brakelight", 1, h[-1][5], None, res.ttc, res.distance))

        F["crash"] = float(bool(self.crash_at))
        F["crash_rise"] = self.crash_rise
        F["oncoming"] = float(res.oncoming or "wrongway" in raw)
        F["weave"] = float("weave" in raw)
        res.features = F
        p = self.risk(t, F)                  # learned model (risk.py): does this look like the seconds before a crash
        if p is not None and p >= self.risk.threshold:
            raw.add("danger")
            lvl = 3 if p >= self.risk.threshold_red else 2
            target = lead or (max(live, key=lambda d: d.box[3]) if live else None)
            if target is not None:
                threats.append(Threat(target, "danger", lvl, None, None, None, None))
        res.raw, res.threats = raw, threats
        res.alerts = self._confirm(t, raw, threats)
        res.risk = p if p is not None else max((th.level for th in threats), default=0) / 3
        return res

    # ================================================================ confirmation
    def _confirm(self, t, raw, threats):
        """A condition must hold CONFIRM_SEC (noise protection), but short dropouts up to RAW_GAP do not
        reset it - so the alarm neither flickers nor lags because of one missed frame."""
        for kind in raw:
            self.since.setdefault(kind, t)
            self.last_raw[kind] = t
        for kind in list(self.since):
            if t - self.last_raw[kind] > C.RAW_GAP:
                del self.since[kind]
        active = {k for k in self.since
                  if t - self.last_raw[k] <= C.ALERT_HOLD and self.last_raw[k] - self.since[k] >= C.CONFIRM_SEC[k]}
        if "ttc" in active:
            active.discard("closing")   # critical replaces ordinary closing
        if "cutin" in active:
            active.discard("drift")
        alerts = []
        for k in active:
            base, screen, voice = C.ALERTS[k]
            levels = [th.level for th in threats if th.kind == k]
            alerts.append(Alert(k, max(levels) if levels else base, screen, voice))
        return alerts

    # ================================================================ geometry
    def _lateral(self, cls, box, aspect, lanes_ok):
        """Lateral position in lane widths: 0 = centre of our lane, +-1 = next lane."""
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
        else:   # offset in metres from person height (bicycle/motorcycle: box height)
            meters = (cx - C.CORRIDOR_CENTER_X) * aspect / max(y2 - y1, 1e-6) * C.PERSON_HEIGHT_M
        return meters / C.LANE_WIDTH_M

    def _distance(self, cls, w, h, aspect, side_view=False):
        """Distance, m: from the width of a car seen from behind; for a side view the box width is the
        car length, so use height; for people and two-wheelers - height."""
        if cls in C.VEHICLE_CLASSES and cls != 3:
            if not side_view:
                return C.VEHICLE_WIDTH_M * focal() / max(w, 1e-6)
            real_h = C.TRUCK_HEIGHT_M if cls in (5, 7) else C.CAR_HEIGHT_M
            return real_h * focal() * aspect / max(h, 1e-6)
        return C.PERSON_HEIGHT_M * focal() * aspect / max(h, 1e-6)

    def _yaw_rate(self, t, live):
        """Own turning: all boxes shift in x together. Median shift speed, frame fractions/s."""
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
        """du/dt, lanes/s. From markings - directly (they turn in the frame together with the cars);
        without markings - from the box shift minus own turning and perspective spreading."""
        pts = [p for p in h if t - p[0] <= C.KIN_WINDOW and p[5] is not None]
        if len(pts) < 4 or pts[-1][0] - pts[0][0] < C.KIN_MIN_SPAN:
            return None
        a = np.array([(p[0], (p[1] + p[3]) / 2, p[3] - p[1], p[5]) for p in pts])
        ts, cx, w, u = a.T
        if lanes_ok:
            return float(np.polyfit(ts, u, 1)[0])
        if not vehicle:      # person: u from height; own turning shifts them by yaw / box height
            hh = pts[-1][4] - pts[-1][2]
            return float(np.polyfit(ts, u, 1)[0] - yaw * aspect / max(hh, 1e-6) * C.PERSON_HEIGHT_M / C.LANE_WIDTH_M)
        # X = (cx - c) / w;  dX/dt = (dcx/dt - yaw) / w - X * dln(w)/dt  (a car with a constant lateral
        # offset that comes closer has dX/dt = 0 - its box drifts sideways only because of perspective)
        dcx = np.polyfit(ts, cx, 1)[0]
        g = np.polyfit(ts, np.log(w), 1)[0]
        X = (cx[-1] - C.CORRIDOR_CENTER_X) / w[-1]
        dX = (dcx - yaw) / w[-1] - X * g
        return float(dX * C.VEHICLE_WIDTH_M / C.LANE_WIDTH_M)

    def _predict(self, h, u, du, yaw):
        """Where the box will shift in PREDICT_SEC (for the on-screen arrow), frame fractions."""
        _, x1, y1, x2, y2, _ = h[-1]
        if du is None or u is None:
            return None
        shift = du * C.PREDICT_SEC                          # lanes
        lane_w = (x2 - x1) * C.LANE_WIDTH_M / C.VEHICLE_WIDTH_M
        return ((x1 + x2) / 2 + shift * lane_w, y2)

    # ---------- time to collision ----------
    @staticmethod
    def _rate(h, t, by_height=False):
        """Box growth rate d ln(size)/dt over TTC_WINDOW and the time spread of the points (None if short)."""
        pts = [(p[0], (p[4] - p[2]) if by_height else (p[3] - p[1])) for p in h if t - p[0] <= C.TTC_WINDOW]
        if len(pts) < 4 or pts[-1][0] - pts[0][0] < C.TTC_MIN_SPAN:
            return None, None, None
        ts, ws = np.array(pts).T
        return float(np.polyfit(ts, np.log(np.maximum(ws, 1e-6)), 1)[0]), ts, ws

    def _ttc(self, h, t, by_height=False):
        """TTC from box growth: d ln(size)/dt = 1/TTC. No distance needed."""
        rate, ts, ws = self._rate(h, t, by_height)
        if rate is None or (not by_height and ws[-1] < C.TTC_MIN_WIDTH) or rate <= C.TTC_MIN_RATE:
            return None
        # the fitted line gives TTC for the middle of the window; now it is smaller by half a window
        return max(1.0 / rate - (ts[-1] - ts.mean()), 0.1)

    def _brake(self, tid, h, t):
        """Hard braking of the car ahead: (growth rate now, its rise over BRAKE_JERK_SEC)."""
        rate, _, _ = self._rate(h, t)
        q = self.rates[tid]
        if rate is None:
            return 0.0, 0.0
        q.append((t, rate))
        while t - q[0][0] > C.BRAKE_JERK_SEC:
            q.popleft()
        return rate, rate - min(r for _, r in q)

    # ---------- weaving ----------
    def _weaving(self, h, t, stats=False):
        """stats=True also returns (amplitude, direction changes) for debugging."""
        out = (lambda ok, st=None: (ok, st)) if stats else (lambda ok, st=None: ok)
        pts = [(p[0], p[3] - p[1], (p[1] + p[3]) / 2) for p in h if t - p[0] <= C.WEAVE_WINDOW]
        if len(pts) < 10 or pts[-1][0] - pts[0][0] < C.WEAVE_MIN_SPAN:
            return out(False)
        ts, ws, xs = np.array(pts).T
        width = ws.mean()
        if width < C.WEAVE_MIN_WIDTH:
            return out(False)
        # 1) amplitude: remove the common trend (road curve), look at the oscillation spread
        resid = xs - np.polyval(np.polyfit(ts, xs, 1), ts)
        amp = resid.std() / width
        # 2) character: weaving is repeated reversal of lateral speed;
        #    a lane change has one direction, an overtake and back - one reversal
        fps = (len(ts) - 1) / (ts[-1] - ts[0])
        k = max(3, int(0.3 * fps))
        smooth = np.convolve(xs, np.ones(k) / k, mode="valid")
        speed = np.diff(smooth) * fps / width                  # car widths per second
        moving = np.sign(speed[np.abs(speed) > C.WEAVE_MIN_SPEED])
        rev = int(np.count_nonzero(np.diff(moving)))
        return out(amp >= C.WEAVE_REL_STD and rev >= C.WEAVE_MIN_REVERSALS, (amp, rev))

    # ---------- crash of any car ----------
    def _crashes(self, t):
        """A car "takes off": its box jumps up and gets taller relative to its width, while the width
        barely changes (the car is not driving away). A normal car never does that: driving away, its
        box bottom rises but the box also shrinks. Own pitch (bumps) shifts all boxes equally -
        subtract the median shift."""
        pairs = {}
        for tid, h in self.hist.items():
            if self.last_seen.get(tid) != t or self.cls.get(tid) not in C.VEHICLE_CLASSES:
                continue
            old = next((p for p in h if t - p[0] <= C.CRASH_WINDOW), None)
            if old is None or old is h[-1] or t - old[0] < C.CRASH_WINDOW / 2:
                continue
            pairs[tid] = (old, h[-1])
        self.crash_rise = 0.0
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
            rise = -(((n[2] + n[4]) - (o[2] + o[4])) / 2 - ego) / h / dt       # box heights per second
            taller = math.log((h / w) / (h0 / w0)) / dt
            scale = math.log(w / w0) / dt
            if abs(scale) < C.CRASH_MAX_SCALE:
                self.crash_rise = max(self.crash_rise, min(rise, 3.0) * float(taller > 0))
            if rise >= C.CRASH_RISE and taller >= C.CRASH_ASPECT and abs(scale) < C.CRASH_MAX_SCALE:
                out.append(tid)
        return out

    # ---------- brake lights (experimental) ----------
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
            for store in (self.hist, self.cls, self.red_base, self.rates, self.first_seen):
                store.pop(tid, None)
