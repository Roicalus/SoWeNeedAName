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
  skid           - a car ahead suddenly turns its body and slides sideways (loss of control);
  aggressive     - a car that already made risky manoeuvres (hard braking, cut-in, weaving, frequent lane
                   changes) is still near us: the driver is told to keep an eye on it;
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
RULE_KINDS = ["ttc", "closing", "hardbrake", "cutin", "crossing", "drift", "ped", "ped_xwalk", "vru_near", "crash",
              "wrongway", "weave", "close", "skid", "aggressive"]
MARK_KINDS = {"hardbrake", "cutin", "crossing", "drift", "weave", "skid"}   # risky manoeuvres a car is remembered for
FEATURES += [f"rule_{k}" for k in RULE_KINDS]       # which rules fired and how loudly (level 0-3)
# per-threat features for the learned threat scorer (risk.py). Scale-free on purpose (TTC, lanes, seconds):
# metres and frame fractions depend on the camera and would let a model recognise the dataset
THREAT_FEATURES = [f"is_{k}" for k in RULE_KINDS] + [
    "level", "vehicle", "person", "two_wheeler", "side_view", "abs_u", "toward", "abs_du", "inv_ttc", "u_impact",
    "age", "jitter", "growth", "in_path", "edge", "n_threats"]
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
    feat: list | None = None    # THREAT_FEATURES for the learned threat scorer
    score: float | None = None  # threat scorer probability (None - no model)


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
    view: tuple = (C.CORRIDOR_TOP_Y, C.HOOD_Y)     # (horizon, hood) rows found by BehaviorEngine.calibrate
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


def corridor_polygon(W, H, view=None):
    """Own-lane trapezoid in pixels; view = (horizon, hood) from BehaviorEngine.calibrate."""
    top, bottom = view if view else (C.CORRIDOR_TOP_Y, 1.0)
    cx = C.CORRIDOR_CENTER_X
    pts = [(cx - C.CORRIDOR_TOP_HALF, top), (cx + C.CORRIDOR_TOP_HALF, top),
           (cx + C.CORRIDOR_BOTTOM_HALF, bottom), (cx - C.CORRIDOR_BOTTOM_HALF, bottom)]
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
        self.bottoms = deque()              # (t, bottom, width) of car boxes - where the road is in this camera
        self.top, self.hood = C.CORRIDOR_TOP_Y, C.HOOD_Y
        self.crash_rise = 0.0
        self.crit_hist = {}                 # track id -> times of critical TTC (for "physically certain")
        self.crash_hits = {}                # track id -> crash signs in the last analyses
        self.marks = defaultdict(list)      # track id -> [(t, kind)] risky manoeuvres this car made
        self.lane_idx = {}                  # track id -> (lane index, since) for counting its lane changes
        self.prev_active = set()            # alert kinds confirmed on the previous step
        self.dense = False                  # dense slow traffic around us
        self.last_level = {}                # alert kind -> last level its threats had

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
        # own speed from video (egomotion.py) is a rough estimate: it only feeds the risk model. Decisions
        # that depend on it ("we stand still", "that car is parked") need a trusted source - the vehicle
        # bus (CAN) passes ego = (speed, yaw, ok, True)
        trusted = ego_ok and len(ego) > 3 and bool(ego[3])
        stopped = trusted and ego[0] < 1.0
        aspect = W / H
        self._calibrate(t, W, H, detections)
        res.view = (self.top, self.hood)

        live = []
        bikes = [d.box for d in detections if d.cls in (C.BICYCLE_CLASS, 3)]
        for d in detections:
            x1, y1, x2, y2 = d.box
            if d.cls == C.PERSON_CLASS and any(self._rider(d.box, b) for b in bikes):
                continue                     # a rider: the motorbike / bicycle box represents them
            if (y1 + y2) / 2 / H > self.hood:
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
        yaw = self._yaw_rate(t, live)
        if ego_ok:
            yaw = ego[1]           # from far background (egomotion.py) - steadier than a few car boxes
        # while we turn sharply, sideways motion of cars in the frame is mostly our own rotation: lane-change
        # and crossing rules for cars then need an imminent collision (parked cars swing through the frame)
        turning = abs(yaw) > C.YAW_TURN
        # the sideways speed a car needs to count as moving towards us grows with our own turning: an error in
        # the turning estimate (fisheye lenses, few far points) makes parked cars "slide" in the frame
        slack = C.YAW_SLACK * abs(yaw)

        # dense, slow traffic (a jam): several cars right next to us. Lane nudges there are normal, so lane
        # changes and crossings are only green unless a collision is imminent; reputation is not counted
        close_cars = 0
        for d in live:
            if d.cls in C.VEHICLE_CLASSES:
                bx1, by1, bx2, by2 = self.hist[d.tid][-1][1:5]
                sv = (bx2 - bx1) * aspect > C.CROSS_MIN_ASPECT * (by2 - by1)
                dd = self._distance(d.cls, bx2 - bx1, by2 - by1, aspect, sv)
                close_cars += dd is not None and dd < C.DENSE_DIST
        self.dense = close_cars >= C.DENSE_CARS
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
            whole = x1 > C.SIDE_MARGIN and x2 < 1 - C.SIDE_MARGIN     # fully in view: cut boxes give wrong geometry
            ahead = by >= self.top
            in_lane = u is not None and abs(u) <= C.IN_LANE and ahead
            # in our path: by lateral offset AND (without lane markings) by where the box bottom sits in our
            # corridor - a car parked at the kerb seen half from the side has a wide box, so its offset in its
            # own widths looks small, but its centre is at the corridor edge
            in_corr = lanes_ok or abs((x1 + x2) / 2 - C.CORRIDOR_CENTER_X) <= self._half(by) * C.PATH_CORRIDOR_FRAC
            in_path = u is not None and abs(u) <= (C.IMPACT_U_CAR if vehicle else C.IMPACT_U) and ahead and in_corr
            in_lane = in_lane and in_corr
            toward = -math.copysign(1, u) * du if (u is not None and du is not None and u != 0) else 0.0
            age = t - self.first_seen[d.tid]
            # how fast the gap shrinks, m/s: danger is a fast approach, not just a small TTC while creeping in a jam
            closing_speed = dist / ttc if (dist is not None and ttc) else 0.0
            # standing still on the road: we close in on it exactly at our own speed (parked cars at the
            # kerb drift sideways in the frame as we pass them - they must not look like a cut-in)
            static = vehicle and trusted and ego[0] > C.STATIC_MIN_EGO and ttc is not None and dist is not None \
                and abs(dist / ttc - ego[0]) < C.STATIC_TOL * ego[0]
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
            if u is not None and du is not None and ahead and whole and not in_lane and abs(u) <= C.CUTIN_MAX_U \
                    and not static and age >= C.CUTIN_MIN_AGE:
                t_in = (abs(u) - C.IN_LANE) / toward if toward > 0 else None
                # dangerous only if we approach it (TTC) or it is already close: at traffic lights the
                # cross traffic and lane changes far ahead are not a threat
                closing_in = (ttc is not None and ttc <= C.CUTIN_TTC and closing_speed >= C.MIN_CLOSING) \
                    or (dist is not None and dist <= C.CRIT_DIST)          # a car cutting in right in front of us
                if vehicle and closing_in and dist <= C.CUTIN_MAX_DIST and toward >= C.CUTIN_MIN_SPEED + slack \
                        and t_in is not None and t_in <= C.CUTIN_HORIZON \
                        and (not turning or (ttc is not None and ttc < C.TTC_CRITICAL)):
                    # seen from the side (driving across - junction, U-turn): "crossing"; else a lane change
                    kind = "crossing" if side_view else "cutin"
                    crit = ttc is not None and ttc < C.CUTIN_CRIT_TTC and dist <= C.CUTIN_CRIT_DIST
                    found.append((kind, 3 if crit else 2))   # a helper, not an alarm: red only if imminent
                elif vehicle and not side_view and dist <= C.DRIFT_MAX_DIST and toward >= C.DRIFT_MIN_SPEED + slack \
                        and t_in is not None and t_in <= C.DRIFT_HORIZON and not stopped and not turning:
                    found.append(("drift", 1))
            # --- pedestrians and two-wheelers
            if not vehicle and dist is not None and ahead:
                lvl = 0
                kind = "ped"
                # a person matters only if we are driving towards them (their box grows: TTC). People on the
                # pavement, waiting at a crossing or walking in front of a car that stands still are not a hazard
                approaching = ttc is not None and ttc <= C.PED_TTC and closing_speed >= C.PED_MIN_CLOSING
                if in_path and dist <= C.VRU_NEAR and approaching:                  # on our path, we close in
                    lvl = 3 if ttc < C.TTC_CRITICAL and dist <= C.PED_RED_DIST else 2
                elif u is not None and du is not None and not in_lane and toward >= C.VRU_MIN_SPEED \
                        and abs(u) <= C.VRU_STEP_U and age >= C.CUTIN_MIN_AGE:
                    t_in = (abs(u) - C.IN_LANE) / toward
                    closing_in = ttc is not None and ttc <= C.CUTIN_TTC
                    if closing_in and dist <= C.VRU_DIST and t_in <= C.VRU_HORIZON \
                            and self._steady_u(h, t, 0.5, 9.0):                          # stepping into our lane
                        lvl = 3 if ttc is not None and ttc < C.CUTIN_CRIT_TTC else 2
                if not lvl and age <= C.VRU_SUDDEN_AGE and dist <= C.VRU_SUDDEN_DIST and inside and approaching \
                        and u is not None and abs(u) <= C.VRU_SUDDEN_U:            # appeared suddenly close
                    lvl = 2
                if not lvl and C.PED_XWALK_ALERT and self.xwalk.near((x1 + x2) / 2, by) and dist <= C.CROSSWALK_DIST:
                    kind, lvl = "ped_xwalk", 1                                   # at a crosswalk ahead
                if not lvl and C.VRU_NEAR_ALERT and u is not None and abs(u) <= C.VRU_EDGE_U and dist <= C.VRU_ATTENTION_DIST:
                    kind, lvl = "vru_near", 1                                    # close to the road edge
                if lvl and stopped and kind == "ped":
                    lvl = 1                  # we stand still: people crossing in front are not a risk yet
                if lvl:
                    found.append((kind, lvl))
                    if kind == "ped":
                        res.peds.append(d)
            # --- collision course: where will the object be sideways when we reach it? A constant bearing
            # means collision - u + du * TTC inside our track. Catches cars crossing right in front,
            # pedestrians running in, cars half in our path; cars ahead in our path are handled below
            if ttc is not None and u is not None and ahead and dist is not None and dist <= C.COURSE_MAX_DIST \
                    and ttc <= C.COURSE_TTC and not (vehicle and in_path and not side_view) and not static \
                    and closing_speed >= (C.MIN_CLOSING if vehicle else C.PED_MIN_CLOSING) \
                    and (whole or not vehicle) \
                    and not found and not (vehicle and turning and ttc >= C.TTC_CRITICAL):
                u_imp = u + (du or 0.0) * ttc
                # a car must really move towards our path - or be right across it (seen from the side, centred)
                moving_in = toward >= C.COURSE_MIN_TOWARD + slack or (side_view and in_path)
                if abs(u_imp) <= C.IMPACT_U and (du is not None or abs(u) <= C.IMPACT_U) \
                        and (moving_in or (not vehicle and (toward >= C.COURSE_MIN_TOWARD or in_path))):
                    kind = "ped" if not vehicle else ("crossing" if side_view else "cutin")
                    crit = ttc < C.CUTIN_CRIT_TTC and dist <= C.CUTIN_CRIT_DIST
                    lvl = 3 if crit else 2
                    if not vehicle and dist > C.PED_COURSE_DIST:
                        lvl = 0                    # far people: their boxes "breathe", TTC from height is unreliable
                    if stopped and not vehicle:
                        lvl = 1
                    if lvl:
                        found.append((kind, lvl))
                        if kind == "ped":
                            res.peds.append(d)
            # --- car in our path closing in fast
            if vehicle and in_path and inside and not side_view and ttc is not None and w >= C.TTC_MIN_WIDTH:
                closing_ms = dist / ttc if dist else 0
                if closing_ms > C.LEAD_MAX_CLOSING:
                    # closing faster than ~43 km/h: oncoming. Lane markings show it is in our lane - danger;
                    # without markings (curve, trapezoid) it is more likely a false "car ahead"
                    # wrong-way car: markings visible, the car steadily in the middle of our lane, well inside
                    # the frame and closing really fast (a parked car we pass can look "in lane" when the
                    # lane network takes a bike lane or a kerb for our lane)
                    if lanes_ok and ttc < C.WRONGWAY_TTC and abs(u) <= C.WRONGWAY_U and closing_ms > C.WRONGWAY_CLOSING \
                            and x1 > 0.05 and x2 < 0.95 and self._steady_u(h, t):
                        found.append(("wrongway", 3 if ttc < C.CUTIN_CRIT_TTC else 2))
                    elif d is lead:
                        res.oncoming = True
                elif ttc < C.TTC_WARNING and closing_ms >= C.MIN_CLOSING:
                    crit = ttc < C.TTC_CRITICAL and w >= C.CRITICAL_MIN_WIDTH and closing_ms >= C.BRAKE_MIN_CLOSING
                    if crit:
                        self.crit_hist.setdefault(d.tid, deque()).append(t)
                    found.append(("ttc", 3) if crit else ("closing", 2))

            # --- skid / loss of control: a car ahead suddenly turns its body (box proportions change fast while
            # it slides sideways) - it may spin into our path
            if C.SKID and vehicle and ahead and inside and u is not None and abs(u) <= C.SKID_MAX_U and dist is not None \
                    and dist <= C.SKID_MAX_DIST and age >= C.CUTIN_MIN_AGE:
                spin = self._spin(h, t)
                if spin >= C.SKID_ASPECT_RATE and du is not None and abs(du) >= C.SKID_MIN_DU:
                    close = dist <= C.SKID_CRIT_DIST or (ttc is not None and ttc < C.TTC_WARNING)
                    found.append(("skid", 3 if close else 2))
            if self.dense:          # a jam: lane nudges one step quieter (red -> yellow, yellow -> green)
                found = [(k, max(1, lv - 1) if k in ("cutin", "crossing", "drift") else lv) for k, lv in found]
            for kind, level in found:
                th = Threat(d, kind, level, u, du, ttc, dist, self._predict(h, u, du, yaw))
                th.feat = self._threat_feat(th, h, t, side_view, in_path, inside, age)
                threats.append(th)
                raw.add(kind)
            res.tracks.append((d, status + "".join(f" {k.upper()}" for k, _ in found)))

        # --- crash of any car in view: keep the alarm CRASH_HOLD seconds even if the track is lost
        hits = set(self._crashes(t))
        for tid in set(self.crash_hits) | hits:
            q = self.crash_hits.setdefault(tid, deque(maxlen=3))
            q.append(tid in hits)
            if sum(q) >= C.CRASH_HITS:         # crash signs in 2 of the last 3 analyses: not a single box jump
                self.crash_at[tid] = t
        self.crash_hits = {k: q for k, q in self.crash_hits.items() if any(q)}
        self.crash_at = {k: v for k, v in self.crash_at.items() if t - v <= C.CRASH_HOLD}
        if self.crash_at:
            raw.add("crash")
            res.crashed = [d for d in live if d.tid in self.crash_at]
            threats += [Threat(d, "crash", 3 if d.box[2] - d.box[0] >= C.CRASH_CRIT_WIDTH * W else 2,
                               None, None, None, None) for d in res.crashed]

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
                    and rate >= C.BRAKE_RATE and jerk >= C.BRAKE_JERK and res.ttc is not None and res.ttc < C.BRAKE_MAX_TTC:
                raw.add("hardbrake")
                crit = res.ttc is not None and res.ttc < C.BRAKE_CRIT_TTC
                threats.append(Threat(lead, "hardbrake", 3 if crit else 2, h[-1][5], None, res.ttc, res.distance))
            # "too close" only if also closing in: standing bumper to bumper in a jam is fine
            if w >= C.CLOSE_WIDTH and res.ttc is not None and res.ttc < C.CLOSE_MAX_TTC:
                raw.add("close")
                threats.append(Threat(lead, "close", 1, h[-1][5], None, res.ttc, res.distance))
            # no weaving from lane markings: their jitter is above the amplitude threshold
            weaving, res.weave = self._weaving(h, t, stats=True)
            if weaving and C.WEAVE:
                raw.add("weave")
                threats.append(Threat(lead, "weave", 2, h[-1][5], None, res.ttc, res.distance))
            if C.BRAKE_LIGHTS and frame is not None and self._brake_lights(frame, lead):
                raw.add("brakelight")
                threats.append(Threat(lead, "brakelight", 1, h[-1][5], None, res.ttc, res.distance))

        self._reputation(t, live, threats, raw, aspect)
        F["crash"] = float(bool(self.crash_at))
        F["crash_rise"] = self.crash_rise
        F["oncoming"] = float(res.oncoming or "wrongway" in raw)
        F["weave"] = float("weave" in raw)
        for th in threats:
            key = f"rule_{th.kind}"
            if key in F:
                F[key] = max(F[key], float(th.level))
            if th.feat is None:                      # threats on the car ahead / crashes
                h = self.hist.get(th.det.tid)
                if h:
                    x1, y1, x2, y2 = h[-1][1:5]
                    sv = (x2 - x1) * aspect > C.CROSS_MIN_ASPECT * (y2 - y1)
                    th.feat = self._threat_feat(th, h, t, sv, True, True, t - self.first_seen.get(th.det.tid, t))
        res.features = F
        # learned threat scorer (risk.py): keeps a rule's yellow / red only if this particular threat looks
        # like the ones before real crashes; otherwise it is shown green ("attention") - still on screen,
        # but no alarm. Physically certain cases (an imminent collision with a close car) are never lowered.
        for th in threats:
            if th.feat is not None:
                th.feat[THREAT_FEATURES.index("n_threats")] = float(len(threats))
            th.score = self.risk.threat_score(th.feat) if th.feat is not None else None
            if th.score is None or self._certain(th, t):
                continue
            if th.level >= 3 and th.score < self.risk.gate_red:
                th.level = 2
            if th.level >= 2 and th.score < self.risk.gate_yellow:
                th.level = 1
        p = self.risk(t, F)                  # scene model: does this look like the seconds before a crash
        if p is not None:
            if p >= self.risk.threshold:
                raw.add("danger")
                lvl = 3 if p >= self.risk.threshold_red else 2
                target = lead or (max(live, key=lambda d: d.box[3]) if live else None)
                if target is not None:
                    threats.append(Threat(target, "danger", lvl, None, None, None, None))
        res.raw, res.threats = raw, threats
        res.alerts = self._confirm(t, raw, threats)
        res.risk = p if p is not None else max((th.level for th in threats), default=0) / 3
        return res

    @staticmethod
    def _rider(p, b):
        """Person box p sits on a two-wheeler box b (most of the person's lower half overlaps it)."""
        ix = max(0.0, min(p[2], b[2]) - max(p[0], b[0]))
        iy = max(0.0, min(p[3], b[3]) - max((p[1] + p[3]) / 2, b[1]))
        return ix * iy >= 0.4 * (p[2] - p[0]) * (p[3] - p[1]) / 2

    def _half(self, y):
        """Half-width of the own-lane corridor at row y (frame fractions), between the calibrated horizon and hood."""
        k = max(0.0, min(1.0, (y - self.top) / max(self.hood - self.top, 1e-6)))
        return C.CORRIDOR_TOP_HALF + k * (C.CORRIDOR_BOTTOM_HALF - C.CORRIDOR_TOP_HALF)

    def _calibrate(self, t, W, H, detections):
        """Where the road is in this camera. Dashcams differ a lot: some see the horizon at 0.3 of the
        height and a hood covering the bottom 40%. Far cars stand on the horizon: the low percentile of
        small car boxes' bottoms gives it; nothing is ever seen below the hood, so the high percentile of
        all car bottoms bounds it."""
        for d in detections:
            x1, y1, x2, y2 = d.box
            if d.cls in C.VEHICLE_CLASSES and (x2 - x1) / W < 0.5 and y2 / H < 0.985:
                self.bottoms.append((t, y2 / H, (x2 - x1) / W))
        while self.bottoms and t - self.bottoms[0][0] > C.CALIB_WINDOW:
            self.bottoms.popleft()
        if len(self.bottoms) < C.CALIB_MIN:
            return
        b = np.array([(y, w) for _, y, w in self.bottoms])
        far = b[b[:, 1] < C.CALIB_FAR_WIDTH, 0]
        if len(far) >= 8:
            top = float(np.clip(np.percentile(far, 15) - 0.015, C.CALIB_TOP_RANGE[0], C.CALIB_TOP_RANGE[1]))
            self.top = 0.9 * self.top + 0.1 * top
        if self.bottoms[-1][0] - self.bottoms[0][0] >= C.CALIB_MIN_SPAN:
            low = float(np.percentile(b[:, 0], 98)) + 0.05
            hood = C.HOOD_Y if low >= C.HOOD_Y else max(low, self.top + 0.3)
            self.hood = 0.9 * self.hood + 0.1 * hood

    def _reputation(self, t, live, threats, raw, aspect):
        """Remember cars that made risky manoeuvres (and cars changing lanes again and again) and, while such
        a car is still close, add an "aggressive" threat on it: green after one episode, yellow after more."""
        for th in threats:
            # only manoeuvres that were confirmed alerts (would have been shown to the driver) mark a car
            if th.kind in MARK_KINDS and th.kind in self.prev_active and th.level >= 2:
                m = self.marks[th.det.tid]
                if not m or t - m[-1][0] > C.MARK_GAP or m[-1][1] != th.kind:
                    m.append((t, th.kind))
        threatened = {th.det.tid for th in threats}
        if self.dense:
            return
        for d in live:
            if d.cls not in C.VEHICLE_CLASSES:
                continue
            h = self.hist[d.tid]
            u = h[-1][5]
            if u is not None and abs(u - round(u)) <= C.LANE_DEEP:   # lane changes: deep inside a new lane, kept
                lane = int(round(u))
                prev = self.lane_idx.get(d.tid)
                if prev is None:
                    self.lane_idx[d.tid] = (lane, t)
                elif lane != prev[0]:
                    if t - prev[1] >= C.LANE_STAY:
                        self.marks[d.tid].append((t, "lanechange"))
                    self.lane_idx[d.tid] = (lane, t)
            m = [k for tt, k in self.marks.get(d.tid, []) if t - tt <= C.MARK_WINDOW and k in C.MARK_COUNTS]
            score = sum(1.0 if k != "lanechange" else C.LANECHANGE_WEIGHT for k in m)
            if score < 1.0 or d.tid in threatened:
                continue
            x1, y1, x2, y2 = h[-1][1:5]
            sv = (x2 - x1) * aspect > C.CROSS_MIN_ASPECT * (y2 - y1)
            dist = self._distance(d.cls, x2 - x1, y2 - y1, aspect, sv)
            if dist is None or dist > C.MARK_DIST or y2 < self.top:
                continue
            level = 2 if score >= C.MARK_YELLOW else 1
            threats.append(Threat(d, "aggressive", level, u, None, None, dist, None))
            raw.add("aggressive")
        self.marks = defaultdict(list, {k: v for k, v in self.marks.items() if k in self.hist})
        self.lane_idx = {k: v for k, v in self.lane_idx.items() if k in self.hist}

    def aggressive_count(self, tid, t):
        """Risky manoeuvres a car made in the last MARK_WINDOW seconds (for the on-screen tag)."""
        return sum(1 for tt, _ in self.marks.get(tid, []) if t - tt <= C.MARK_WINDOW)

    def _steady_u(self, h, t, span=0.5, limit=None):
        """The track exists for `span` seconds and stayed within `limit` lanes of our lane centre
        (default WRONGWAY_U)."""
        limit = C.WRONGWAY_U if limit is None else limit
        pts = [p[5] for p in h if t - p[0] <= span and p[5] is not None]
        return len(pts) >= 4 and max(abs(v) for v in pts) <= limit and h[-1][0] - h[0][0] >= span

    @staticmethod
    def _spin(h, t):
        """How fast the box proportions change, |d ln(w/h)/dt| over KIN_WINDOW: a car turning its body."""
        pts = [(p[0], math.log(max(p[3] - p[1], 1e-6) / max(p[4] - p[2], 1e-6))) for p in h if t - p[0] <= C.KIN_WINDOW]
        if len(pts) < 4 or pts[-1][0] - pts[0][0] < C.KIN_MIN_SPAN:
            return 0.0
        ts, a = np.array(pts).T
        return float(abs(np.polyfit(ts, a, 1)[0]))

    def _certain(self, th, t):
        """Physically certain imminent collision: a close car in our path with TTC < TTC_CERTAIN, critical for
        at least CERTAIN_HOLD seconds (one jumpy box is not enough) - never lowered by the threat scorer."""
        if th.kind != "ttc" or th.ttc is None or th.ttc >= C.TTC_CERTAIN:
            return False
        q = self.crit_hist.get(th.det.tid, ())
        return len(q) >= 3 and t - q[0] >= C.CERTAIN_HOLD

    def _threat_feat(self, th, h, t, side_view, in_path, inside, age):
        """THREAT_FEATURES of one threat (scale-free: lanes, seconds, 1/TTC)."""
        u, du, ttc = th.u, th.du, th.ttc
        cls = self.cls.get(th.det.tid, th.det.cls)
        pts = [(p[0], p[5]) for p in h if t - p[0] <= 1.0 and p[5] is not None]
        jitter = 1.0
        if len(pts) >= 4:
            ts, us = np.array(pts).T
            jitter = float(np.std(us - np.polyval(np.polyfit(ts, us, 1), ts)))
        rate, _, _ = self._rate(h, t, by_height=side_view or cls not in C.VEHICLE_CLASSES)
        toward = -math.copysign(1, u) * du if (u is not None and du is not None and u != 0) else 0.0
        u_imp = abs(u + (du or 0.0) * ttc) if (u is not None and ttc is not None) else 3.0
        f = [float(th.kind == k) for k in RULE_KINDS] + [
            float(th.level), float(cls in C.VEHICLE_CLASSES), float(cls == C.PERSON_CLASS), float(cls in (1, 3)),
            float(side_view), min(abs(u), 3.0) if u is not None else 3.0, float(np.clip(toward, -3, 3)),
            min(abs(du), 3.0) if du is not None else 0.0, min(1 / ttc, 5.0) if ttc else 0.0, min(u_imp, 3.0),
            min(age, 5.0), min(jitter, 1.0), float(np.clip(rate or 0.0, -2, 2)), float(in_path), float(not inside), 0.0]
        return f

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
                  if t - self.last_raw[k] <= C.ALERT_HOLD + 1e-6
                  and self.last_raw[k] - self.since[k] >= C.CONFIRM_SEC[k] - 1e-6}   # float times: 7.0 - 6.9 < 0.1
        if "ttc" in active:
            active.discard("closing")   # critical replaces ordinary closing
        if "cutin" in active:
            active.discard("drift")
        self.prev_active = set(active)
        alerts = []
        for k in active:
            base, screen, voice = C.ALERTS[k]
            levels = [th.level for th in threats if th.kind == k]
            if levels:
                self.last_level[k] = max(levels)
            # while an alert is only held (its threat blinked out for a moment) keep the last real level
            alerts.append(Alert(k, self.last_level.get(k, base), screen, voice))
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
        if y2 < self.top:
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
                or (n[2] + n[4]) / 2 > self.hood
            young = t - self.first_seen.get(tid, t) < C.CRASH_MIN_AGE      # new tracks jump (exposure, occlusion)
            if cut or young or w < C.CRASH_MIN_WIDTH or h <= 0 or h0 <= 0:
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
        for q in self.crit_hist.values():
            while q and t - q[0] > 1.0:
                q.popleft()
        for tid in [k for k, seen in self.last_seen.items() if t - seen > C.TRACK_TTL]:
            del self.last_seen[tid]
            for store in (self.hist, self.cls, self.red_base, self.rates, self.first_seen):
                store.pop(tid, None)
