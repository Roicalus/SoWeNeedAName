"""Logic tests on synthetic tracks (no model, no video).

    python test_behavior.py
"""
import math

import numpy as np

import config as C
from advisor import Advisor
from behavior import Alert, BehaviorEngine, Detection, FrameResult, Threat
from tracker import Tracker

W, H, FPS = 1280, 720, 30


def car(cx, width, tid=1, bottom=0.75):
    """Car: centre cx and width as frame fractions."""
    w = width * W
    x1, y2 = cx * W - w / 2, bottom * H
    return Detection((x1, y2 - 0.8 * w, x1 + w, y2), 2, tid)


def run(frames, ego=None, crosswalk=None):
    """frames: list of detection lists; returns ({kind: level}, ttc) per frame."""
    eng = BehaviorEngine()
    out = []
    for i, dets in enumerate(frames):
        kw = {} if crosswalk is None else {"crosswalk": crosswalk}
        res = eng.update(i / FPS, W, H, dets, None, None, ego, **kw)
        out.append(({a.kind: a.level for a in res.alerts}, res.ttc))
    return out


def seen(out, kind, level=None):
    return any(kind in kinds and (level is None or kinds[kind] == level) for kinds, _ in out)


def test_steady_car_no_alerts():
    out = run([[car(0.5, 0.15)] for _ in range(150)])
    assert not any(kinds for kinds, _ in out), "a car at a constant distance must not raise alerts"


def test_ttc_estimate():
    # real TTC: distance Z(t) = Z0 - v t, width ~ 1/Z; with Z0 = 20 m and v = 10 m/s TTC at t is 2 - t
    frames = [[car(0.5, 0.10 * 20 / (20 - 10 * i / FPS))] for i in range(int(1.2 * FPS))]
    out = run(frames)
    ttc = out[-1][1]
    expected = 2 - (len(frames) - 1) / FPS
    assert ttc is not None and abs(ttc - expected) < 0.25, f"TTC {ttc} instead of ~{expected:.2f}"
    assert seen(out, "ttc", 3), "the critical (red) alert must fire"


def test_slow_approach_no_alerts():
    frames = [[car(0.5, 0.10 * 40 / (40 - 2 * i / FPS))] for i in range(90)]    # TTC > 10 s
    out = run(frames)
    assert not seen(out, "closing") and not seen(out, "ttc") and not seen(out, "hardbrake")


def test_hard_braking_ahead():
    """We follow a car at 15 m with the same speed; at 2 s it brakes at 7 m/s^2."""
    frames = []
    for i in range(int(3.5 * FPS)):
        t = i / FPS
        tb = max(0.0, t - 2.0)
        z = 15 - 0.5 * 7 * tb * tb                  # gap shrinks as the car ahead decelerates
        frames.append([car(0.5, 0.10 * 15 / max(z, 3))])
    out = run(frames)
    first = next((i for i, (k, _) in enumerate(out) if "hardbrake" in k), None)
    assert first is not None, "hard braking ahead must be detected"
    assert first < int(3.2 * FPS), f"hard braking detected too late (frame {first})"
    assert not any("hardbrake" in k for k, _ in out[:int(2.0 * FPS)]), "no alert before the braking"


def test_weaving_detected():
    if not C.WEAVE:
        return
    frames = [[car(0.5 + 0.3 * 0.12 * math.sin(2 * math.pi * i / FPS / 2), 0.12)] for i in range(150)]
    assert seen(run(frames), "weave")


def test_lane_change_not_weaving():
    def x(i):
        k = min(max((i / FPS - 1) / 2, 0), 1)
        return 0.5 + 0.12 * (1 - math.cos(math.pi * k)) / 2
    frames = [[car(x(i), 0.12)] for i in range(150)]
    assert not seen(run(frames), "weave")


def test_truck_crossing_in_front():
    """A truck seen from the side drives across our path 12 m ahead and we close in: collision course."""
    frames = []
    for i in range(45):
        t = i / FPS
        z = 12 - 8 * t                                   # we approach at 8 m/s
        h = 0.25 * 12 / z                                # box height ~ 1 / distance
        cx = 0.62 - 0.12 * t                             # moves towards the centre
        w = 2.6 * h * H / W
        frames.append([Detection(((cx - w / 2) * W, 0.75 * H - h * H, (cx + w / 2) * W, 0.75 * H), 7, 3)])
    out = run(frames)
    assert seen(out, "crossing"), "a truck crossing right in front must raise an alert"


def test_pedestrian_on_sidewalk_ignored():
    ped = Detection((50, 400, 110, 560), C.PERSON_CLASS, 7)
    out = run([[ped] for _ in range(10)])
    assert not seen(out, "ped")


def ped(cx, height, tid=7, bottom=0.78):
    """Pedestrian box: centre cx (frame fraction), height (frame fraction)."""
    hp = height * H
    return Detection((cx * W - 0.18 * hp, bottom * H - hp, cx * W + 0.18 * hp, bottom * H), C.PERSON_CLASS, tid)


def test_pedestrian_on_road_while_approaching():
    """A person on our path 20 m ahead and we drive towards them at 8 m/s: warn in advance."""
    out = run([[ped(0.5, 0.12 * 20 / (20 - 8 * i / FPS))] for i in range(40)])
    assert seen(out, "ped", 2) or seen(out, "ped", 3)
    first = next(i for i, (k, _) in enumerate(out) if k.get("ped", 0) >= 2)
    assert first < 25, f"warned too late (frame {first})"


def test_pedestrian_in_front_of_standing_car_no_alert():
    """We stand at a crossing; people walk across right in front: no alert (the driver sees them)."""
    out = run([[ped(0.3 + 0.01 * i, 0.25)] for i in range(40)])
    assert not seen(out, "ped")


def test_pedestrian_on_sidewalk_no_alert():
    """A person standing on the pavement / waiting at a crosswalk while we drive past: no alert."""
    def frame(i):
        k = 25 / (25 - 8 * i / FPS)                       # we approach: everything grows as 1 / distance,
        return [ped(0.5 + 0.12 * k, 0.10 * k)]           # including the person's offset from the centre
    frames = [frame(i) for i in range(40)]
    out = run(frames, crosswalk=(0.2, 0.6, 0.8, 0.68))
    assert not any(set(k) & {"ped", "ped_xwalk", "vru_near"} for k, _ in out)


def test_sudden_pedestrian():
    """A pedestrian steps out from behind a parked van right in front while we drive: warn at once."""
    frames = [[] for _ in range(10)] + [[ped(0.53, 0.20 * 14 / (14 - 8 * i / FPS), tid=9)] for i in range(25)]
    out = run(frames)
    assert seen(out, "ped", 2) or seen(out, "ped", 3)


def test_pedestrian_stepping_into_lane():
    """A pedestrian walks from the kerb into our lane while we approach: warn before they are in it."""
    frames = []
    for i in range(30):
        h = 0.16 * 15 / (15 - 6 * i / FPS)
        frames.append([ped(0.78 - 0.012 * i, h)])
    out = run(frames)
    assert seen(out, "ped", 2) or seen(out, "ped", 3)


def test_car_in_other_lane_ignored():
    frames = [[car(0.08, 0.10 * 20 / (20 - 10 * i / FPS))] for i in range(36)]
    out = run(frames)
    assert not seen(out, "ttc") and not seen(out, "closing")


def test_parked_car_passing_ignored():
    def det(i):
        w = 0.10 * 20 / (20 - 10 * i / FPS)
        return [car(0.5 + 1.5 * w, w, bottom=0.6 + w)]
    out = run([det(i) for i in range(36)])
    assert not seen(out, "ttc") and not seen(out, "closing")


def _crash_frames(width):
    """A car in the next lane "takes off" from second 2: its box jumps up and stretches, the width stays."""
    frames = []
    for i in range(90):
        k = max(0, i - 60) / FPS
        w, x1, y2 = width * W, 0.72 * W, 0.62 * H - 1.0 * 0.07 * H * k / 0.3
        h = 0.8 * w * (1 + 2.0 * k)
        frames.append([Detection((x1, y2 - h, x1 + w, y2), 2, 7), car(0.3, 0.05, tid=8, bottom=0.6)])
    return frames


def test_crash_in_other_lane_detected():
    out = run(_crash_frames(0.12))
    assert not any("crash" in kinds for kinds, _ in out[:60]), "no alarm before the crash"
    assert seen(out[60:], "crash", 3), "a crash of a close neighbouring car must be red"


def test_far_crash_is_yellow():
    out = run(_crash_frames(0.07))
    assert seen(out[60:], "crash", 2) and not seen(out, "crash", 3)


def test_box_stretched_into_shadow_not_crash():
    """Under a bridge the detector merges the car with the shadow above it: the box top shoots up,
    the bottom stays on the road. That is not a car leaving the ground."""
    frames = []
    for i in range(90):
        k = max(0, i - 60) / FPS
        w, x1, y2 = 0.12 * W, 0.6 * W, 0.62 * H
        h = 0.8 * w * (1 + 4.0 * k)
        frames.append([Detection((x1, y2 - h, x1 + w, y2), 2, 7), car(0.3, 0.05, tid=8, bottom=0.6)])
    assert not seen(run(frames), "crash")


def test_car_driving_away_not_crash():
    frames = []
    for i in range(90):
        width = 0.2 * math.exp(-0.8 * i / FPS)
        frames.append([car(0.75, width, tid=3, bottom=0.45 + 1.2 * width)])
    assert not seen(run(frames), "crash")


def test_cut_in_detected():
    """A car from the next lane (2 widths offset, ~9 m ahead) moves in front of us within a second."""
    frames = []
    for i in range(60):
        k = min(max((i / FPS - 0.7) / 1.0, 0), 1)
        width = 0.14
        X = 2.0 * (1 - k)
        frames.append([car(0.5 + X * width, width, tid=4, bottom=0.62 + width),
                       car(0.2, 0.05, tid=5, bottom=0.55), car(0.8, 0.05, tid=6, bottom=0.55)])
    out = run(frames)
    assert seen(out[:45], "cutin") or seen(out[:45], "crossing"), "a cut-in must be caught before it is in our lane"


def test_slow_drift_is_green():
    """A car 25 m ahead in the next lane slowly drifts towards ours (0.4 lanes/s): attention only."""
    frames = []
    for i in range(75):
        width = 0.05
        X = 2.0 - 0.8 * i / FPS                       # car widths; 0.8 widths/s ~ 0.4 lanes/s
        frames.append([car(0.5 + X * width, width, tid=4, bottom=0.55),
                       car(0.2, 0.05, tid=5, bottom=0.55), car(0.8, 0.05, tid=6, bottom=0.55)])
    out = run(frames)
    assert seen(out, "drift", 1) and not seen(out, "cutin")


def test_neighbor_lane_steady_not_cut_in():
    frames = [[car(0.5 + 2.0 * 0.12, 0.12, tid=4, bottom=0.74), car(0.2, 0.05, tid=5, bottom=0.55),
               car(0.8, 0.05, tid=6, bottom=0.55)] for _ in range(90)]
    out = run(frames)
    assert not any(set(kinds) & {"cutin", "crossing", "drift"} for kinds, _ in out)


def test_skid_ahead_detected():
    """A car 20 m ahead loses control: it slides sideways while its body turns (box gets much wider)."""
    if not C.SKID:
        return
    frames = []
    for i in range(60):
        t = i / FPS
        k = max(0.0, t - 1.0)                          # skid from second 1
        w0 = 0.09
        w = w0 * (1 + 1.2 * k)                          # body turns: the box widens, the height stays
        hp = 0.8 * w0 * W                               # box height in pixels
        cx = 0.5 + 0.12 * k
        frames.append([Detection(((cx - w / 2) * W, 0.62 * H - hp, (cx + w / 2) * W, 0.62 * H), 2, 5)])
    out = run(frames)
    assert seen(out, "skid"), "a car spinning ahead must raise an alert"
    assert not any("skid" in k for k, _ in out[:30]), "no alert while it drives straight"


def test_parked_car_while_turning_no_cutin():
    """We turn (whole background shifts right) past a parked car at the kerb: no cut-in / crossing."""
    frames = []
    for i in range(45):
        t = i / FPS
        cx = 0.25 + 0.3 * t                          # the car sweeps through the frame because of our turn
        frames.append([car(cx, 0.12, tid=2, bottom=0.72)])
    out = run(frames, ego=(5.0, 0.3, True))          # yaw 0.3 frame widths/s from the background flow
    assert not any(set(k) & {"cutin", "crossing", "drift"} for k, _ in out)


def test_aggressive_driver_remembered():
    """A car 15 m ahead hops between lanes again and again: we are told to watch it (green, then yellow)."""
    frames = []
    for i in range(int(9 * FPS)):
        t = i / FPS
        lane = [0, 1, 0, -1, 0, 1][min(int(t / 1.5), 5)]       # a new lane every 1.5 s
        width = 0.1
        X = lane * 3.5 / 1.8                                   # lane offset in car widths
        frames.append([car(0.5 + X * width, width, tid=4, bottom=0.62),
                       car(0.15, 0.04, tid=5, bottom=0.55), car(0.85, 0.04, tid=6, bottom=0.55)])
    out = run(frames)
    assert seen(out, "aggressive", 1), "a car changing lanes again and again must be flagged (green)"
    assert not seen(out, "aggressive", 3)


def test_steady_neighbour_not_aggressive():
    frames = [[car(0.5 + 2.0 * 0.1, 0.1, tid=4, bottom=0.62)] for _ in range(int(8 * FPS))]
    assert not seen(run(frames), "aggressive")


def _res(kind, level, box=(900, 300, 1000, 400)):
    det = Detection(box, 0, 1)
    r = FrameResult()
    r.alerts = [Alert(kind, level, C.ALERTS[kind][1], C.ALERTS[kind][2])]
    r.threats = [Threat(det, kind, level, None, None, None, None)]
    return r


def test_advisor_red_always_speaks_yellow_has_budget():
    adv = Advisor()
    a1 = adv.update(0.0, _res("cutin", 2), W)
    a2 = adv.update(1.0, _res("ped", 2), W)               # within HINT_GAP[2]: silent glow
    a3 = adv.update(1.5, _res("ttc", 3), W)               # red: no budget
    assert a1.cue and a1.cue["phrase"].endswith("on the right") and a1.side == "right"
    assert a2.cue is None and a2.level == 2 and a2.glow > 0
    assert a3.cue and a3.cue["level"] == 3


def test_advisor_same_phrase_cooldown():
    adv = Advisor()
    assert adv.update(0.0, _res("ttc", 3), W).cue
    assert adv.update(1.0, _res("ttc", 3), W).cue is None              # VOICE_COOLDOWN["ttc"] = 2.5 s
    assert adv.update(3.0, _res("ttc", 3), W).cue


def test_tracker_keeps_id():
    tr = Tracker()
    ids = set()
    for i in range(30):
        x = 500 + 3 * i
        dets = np.array([[x, 300, x + 150, 420, 0.9 if i % 5 else 0.2, 2]])   # every 5th detection is weak
        for _, _, tid in tr.update(dets):
            ids.add(tid)
    assert ids == {1}, f"one id expected, got {ids}"


def test_tracker_car_truck_same_track():
    tr = Tracker()
    ids = set()
    for i in range(10):
        dets = np.array([[500, 300, 650, 420, 0.8, 2 if i % 2 else 7]])     # car / truck alternating
        for _, _, tid in tr.update(dets):
            ids.add(tid)
    assert ids == {1}


if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK    {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
