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
    frames = [[car(0.5 + 0.3 * 0.12 * math.sin(2 * math.pi * i / FPS / 2), 0.12)] for i in range(150)]
    assert seen(run(frames), "weave")


def test_lane_change_not_weaving():
    def x(i):
        k = min(max((i / FPS - 1) / 2, 0), 1)
        return 0.5 + 0.12 * (1 - math.cos(math.pi * k)) / 2
    frames = [[car(x(i), 0.12)] for i in range(150)]
    assert not seen(run(frames), "weave")


def test_pedestrian_in_lane():
    ped = Detection((600, 400, 660, 560), C.PERSON_CLASS, 7)
    assert seen(run([[ped] for _ in range(10)]), "ped")


def test_pedestrian_in_lane_while_stopped_is_green():
    """We stand at a crossing (own speed 0): people walking in front are attention, not risk."""
    ped = Detection((600, 400, 660, 560), C.PERSON_CLASS, 7)
    out = run([[ped] for _ in range(10)], ego=(0.0, 0.0, True))
    assert seen(out, "ped", 1) and not seen(out, "ped", 2) and not seen(out, "ped", 3)


def test_pedestrian_on_sidewalk_ignored():
    ped = Detection((50, 400, 110, 560), C.PERSON_CLASS, 7)
    out = run([[ped] for _ in range(10)])
    assert not seen(out, "ped")


def test_sudden_pedestrian():
    """A pedestrian appears from behind a parked van, close and next to our lane: warn at once."""
    frames = [[] for _ in range(10)] + [[Detection((830, 330, 900, 560), C.PERSON_CLASS, 9)] for _ in range(10)]
    out = run(frames)
    assert seen(out, "ped", 2), "a pedestrian that suddenly appears close must be yellow"


def test_pedestrian_at_crosswalk_is_green():
    """A pedestrian standing at the edge of a crosswalk 20 m ahead: attention."""
    ped = Detection((880, 300, 905, 370), C.PERSON_CLASS, 5)
    out = run([[ped] for _ in range(30)], crosswalk=(0.2, 0.48, 0.8, 0.53))
    assert seen(out, "ped_xwalk", 1)


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


def test_crash_in_other_lane_detected():
    """A car in the next lane "takes off": its box jumps up and stretches, the width stays."""
    frames = []
    for i in range(90):
        k = max(0, i - 60) / FPS                  # crash from second 2
        w, x1, y2 = 0.09 * W, 0.75 * W, 0.62 * H - 1.0 * 0.07 * H * k / 0.3
        h = 0.8 * w * (1 + 2.0 * k)
        frames.append([Detection((x1, y2 - h, x1 + w, y2), 2, 7), car(0.3, 0.05, tid=8, bottom=0.6)])
    out = run(frames)
    assert not any("crash" in kinds for kinds, _ in out[:60]), "no alarm before the crash"
    assert seen(out[60:], "crash", 3), "a crash of a neighbouring car must be red"


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


def test_pedestrian_stepping_into_lane():
    frames = []
    for i in range(30):
        x = 900 - 12 * i                                       # ~1.3 m/s towards the centre at 160 px height
        frames.append([Detection((x, 400, x + 60, 560), C.PERSON_CLASS, 7)])
    out = run(frames)
    first = next(i for i, (kinds, _) in enumerate(out) if kinds.get("ped", 0) >= 2)
    assert first < 25, f"pedestrian noticed late (frame {first})"


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
