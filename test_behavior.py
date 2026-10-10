"""Проверка логики на синтетических треках (без модели и видео).

    python test_behavior.py
"""
import math

import numpy as np

import config as C
from behavior import BehaviorEngine, Detection
from tracker import Tracker

W, H, FPS = 1280, 720, 30


def car(cx, width, tid=1, bottom=0.75):
    """Машина: центр cx и ширина в долях кадра."""
    w = width * W
    x1, y2 = cx * W - w / 2, bottom * H
    return Detection((x1, y2 - 0.8 * w, x1 + w, y2), 2, tid)


def run(frames):
    """frames: список списков детекций; возвращает типы предупреждений по кадрам."""
    eng = BehaviorEngine()
    out = []
    for i, dets in enumerate(frames):
        res = eng.update(i / FPS, W, H, dets)
        out.append(({a.kind for a in res.alerts}, res.ttc))
    return out


def seen(out, kind):
    return any(kind in kinds for kinds, _ in out)


def test_steady_car_no_alerts():
    out = run([[car(0.5, 0.15)] for _ in range(150)])
    assert not any(kinds for kinds, _ in out), "машина на постоянной дистанции не должна вызывать тревог"


def test_ttc_estimate():
    # реальный TTC: расстояние Z(t) = Z0 - v t, ширина ~ 1/Z; при Z0=20 м и v=10 м/с TTC в момент t равен 2 - t
    frames = [[car(0.5, 0.10 * 20 / (20 - 10 * i / FPS))] for i in range(int(1.2 * FPS))]
    out = run(frames)
    ttc = out[-1][1]
    expected = 2 - (len(frames) - 1) / FPS
    assert ttc is not None and abs(ttc - expected) < 0.25, f"TTC {ttc} вместо ~{expected:.2f}"
    assert seen(out, "ttc"), "должно сработать критическое предупреждение"


def test_slow_approach_no_alerts():
    # медленное сближение: TTC > 10 с
    frames = [[car(0.5, 0.10 * 40 / (40 - 2 * i / FPS))] for i in range(90)]
    out = run(frames)
    assert not seen(out, "closing") and not seen(out, "ttc")


def test_weaving_detected():
    # колебания ±0.3 ширины машины с периодом 2 с
    frames = [[car(0.5 + 0.3 * 0.12 * math.sin(2 * math.pi * i / FPS / 2), 0.12)] for i in range(150)]
    assert seen(run(frames), "weave")


def test_lane_change_not_weaving():
    # одно плавное перестроение на соседнюю полосу за 2 секунды
    def x(i):
        k = min(max((i / FPS - 1) / 2, 0), 1)
        return 0.5 + 0.12 * (1 - math.cos(math.pi * k)) / 2
    frames = [[car(x(i), 0.12)] for i in range(150)]
    assert not seen(run(frames), "weave")


def test_pedestrian_in_lane():
    ped = Detection((600, 400, 660, 560), C.PERSON_CLASS, 7)
    assert seen(run([[ped] for _ in range(10)]), "ped")


def test_pedestrian_on_sidewalk_ignored():
    ped = Detection((50, 400, 110, 560), C.PERSON_CLASS, 7)
    assert not seen(run([[ped] for _ in range(10)]), "ped")


def test_car_in_other_lane_ignored():
    # быстро приближающаяся машина в соседней полосе (слева, у края кадра)
    frames = [[car(0.08, 0.10 * 20 / (20 - 10 * i / FPS))] for i in range(36)]
    out = run(frames)
    assert not seen(out, "ttc") and not seen(out, "closing")


def test_parked_car_passing_ignored():
    # проезжаем мимо машины у обочины: её реальное смещение 1.5 ширины, она быстро растёт и уходит вбок
    def det(i):
        w = 0.10 * 20 / (20 - 10 * i / FPS)
        return [car(0.5 + 1.5 * w, w, bottom=0.6 + w)]
    out = run([det(i) for i in range(36)])
    assert not seen(out, "ttc") and not seen(out, "closing")


def test_crash_in_other_lane_detected():
    """Машина в соседнем ряду «взлетает»: рамка уходит вверх и вытягивается, ширина та же."""
    frames = []
    for i in range(90):
        k = max(0, i - 60) / FPS                  # с 2-й секунды — авария
        w, x1, y2 = 0.09 * W, 0.75 * W, 0.62 * H - 1.0 * 0.07 * H * k / 0.3
        h = 0.8 * w * (1 + 2.0 * k)
        frames.append([Detection((x1, y2 - h, x1 + w, y2), 2, 7), car(0.3, 0.05, tid=8, bottom=0.6)])
    out = run(frames)
    assert not any("crash" in kinds for kinds, _ in out[:60]), "до аварии тревоги быть не должно"
    assert seen(out[60:], "crash"), "авария соседней машины должна вызывать тревогу"


def test_car_driving_away_not_crash():
    """Машина уезжает вперёд: низ рамки поднимается, но рамка сужается — это не авария."""
    frames = []
    for i in range(90):
        width = 0.2 * math.exp(-0.8 * i / FPS)
        frames.append([car(0.75, width, tid=3, bottom=0.45 + 1.2 * width)])
    assert not seen(run(frames), "crash")


def test_tracker_keeps_id():
    tr = Tracker()
    ids = set()
    for i in range(30):
        x = 500 + 3 * i
        dets = np.array([[x, 300, x + 150, 420, 0.9 if i % 5 else 0.2, 2]])   # каждый 5-й кадр — слабая детекция
        for _, _, tid in tr.update(dets):
            ids.add(tid)
    assert ids == {1}, f"id должен быть один, получили {ids}"


def test_tracker_car_truck_same_track():
    tr = Tracker()
    ids = set()
    for i in range(10):
        dets = np.array([[500, 300, 650, 420, 0.8, 2 if i % 2 else 7]])     # car / truck вперемешку
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
    print(f"\n{len(tests) - failed}/{len(tests)} тестов пройдено")
    raise SystemExit(1 if failed else 0)
