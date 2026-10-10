"""Экран регистратора: минимум информации, чтобы не отвлекать водителя.

Как выглядят угрозы:
  своя полоса — полупрозрачная заливка по найденной разметке (без разметки — контур трапеции),
                цвет — общий уровень опасности: зелёный → жёлтый → оранжевый → красный;
  участник    — рамка цвета своего уровня + короткая подпись (CUT-IN, CROSSING, PED, BRAKE…),
                у движущихся вбок — стрелка: где он будет через секунду;
  рамка кадра и крупная надпись — самая опасная угроза сейчас.
"""
import cv2
import numpy as np

import config as C
from behavior import corridor_polygon

COLORS = [(0, 200, 0), (0, 220, 255), (0, 140, 255), (0, 0, 255)]   # BGR: норма → критично
FONT = cv2.FONT_HERSHEY_DUPLEX
SHORT = {"danger": "DANGER", "ttc": "BRAKE", "closing": "CLOSING", "cutin": "CUT-IN", "crossing": "CROSSING", "ped": "PED",
         "crash": "CRASH", "wrongway": "WRONG WAY", "weave": "WEAVING", "close": "CLOSE", "brakelight": "STOP"}


def _text(img, text, org, scale, color, thick=2):
    cv2.putText(img, text, (org[0] + 2, org[1] + 2), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)   # тень: у толстой обводки другой шаг букв
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _label(img, text, org, scale, color, thick):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    pad = int(12 * scale)
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_lanes(frame, lanes, level, k=1.0):
    """Своя полоса — заливка между линиями 1 и 2, соседние линии — тонко. k — масштаб кадра (не нужен)."""
    if lanes is None:
        return False
    H, W = frame.shape[:2]
    left, right = [], []
    for y in lanes.rows:
        e = lanes.ego(float(y))
        if e is None:
            continue
        c, w = e
        left.append((int((c - w / 2) * W), int(y * H)))
        right.append((int((c + w / 2) * W), int(y * H)))
    if len(left) < 3:
        return False
    poly = np.array(left + right[::-1], np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], COLORS[level])
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
    cv2.polylines(frame, [np.array(left, np.int32), np.array(right, np.int32)], False, COLORS[level], 2, cv2.LINE_AA)
    for side in (0, 3):                                    # линии соседних полос
        pts = [(int(x * W), int(y * H)) for y, x in zip(lanes.rows, lanes.lines[side]) if np.isfinite(x)]
        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts, np.int32)], False, (220, 220, 220), 1, cv2.LINE_AA)
    return True


def draw_threats(frame, res, k=1.0, label_scale=0.6):
    """Рамки угроз (по одной на участника — самая серьёзная) и стрелки движения. k — масштаб кадра."""
    H, W = frame.shape[:2]
    worst = {}
    for th in res.threats:
        if th.det.tid not in worst or th.level > worst[th.det.tid].level:
            worst[th.det.tid] = th
    for th in worst.values():
        x1, y1, x2, y2 = (int(v * k) for v in th.det.box)
        color = COLORS[th.level]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if th.level >= 3 else 2)
        tag = SHORT.get(th.kind, th.kind)
        if th.dist is not None and th.kind not in ("crash",):
            tag += f" {th.dist:.0f}m"
        _text(frame, tag, (x1, max(y1 - 6, 14)), label_scale, color, 1)
        if th.path is not None:
            p0 = ((x1 + x2) // 2, y2)
            p1 = (int(th.path[0] * W), int(th.path[1] * H))
            cv2.arrowedLine(frame, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.25)


def draw(frame, res, show_corridor=True, proc_fps=None, recording=False):
    H, W = frame.shape[:2]
    s = W / 1280                       # масштаб под разрешение
    color = COLORS[res.level]

    if show_corridor and not draw_lanes(frame, res.lanes, res.level):
        cv2.polylines(frame, [corridor_polygon(W, H)], True, (255, 255, 255), 1, cv2.LINE_AA)
    if res.lead and not any(th.det.tid == res.lead.tid for th in res.threats):
        x1, y1, x2, y2 = map(int, res.lead.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[0], max(2, int(2 * s)))
        if res.distance is not None and not res.oncoming:
            _text(frame, f"{res.distance:.0f} m", (x1, max(y1 - int(10 * s), 20)), 0.8 * s, COLORS[0])
    draw_threats(frame, res, label_scale=0.7 * s)

    cv2.rectangle(frame, (0, 0), (W - 1, H - 1), color, max(8, int(18 * s)))

    if res.top:
        _label(frame, res.top.screen, (int(40 * s), int(90 * s)), 1.8 * s, color, max(2, int(4 * s)))
    if res.ttc is not None and res.ttc < 10:
        _text(frame, f"TTC {res.ttc:.1f} s", (int(40 * s), H - int(40 * s)), 1.1 * s, (255, 255, 255))
    if recording:
        cv2.circle(frame, (W - int(110 * s), int(60 * s)), int(12 * s), (0, 0, 255), -1)
        _text(frame, "REC", (W - int(90 * s), int(72 * s)), 0.9 * s, (0, 0, 255))
    if proc_fps:
        _text(frame, f"{proc_fps:.0f} FPS", (W - int(170 * s), H - int(40 * s)), 0.8 * s, (200, 200, 200), 1)
    return frame


__all__ = ["draw", "draw_lanes", "draw_threats", "COLORS", "C"]
