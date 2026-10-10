"""Экран регистратора: минимум информации, чтобы не отвлекать водителя."""
import cv2

from behavior import corridor_polygon

COLORS = [(0, 200, 0), (0, 220, 255), (0, 140, 255), (0, 0, 255)]   # BGR: норма → критично
FONT = cv2.FONT_HERSHEY_DUPLEX


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


def draw(frame, res, show_corridor=True, proc_fps=None, recording=False):
    H, W = frame.shape[:2]
    s = W / 1280                       # масштаб под разрешение
    color = COLORS[res.level]

    if show_corridor:
        cv2.polylines(frame, [corridor_polygon(W, H)], True, (255, 255, 255), 1, cv2.LINE_AA)
    if res.lead:
        x1, y1, x2, y2 = map(int, res.lead.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, int(3 * s)))
        if res.distance is not None and not res.oncoming:
            _text(frame, f"{res.distance:.0f} m", (x1, max(y1 - int(10 * s), 20)), 0.8 * s, color)
    for d in res.crashed:                       # авария у любой машины — всегда красным
        x1, y1, x2, y2 = map(int, d.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[3], max(3, int(4 * s)))
    for p in res.peds:
        x1, y1, x2, y2 = map(int, p.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, int(3 * s)))

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
