"""Dashcam screen: as little as possible, so the driver is not distracted.

Colour code (config.LEVEL_NAMES):
  GREEN  - attention: nothing dangerous yet, but worth watching (pedestrian at a crosswalk,
           car drifting towards our lane, too close);
  YELLOW - possible risk: be ready to brake (cut-in, hard braking ahead, pedestrian heading into our lane);
  RED    - dangerous manoeuvre or crash: act now (Brake!, crash ahead, wrong-way car).
What is drawn:
  own lane    - translucent fill between the detected lane lines (no markings - trapezoid outline),
                coloured by the overall level;
  each threat - box in its own colour + short tag (CUT-IN 12m, PED, BRAKE...), an arrow where it
                will be in a second; the crosswalk, when seen, is outlined;
  edge glow   - on the side of the most important hazard (left / right / ahead);
  banner      - the most important alert in large letters; risk bar - the learned risk model.
"""
import cv2
import numpy as np

import config as C
from behavior import corridor_polygon

# BGR: clear (neutral), green, yellow, red
COLORS = [(200, 200, 200), (60, 200, 60), (0, 210, 255), (40, 40, 240)]
FONT = cv2.FONT_HERSHEY_DUPLEX
SHORT = {"danger": "DANGER", "ttc": "BRAKE", "closing": "CLOSING", "cutin": "CUT-IN", "crossing": "CROSSING",
         "ped": "PED", "ped_xwalk": "PED@XWALK", "vru_near": "PED NEAR", "drift": "DRIFT", "hardbrake": "HARD BRAKE",
         "crash": "CRASH", "wrongway": "WRONG WAY", "weave": "ERRATIC", "skid": "SKID", "aggressive": "AGGRESSIVE", "close": "CLOSE", "brakelight": "STOP"}


def _text(img, text, org, scale, color, thick=2):
    cv2.putText(img, text, (org[0] + 2, org[1] + 2), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _label(img, text, org, scale, color, thick):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    pad = int(12 * scale)
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), color, max(2, thick // 2))
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_lanes(frame, lanes, level, k=1.0):
    """Own lane: fill between lines 1 and 2; neighbouring lines thin. Returns False without markings."""
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
    cv2.addWeighted(overlay, 0.25 if level else 0.12, frame, 0.75 if level else 0.88, 0, frame)
    cv2.polylines(frame, [np.array(left, np.int32), np.array(right, np.int32)], False, COLORS[level], 2, cv2.LINE_AA)
    for side in (0, 3):                                    # lines of the neighbouring lanes
        pts = [(int(x * W), int(y * H)) for y, x in zip(lanes.rows, lanes.lines[side]) if np.isfinite(x)]
        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts, np.int32)], False, (220, 220, 220), 1, cv2.LINE_AA)
    return True


def draw_crosswalk(frame, box):
    if box is None:
        return
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    p1, p2 = (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H))
    for x in range(p1[0], p2[0], 18):                     # dashed outline: "zebra here"
        cv2.line(frame, (x, p1[1]), (min(x + 9, p2[0]), p1[1]), (255, 255, 255), 2)
        cv2.line(frame, (x, p2[1]), (min(x + 9, p2[0]), p2[1]), (255, 255, 255), 2)
    _text(frame, "CROSSWALK", (p1[0], max(p1[1] - 6, 14)), 0.5 * W / 1280, (255, 255, 255), 1)


def draw_threats(frame, res, k=1.0, label_scale=0.6):
    """Threat boxes (one per object - the most severe) and motion arrows. k - frame scale."""
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
        if th.dist is not None and th.kind not in ("crash", "danger"):
            tag += f" {th.dist:.0f}m"
        _text(frame, tag, (x1, max(y1 - 6, 14)), label_scale, color, 1)
        if th.path is not None:
            p0 = ((x1 + x2) // 2, y2)
            p1 = (int(th.path[0] * W), int(th.path[1] * H))
            cv2.arrowedLine(frame, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.25)


def draw_glow(frame, side, level, strength):
    """Soft colour glow on the frame edge on the hazard side."""
    if not level or not strength:
        return
    H, W = frame.shape[:2]
    band = max(12, int(W * 0.06))
    ramp = np.linspace(strength * 0.75, 0, band, dtype=np.float32)
    color = np.array(COLORS[level], np.float32)
    regions = {"left": [(slice(None), slice(0, band), ramp[None, :, None])],
               "right": [(slice(None), slice(W - band, W), ramp[::-1][None, :, None])],
               "ahead": [(slice(0, band), slice(None), ramp[:, None, None])]}
    if level >= 3:                                        # red: the whole frame edge
        sides = ["left", "right", "ahead"]
    else:
        sides = [side or "ahead"]
    for s in sides:
        for ys, xs, a in regions[s]:
            roi = frame[ys, xs].astype(np.float32)
            frame[ys, xs] = (roi * (1 - a) + color * a).astype(np.uint8)


def draw(frame, res, show_corridor=True, proc_fps=None, recording=False, advice=None):
    H, W = frame.shape[:2]
    s = W / 1280                       # scale for the resolution
    level = res.level

    if show_corridor and not draw_lanes(frame, res.lanes, level):
        cv2.polylines(frame, [corridor_polygon(W, H, res.view)], True, COLORS[level] if level else (255, 255, 255), 1,
                      cv2.LINE_AA)
    draw_crosswalk(frame, res.crosswalk)
    if res.lead and not any(th.det.tid == res.lead.tid for th in res.threats):
        x1, y1, x2, y2 = map(int, res.lead.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[0], max(1, int(2 * s)))
        if res.distance is not None and not res.oncoming:
            _text(frame, f"{res.distance:.0f} m", (x1, max(y1 - int(10 * s), 20)), 0.7 * s, COLORS[0])
    draw_threats(frame, res, label_scale=0.6 * s)
    if advice is not None:
        draw_glow(frame, advice.side, advice.level, advice.glow)

    if res.top:
        _label(frame, res.top.screen, (int(40 * s), int(80 * s)), 1.5 * s, COLORS[level], max(2, int(3 * s)))
    # status line: level name, TTC, own speed
    status = C.LEVEL_NAMES[level].upper()
    if res.ttc is not None and res.ttc < 10:
        status += f"   TTC {res.ttc:.1f}s"
    if res.ego_speed is not None:
        status += f"   ~{res.ego_speed * 3.6:.0f} km/h"
    _text(frame, status, (int(40 * s), H - int(30 * s)), 0.8 * s, COLORS[level] if level else (255, 255, 255))
    # risk bar (learned model or rule level)
    bx, by, bw, bh = W - int(60 * s), int(120 * s), int(18 * s), int(200 * s)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
    fill = int(bh * min(max(res.risk, 0.0), 1.0))
    cv2.rectangle(frame, (bx, by + bh - fill), (bx + bw, by + bh), COLORS[min(3, max(1, level))], -1)
    _text(frame, "RISK", (bx - int(12 * s), by - int(8 * s)), 0.45 * s, (230, 230, 230), 1)
    if recording:
        cv2.circle(frame, (W - int(110 * s), int(60 * s)), int(12 * s), (0, 0, 255), -1)
        _text(frame, "REC", (W - int(90 * s), int(72 * s)), 0.8 * s, (0, 0, 255))
    if proc_fps:
        _text(frame, f"{proc_fps:.0f} FPS", (W - int(170 * s), H - int(30 * s)), 0.6 * s, (200, 200, 200), 1)
    return frame


__all__ = ["draw", "draw_lanes", "draw_threats", "draw_glow", "draw_crosswalk", "COLORS"]
