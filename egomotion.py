"""Own motion from a single camera: forward speed and yaw from sparse optical flow on the road.

Why: "the car ahead grows fast" means different things when we drive at 60 km/h and when we
stand in a jam; and without a gyroscope our own turning looks like every object moving sideways.
A car maker would take speed and yaw rate from the CAN bus - `update()` accepts them directly;
for dashcam video we estimate them from the image.

Model: points on the road below the horizon. Driving forward at speed v, a ground point at
distance Z moves away from the focus of expansion with rate v / Z, and for a flat road
Z = f * cam_height / (y - horizon). So every point gives v = rate * f * cam_height / (y - horizon).
Yaw shifts all points sideways by the same amount: the median horizontal flow left after removing
expansion. Moving objects (detection boxes) are masked out; the median makes it robust to the rest.
Accuracy is rough (+-30% on speed), which is enough for "standing / slow / fast" and for the risk model.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

import config as C

WORK_W = 320
HORIZON = 0.47          # horizon row (fraction of height); dashcams are mounted roughly level
CAM_HEIGHT_M = 1.3
ROI = (0.55, 0.84)      # rows used for road points (below the horizon, above the hood)
MAX_POINTS = 120


class EgoMotion:
    def __init__(self):
        self.prev = None
        self.t = None
        self.speed = 0.0      # m/s, smoothed
        self.yaw = 0.0        # frame widths per second (same unit as box shifts in behavior.py)
        self.ok = False

    def update(self, t, frame, boxes=(), can_speed=None, can_yaw=None):
        """frame - BGR; boxes - detections in pixels (masked out). Returns (speed m/s, yaw frac/s, ok)."""
        if can_speed is not None:
            self.speed, self.yaw, self.ok = can_speed, can_yaw or 0.0, True
            return self.speed, self.yaw, self.ok
        H0, W0 = frame.shape[:2]
        k = WORK_W / W0
        g = cv2.cvtColor(cv2.resize(frame, (WORK_W, max(1, int(H0 * k))), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        H, W = g.shape
        prev, prev_t = self.prev, self.t
        self.prev, self.t = g, t
        if prev is None or prev.shape != g.shape or t - prev_t <= 0 or t - prev_t > 0.5:
            self.ok = False
            return self.speed, self.yaw, self.ok
        dt = t - prev_t
        mask = np.zeros_like(prev)
        mask[int(ROI[0] * H):int(ROI[1] * H), int(0.05 * W):int(0.95 * W)] = 255
        for x1, y1, x2, y2 in boxes:
            mask[max(0, int(y1 * k)):int(y2 * k) + 1, max(0, int(x1 * k)):int(x2 * k) + 1] = 0
        p0 = cv2.goodFeaturesToTrack(prev, MAX_POINTS, 0.01, 6, mask=mask)
        if p0 is None or len(p0) < 8:
            self.ok = False
            return self.speed, self.yaw, self.ok
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, winSize=(15, 15), maxLevel=2)
        good = st.reshape(-1) == 1
        p0, p1 = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
        if len(p0) < 8:
            self.ok = False
            return self.speed, self.yaw, self.ok
        cx, hy = W / 2, HORIZON * H
        f = 0.5 * W / math.tan(math.radians(C.CAMERA_HFOV_DEG) / 2)
        r0 = p0 - (cx, hy)
        d = p1 - p0
        dy = np.maximum(r0[:, 1], 2.0)
        # expansion rate from the vertical component (yaw does not move points vertically)
        rate = d[:, 1] / dy / dt
        v = rate * f * CAM_HEIGHT_M / dy
        speed = float(np.clip(np.median(v), 0, 60))
        yaw = float(np.median(d[:, 0] - rate * dt * r0[:, 0]) / W / dt)
        a = 0.6
        self.speed = a * self.speed + (1 - a) * speed
        self.yaw = a * self.yaw + (1 - a) * yaw
        self.ok = True
        return self.speed, self.yaw, self.ok
