"""Lanes: Ultra-Fast-Lane-Detection v2 network (CULane, ResNet-18, ONNX) + temporal smoothing.

The model finds up to 4 lane lines: 0 - left line of the left lane, 1 - left line of our lane,
2 - right line of our lane, 3 - right line of the right lane. Each line is stored as x (fraction of
frame width) on fixed rows LANE_ROWS (fraction of height); NaN - no line on that row.

Why: a car's position relative to the markings depends neither on distance nor on our own turning
(the markings turn in the frame together with the cars). "Which lane is the car in" and "is it
crossing the line into our lane" are the main signs of a cut-in.

Model (825 MB, not in git): PINTO model zoo, 324_Ultra-Fast-Lane-Detection-v2:
    curl -L -o models/lanes_src/resources.tar.gz \\
        https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/324_Ultra-Fast-Lane-Detection-v2/resources.tar.gz
    tar -xzf models/lanes_src/resources.tar.gz --include='*ufldv2_culane_res18_320x1600.onnx' -C models/lanes_src
and put the file at models/ufldv2_culane_res18_320x1600.onnx.
"""
import os

import cv2
import numpy as np
import onnxruntime as ort

import config as C

# CULane training parameters of UFLDv2
ROW_ANCHOR = np.linspace(0.42, 1.0, 72)        # rows (fraction of height) for lines 1, 2
COL_ANCHOR = np.linspace(0.0, 1.0, 81)         # columns (fraction of width) for lines 0, 3
CROP_RATIO = 0.6                               # the network sees the bottom 60% of the frame stretched to 320 rows
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _softmax(x, axis=0):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


class LaneDetector:
    def __init__(self, model=C.LANE_MODEL):
        if not os.path.exists(model):
            raise FileNotFoundError(f"No lane model {model} - see the top of lanes.py")
        from detector import providers
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(model, so, providers=providers())
        _, _, self.h, self.w = self.sess.get_inputs()[0].shape
        self.rows = np.array(C.LANE_ROWS, np.float32)

    def _prep(self, frame):
        full_h = int(self.h / CROP_RATIO)
        img = cv2.resize(frame, (self.w, full_h), interpolation=cv2.INTER_AREA)[full_h - self.h:]
        img = (img[..., ::-1].astype(np.float32) / 255 - MEAN) / STD
        return img.transpose(2, 0, 1)[None]

    def detect(self, frame):
        """4 x len(rows): line x (fraction of width) on self.rows, NaN - no line."""
        loc_row, loc_col, ex_row, ex_col = self.sess.run(None, {self.sess.get_inputs()[0].name: self._prep(frame)})
        out = np.full((4, len(self.rows)), np.nan, np.float32)
        n_row, n_col = loc_row.shape[1], loc_col.shape[1]
        valid_row, valid_col = ex_row[0].argmax(0), ex_col[0].argmax(0)       # anchors x 4
        for lane in range(4):
            pts = []
            if lane in (1, 2):                                                # row anchors
                if valid_row[:, lane].sum() < len(ROW_ANCHOR) * C.LANE_MIN_SHARE:
                    continue
                for k in np.nonzero(valid_row[:, lane])[0]:
                    pos = self._subcell(loc_row[0, :, k, lane], n_row)
                    pts.append((ROW_ANCHOR[k], pos))                          # (y, x)
            else:                                                             # column anchors
                if valid_col[:, lane].sum() < len(COL_ANCHOR) * C.LANE_MIN_SHARE / 2:
                    continue
                for k in np.nonzero(valid_col[:, lane])[0]:
                    pos = self._subcell(loc_col[0, :, k, lane], n_col)
                    pts.append((pos, COL_ANCHOR[k]))
            if len(pts) < 4:
                continue
            ys, xs = np.array(pts).T
            order = np.argsort(ys)
            ys, xs = ys[order], xs[order]
            inside = (self.rows >= ys.min()) & (self.rows <= ys.max())
            out[lane, inside] = np.interp(self.rows[inside], ys, xs)
        return out

    @staticmethod
    def _subcell(logits, n):
        """Position inside the grid: soft argmax around the maximum (as in UFLDv2)."""
        i = int(logits.argmax())
        idx = np.arange(max(0, i - 1), min(n - 1, i + 1) + 1)
        return float((_softmax(logits[idx]) * idx).sum() + 0.5) / (n - 1)


class LaneModel:
    """Temporally smoothed markings + geometry: own lane, position of cars in lanes."""

    def __init__(self):
        self.rows = np.array(C.LANE_ROWS, np.float32)
        self.lines = np.full((4, len(self.rows)), np.nan, np.float32)
        self.seen = np.full(4, -1e9)

    def _fit(self, line):
        """A lane line is a smooth curve x(y): a straight line or a parabola (curve). Network points are noisy,
        especially far away; a curve with a large scatter is dropped - no line is better than a wrong one."""
        ok = np.isfinite(line)
        if ok.sum() < 4:
            return None
        ys, xs = self.rows[ok], line[ok]
        deg = 2 if ys.max() - ys.min() > 0.2 else 1
        coef = np.polyfit(ys, xs, deg)
        if np.sqrt(np.mean((np.polyval(coef, ys) - xs) ** 2)) > C.LANE_FIT_RMS:
            return None
        out = np.full_like(line, np.nan)
        span = (self.rows >= ys.min() - C.LANE_EXTEND) & (self.rows <= 1.0)
        out[span] = np.polyval(coef, self.rows[span])
        return out

    def update(self, t, lines):
        if lines is None:
            return
        for k in range(4):
            new = self._fit(lines[k])
            if new is None:
                continue
            old = self.lines[k]
            both = np.isfinite(old) & np.isfinite(new)
            if t - self.seen[k] < C.LANE_HOLD and both.sum() >= 3 and np.abs(old[both] - new[both]).mean() < 0.08:
                a = C.LANE_SMOOTH
                merged = np.where(both, a * old + (1 - a) * new, np.where(np.isfinite(new), new, old))
            else:
                merged = new
            self.lines[k], self.seen[k] = merged, t
        for k in range(4):                                  # line not seen for a while - forget it
            if t - self.seen[k] > C.LANE_HOLD:
                self.lines[k] = np.nan

    def at(self, k, y):
        """x of line k at height y (frame fractions), extended linearly beyond the found part."""
        line = self.lines[k]
        ok = np.isfinite(line)
        if ok.sum() < 3:
            return None
        ys, xs = self.rows[ok], line[ok]
        if ys.min() <= y <= ys.max():
            return float(np.interp(y, ys, xs))
        near = slice(0, 4) if y < ys.min() else slice(-4, None)          # extend the nearest segment
        a, b = np.polyfit(ys[near], xs[near], 1)
        return float(a * y + b)

    def ego(self, y):
        """Own lane at height y: (centre, width) as frame fractions, or None if markings are not visible."""
        left, right = self.at(1, y), self.at(2, y)
        if left is None and right is None:
            return None
        width = None
        if left is not None and right is not None:
            width = right - left
        else:                                   # only one line visible - take the width from the neighbouring lane
            other = (self.at(0, y), left) if left is not None else (right, self.at(3, y))
            if None not in other:
                width = other[1] - other[0]
        if width is None or not (C.LANE_MIN_WIDTH <= width <= C.LANE_MAX_WIDTH * min(1.0, y * 1.6)):
            return None
        center = (left + right) / 2 if left is not None and right is not None else \
            (left + width / 2 if left is not None else right - width / 2)
        return center, width

    def valid(self):
        """Markings are trusted if our lane is visible at the bottom of the frame, narrows towards the horizon
        and the camera is roughly inside it (lane centre at the bottom is not at the frame edge)."""
        near, far = self.ego(0.9), self.ego(0.65)
        if near is None or far is None:
            return False
        return far[1] < near[1] and 0.15 < near[0] < 0.85

    def lane_pos(self, x, y):
        """Lateral position of a point (bottom of a car): 0 - centre of our lane, +-1 - centre of the next one."""
        e = self.ego(y)
        if e is None:
            return None
        return (x - e[0]) / e[1]
