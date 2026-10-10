"""Pedestrian crosswalk (zebra) from the image, no neural network.

Neither COCO nor BDD detection has a "crosswalk" class, but the rule "pedestrian at / near a
crosswalk" needs it. Seen from a dashcam, a zebra is a band of road where bright paint stripes and
dark asphalt alternate along an image row with an almost constant period (the stripes are parallel
to the driving direction, so in the frame they are nearly vertical and converge to the horizon).
Ordinary markings give 2-4 thin lines per row with an irregular period; text and arrows - 1-3 blobs.

Algorithm (~2 ms per frame):
  1. grey frame 640 px wide, road area [ROI_TOP, ROI_BOTTOM] of the height;
  2. "bright" = pixel brighter than the local row mean (window ~1/8 of the frame) by DELTA;
  3. per row - runs of bright pixels; a row is a "zebra row" if >= MIN_STRIPES consecutive stripes
     have a similar period (period spread <= PERIOD_CV) and a paint share of 0.25-0.75;
  4. neighbouring zebra rows are merged into a band; a band thinner than MIN_BAND is noise.
Temporal smoothing (CrosswalkModel): a crosswalk is "seen" if found in >= 2 of the last 5 analyses
and is kept HOLD seconds - under the wheels of the car ahead or pedestrians the zebra is often partial.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

ROI_TOP = 0.48          # above - horizon, a zebra cannot be resolved
ROI_BOTTOM = 0.85       # below - the hood (config.HOOD_Y)
WORK_W = 640
DELTA = 18              # how much brighter a paint stripe is than the local background (0..255)
MIN_STRIPES = 6
MIN_PAINT = 140         # paint is bright in absolute terms too (grey level)
MAX_SAT = 60            # paint and asphalt are unsaturated; cars, signs and facades often are not
CENTER = (0.42, 0.58)   # a crosswalk ahead crosses our path: the band must cover the frame centre
PERIOD_CV = 0.35
MIN_SPAN = 0.30         # a zebra spans at least this share of the frame width
MIN_BAND = 0.025        # minimum thickness of the zebra band (fraction of height)
HOLD = 1.0
MIN_BOTTOM = 0.55        # a band ending right at the horizon is too far to tell from facades and kerbs


def _runs(mask_row):
    """Starts and ends of True runs in a row."""
    d = np.diff(np.concatenate([[0], mask_row.astype(np.int8), [0]]))
    return np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]


def _zebra_row(mask_row, W):
    """(x_left, x_right) of a zebra in the row, or None."""
    s, e = _runs(mask_row)
    if len(s) < MIN_STRIPES:
        return None
    widths = e - s
    keep = (widths >= 2) & (widths <= W * 0.12)
    s, e, widths = s[keep], e[keep], widths[keep]
    if len(s) < MIN_STRIPES:
        return None
    centers = (s + e) / 2
    # find the longest chain of stripes with a similar period
    best = None
    i = 0
    while i < len(s) - 1:
        j = i + 1
        while j < len(s):
            per = np.diff(centers[i:j + 1])
            if per.std() / max(per.mean(), 1e-6) > PERIOD_CV or per[-1] > 3 * np.median(per):
                break
            j += 1
        n = j - i
        if n >= MIN_STRIPES and (best is None or n > best[1] - best[0]):
            best = (i, j)
        i = max(i + 1, j - 1)
    if best is None:
        return None
    i, j = best
    x0, x1 = s[i], e[j - 1]
    if (x1 - x0) < MIN_SPAN * W:
        return None
    fill = widths[i:j].sum() / max(x1 - x0, 1)
    if not 0.25 <= fill <= 0.75:
        return None
    return x0, x1


def detect(frame, boxes=()):
    """Zebra in the frame: (x1, y1, x2, y2) as frame fractions, or None.
    boxes - detections in pixels (x1, y1, x2, y2): their pixels are not paint."""
    H0, W0 = frame.shape[:2]
    k = WORK_W / W0
    small = cv2.resize(frame, (WORK_W, max(1, int(H0 * k))), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1]
    H, W = g.shape
    y0, y1 = int(ROI_TOP * H), int(ROI_BOTTOM * H)
    roi = g[y0:y1].astype(np.int16)
    bg = cv2.blur(roi, (W // 8 | 1, 1)).astype(np.int16)
    mask = ((roi - bg) > DELTA) & (roi >= MIN_PAINT) & (sat[y0:y1] <= MAX_SAT)
    for bx1, by1, bx2, by2 in boxes:
        mask[max(0, int(by1 * k) - y0):max(0, int(by2 * k) - y0 + 1), max(0, int(bx1 * k)):int(bx2 * k) + 1] = False
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 1), np.uint8)).astype(bool)
    rows = []
    for r in range(0, y1 - y0, 2):
        z = _zebra_row(mask[r], W)
        if z is not None:
            rows.append((r + y0, *z))
    if not rows:
        return None
    # merge neighbouring rows (gaps up to 3% of height), take the thickest band
    bands, cur = [], [rows[0]]
    for row in rows[1:]:
        if row[0] - cur[-1][0] <= 0.03 * H:
            cur.append(row)
        else:
            bands.append(cur)
            cur = [row]
    bands.append(cur)
    band = max(bands, key=lambda b: b[-1][0] - b[0][0])
    if band[-1][0] - band[0][0] < MIN_BAND * H or len(band) < 4:
        return None
    a = np.array(band, float)
    if a[-1, 0] < MIN_BOTTOM * H:
        return None
    if np.median(a[:, 1]) > CENTER[0] * W or np.median(a[:, 2]) < CENTER[1] * W:
        return None
    # the gaps between stripes are road: grey and unsaturated
    gap = sat[int(a[0, 0]):int(a[-1, 0]) + 1, int(np.median(a[:, 1])):int(np.median(a[:, 2]))]
    if gap.size == 0 or np.median(gap) > MAX_SAT:
        return None
    return (float(np.median(a[:, 1]) / W), float(a[0, 0] / H), float(np.median(a[:, 2]) / W), float(a[-1, 0] / H))


class CrosswalkModel:
    """Temporal smoothing: where the crosswalk is now (or None)."""

    def __init__(self):
        self.recent = deque(maxlen=5)
        self.box = None
        self.seen = -1e9

    def update(self, t, box):
        """box - result of detect() or None; do not call on steps where the zebra was not analysed."""
        self.recent.append(box is not None)
        if box is not None and sum(self.recent) >= 2:
            if self.box is not None and t - self.seen < HOLD:
                self.box = tuple(0.5 * a + 0.5 * b for a, b in zip(self.box, box))
            else:
                self.box = box
            self.seen = t
        if t - self.seen > HOLD:
            self.box = None
        return self.box

    def near(self, x, y, margin=0.06):
        """A point (bottom of a pedestrian box, frame fractions) on or next to the crosswalk."""
        if self.box is None:
            return False
        x1, y1, x2, y2 = self.box
        dx = (x2 - x1) * 0.25
        return x1 - dx <= x <= x2 + dx and y1 - margin <= y <= y2 + margin


if __name__ == "__main__":
    # quick check on a video: python crosswalk.py video.mp4 [frame_step]
    import sys
    cap = cv2.VideoCapture(sys.argv[1])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    i = hits = n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            b = detect(fr)
            n += 1
            if b:
                hits += 1
                print(f"{i / fps:7.1f} s  zebra x {b[0]:.2f}-{b[2]:.2f}  y {b[1]:.2f}-{b[3]:.2f}")
        i += 1
    print(f"frames with a zebra: {hits}/{n}")
