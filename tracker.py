"""Light ByteTrack-style tracker: two-stage IoU matching, numpy only.

1. Confident detections are matched to tracks (track positions predicted from their velocity).
2. Weak detections pick up the remaining tracks - so a car is not lost in shadow or rain.
3. Unmatched confident detections start new tracks.
"""
import numpy as np

import config as C


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(iou, thr):
    """(track, detection) pairs in decreasing IoU order."""
    pairs = []
    if iou.size == 0:
        return pairs
    used_t, used_d = set(), set()
    for k in np.argsort(-iou, axis=None):
        t, d = np.unravel_index(k, iou.shape)
        if iou[t, d] < thr:
            break
        if t not in used_t and d not in used_d:
            used_t.add(t)
            used_d.add(d)
            pairs.append((t, d))
    return pairs


def group(cls):
    """A car is often confused with a truck - for tracking they are one class; people and bicycles apart."""
    if cls in C.VEHICLE_CLASSES:
        return 0
    return 1 if cls == C.PERSON_CLASS else 2


class Track:
    def __init__(self, tid, box, cls):
        self.id, self.cls = tid, cls
        self.box = box.copy()     # current (predicted or observed) position
        self.obs = box.copy()     # last observation
        self.vel = np.zeros(4)
        self.hits, self.missed = 1, 0

    def predict(self):
        self.box = self.box + self.vel

    def update(self, box, cls):
        self.vel = 0.5 * self.vel + 0.5 * (box - self.obs) / (self.missed + 1)
        self.box, self.obs, self.cls = box.copy(), box.copy(), cls
        self.hits += 1
        self.missed = 0


class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def update(self, dets):
        """dets: N x 6 array [x1, y1, x2, y2, confidence, class]. Returns [(box, cls, id)]."""
        for tr in self.tracks:
            tr.predict()

        high = dets[dets[:, 4] >= C.TRACK_HIGH_CONF]
        low = dets[dets[:, 4] < C.TRACK_HIGH_CONF]
        free = list(range(len(self.tracks)))

        def associate(cand, free):
            if not free or len(cand) == 0:
                return free, list(range(len(cand)))
            boxes = np.array([self.tracks[i].box for i in free])
            iou = iou_matrix(boxes, cand[:, :4])
            same = np.array([[group(self.tracks[i].cls) == group(int(c)) for c in cand[:, 5]] for i in free])
            pairs = greedy_match(iou * same, C.TRACK_IOU)
            for ti, di in pairs:
                self.tracks[free[ti]].update(cand[di, :4], int(cand[di, 5]))
            matched_t = {ti for ti, _ in pairs}
            matched_d = {di for _, di in pairs}
            return ([free[i] for i in range(len(free)) if i not in matched_t],
                    [d for d in range(len(cand)) if d not in matched_d])

        free, new_high = associate(high, free)
        free, _ = associate(low, free)

        for i in free:
            self.tracks[i].missed += 1
        for d in new_high:
            self.tracks.append(Track(self.next_id, high[d, :4], int(high[d, 5])))
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t.missed <= C.TRACK_MAX_MISSED]

        return [(tuple(t.obs), t.cls, t.id) for t in self.tracks
                if t.missed == 0 and t.hits >= C.TRACK_MIN_HITS]
