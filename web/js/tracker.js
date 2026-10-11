// Port of tracker.py: ByteTrack-style two-stage IoU matching, constant-velocity prediction.
// Keep in sync with tracker.py (tools/parity_check.py compares both on the same detections).

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const aa = (a[2] - a[0]) * (a[3] - a[1]), bb = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (aa + bb - inter + 1e-9);
}

function greedyMatch(m, thr) {
  // m: rows = tracks, cols = detections; pairs by descending IoU
  const cells = [];
  m.forEach((row, t) => row.forEach((v, d) => cells.push([v, t, d])));
  cells.sort((p, q) => q[0] - p[0]);
  const ut = new Set(), ud = new Set(), pairs = [];
  for (const [v, t, d] of cells) {
    if (v < thr) break;
    if (!ut.has(t) && !ud.has(d)) { ut.add(t); ud.add(d); pairs.push([t, d]); }
  }
  return pairs;
}

export class Tracker {
  constructor(C) {
    this.C = C;
    this.tracks = [];
    this.nextId = 1;
    this.vehicle = new Set(Object.keys(C.VEHICLE_CLASSES).map(Number));
  }

  group(cls) { return this.vehicle.has(cls) ? 0 : cls === this.C.PERSON_CLASS ? 1 : 2; }

  // dets: [[x1, y1, x2, y2, conf, cls], ...] in pixels -> [{box, cls, tid}]
  update(dets) {
    const C = this.C;
    for (const tr of this.tracks) tr.box = tr.box.map((v, i) => v + tr.vel[i]);
    const high = dets.filter((d) => d[4] >= C.TRACK_HIGH_CONF);
    const low = dets.filter((d) => d[4] < C.TRACK_HIGH_CONF);
    let free = this.tracks.map((_, i) => i);

    const associate = (cand, free) => {
      if (!free.length || !cand.length) return [free, cand.map((_, i) => i)];
      const m = free.map((ti) => cand.map((d) =>
        this.group(this.tracks[ti].cls) === this.group(d[5]) ? iou(this.tracks[ti].box, d) : 0));
      const pairs = greedyMatch(m, C.TRACK_IOU);
      for (const [ti, di] of pairs) {
        const tr = this.tracks[free[ti]], box = cand[di].slice(0, 4);
        tr.vel = tr.vel.map((v, i) => 0.5 * v + 0.5 * (box[i] - tr.obs[i]) / (tr.missed + 1));
        tr.box = box.slice(); tr.obs = box.slice(); tr.cls = cand[di][5];
        tr.hits += 1; tr.missed = 0;
      }
      const mt = new Set(pairs.map((p) => p[0])), md = new Set(pairs.map((p) => p[1]));
      return [free.filter((_, i) => !mt.has(i)), cand.map((_, i) => i).filter((i) => !md.has(i))];
    };

    let newHigh;
    [free, newHigh] = associate(high, free);
    [free] = associate(low, free);
    for (const i of free) this.tracks[i].missed += 1;
    for (const d of newHigh) {
      const box = high[d].slice(0, 4);
      this.tracks.push({ id: this.nextId++, cls: high[d][5], box: box.slice(), obs: box.slice(), vel: [0, 0, 0, 0], hits: 1, missed: 0 });
    }
    this.tracks = this.tracks.filter((t) => t.missed <= C.TRACK_MAX_MISSED);
    return this.tracks.filter((t) => t.missed === 0 && t.hits >= C.TRACK_MIN_HITS)
      .map((t) => ({ box: t.obs.slice(), cls: t.cls, tid: t.id }));
  }
}
