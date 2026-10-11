// Port of behavior.py (BehaviorEngine.update) for the browser. The lane network (825 MB) does not run
// in a browser, so the own lane comes from geometry (the same path behavior.py takes without markings);
// own turning comes from the shift of the car boxes (no optical flow here). Keep in sync with behavior.py.

function polyfit1(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
  const slope = sxx > 0 ? sxy / sxx : 0;
  return [slope, my - slope * mx];
}
const median = (a) => { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const std = (a) => { const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length); };
function percentile(xs, q) {
  const a = [...xs].sort((x, y) => x - y), pos = q / 100 * (a.length - 1), i = Math.floor(pos);
  return i + 1 >= a.length ? a[i] : a[i] + (a[i + 1] - a[i]) * (pos - i);
}
const EPS = 1e-6;

export class BehaviorEngine {
  constructor(C) {
    this.C = C;
    this.vehicle = new Set(Object.keys(C.VEHICLE_CLASSES).map(Number));
    this.vru = new Set([C.PERSON_CLASS, C.BICYCLE_CLASS, 3]);
    this.hist = new Map();      // tid -> [[t, x1, y1, x2, y2, u]] (fractions)
    this.cls = new Map(); this.lastSeen = new Map(); this.firstSeen = new Map(); this.rates = new Map(); this.critHist = new Map();
    this.since = new Map(); this.lastRaw = new Map(); this.crashAt = new Map();
    this.yawQ = []; this.bottoms = [];
    this.top = C.CORRIDOR_TOP_Y; this.hood = C.HOOD_Y;
    this.xwalk = { box: null, seen: -1e9, recent: [] };
    this.marks = new Map(); this.laneIdx = new Map();   // reputation: risky manoeuvres per car, lane index per car
    this.prevActive = new Set();                         // alert kinds confirmed on the previous step
  }

  focal() { return 0.5 / Math.tan((this.C.CAMERA_HFOV_DEG * Math.PI / 180) / 2); }

  // own lane trapezoid in pixels (corridor_polygon in behavior.py)
  corridor(W, H) {
    const C = this.C, cx = C.CORRIDOR_CENTER_X;
    return [[cx - C.CORRIDOR_TOP_HALF, this.top], [cx + C.CORRIDOR_TOP_HALF, this.top],
            [cx + C.CORRIDOR_BOTTOM_HALF, this.hood], [cx - C.CORRIDOR_BOTTOM_HALF, this.hood]].map(([x, y]) => [x * W, y * H]);
  }

  half(y) {
    const C = this.C, k = Math.max(0, Math.min(1, (y - this.top) / Math.max(this.hood - this.top, 1e-6)));
    return C.CORRIDOR_TOP_HALF + k * (C.CORRIDOR_BOTTOM_HALF - C.CORRIDOR_TOP_HALF);
  }

  // where the road is in this camera: horizon from far cars' bottoms, hood below the lowest car (behavior._calibrate)
  calibrate(t, W, H, dets) {
    const C = this.C;
    for (const d of dets) {
      const [x1, , x2, y2] = d.box;
      if (this.vehicle.has(d.cls) && (x2 - x1) / W < 0.5 && y2 / H < 0.985) this.bottoms.push([t, y2 / H, (x2 - x1) / W]);
    }
    while (this.bottoms.length && t - this.bottoms[0][0] > C.CALIB_WINDOW) this.bottoms.shift();
    if (this.bottoms.length < C.CALIB_MIN) return;
    const far = this.bottoms.filter((b) => b[2] < C.CALIB_FAR_WIDTH).map((b) => b[1]);
    if (far.length >= 8) {
      const top = Math.min(Math.max(percentile(far, 15) - 0.015, C.CALIB_TOP_RANGE[0]), C.CALIB_TOP_RANGE[1]);
      this.top = 0.9 * this.top + 0.1 * top;
    }
    if (this.bottoms.at(-1)[0] - this.bottoms[0][0] >= C.CALIB_MIN_SPAN) {
      const low = percentile(this.bottoms.map((b) => b[1]), 98) + 0.05;
      const hood = low >= C.HOOD_Y ? C.HOOD_Y : Math.max(low, this.top + 0.3);
      this.hood = 0.9 * this.hood + 0.1 * hood;
    }
  }

  updateCrosswalk(t, box) {
    const X = this.xwalk, C = this.C;
    X.recent.push(box !== null); if (X.recent.length > 5) X.recent.shift();
    if (box && X.recent.filter(Boolean).length >= 2) {
      X.box = X.box && t - X.seen < C.XWALK_HOLD ? X.box.map((v, i) => 0.5 * v + 0.5 * box[i]) : box;
      X.seen = t;
    }
    if (t - X.seen > C.XWALK_HOLD) X.box = null;
  }

  // dets: [{box:[x1,y1,x2,y2] px, cls, tid}]; crosswalk: undefined (not analysed) | null | [x1,y1,x2,y2] fractions
  update(t, W, H, dets, crosswalk) {
    const C = this.C, aspect = W / H;
    const res = { alerts: [], lead: null, ttc: null, distance: null, oncoming: false, peds: [], crashed: [], threats: [],
                  tracks: [], crosswalk: null, risk: 0, level: 0, top: null };
    if (crosswalk !== undefined) this.updateCrosswalk(t, crosswalk);
    res.crosswalk = this.xwalk.box;
    this.calibrate(t, W, H, dets);
    const live = [], bikes = dets.filter((d) => d.cls === C.BICYCLE_CLASS || d.cls === 3).map((d) => d.box);
    const rider = (p, b) => Math.max(0, Math.min(p[2], b[2]) - Math.max(p[0], b[0])) * Math.max(0, Math.min(p[3], b[3]) - Math.max((p[1] + p[3]) / 2, b[1]))
      >= 0.4 * (p[2] - p[0]) * (p[3] - p[1]) / 2;
    for (const d of dets) {
      const [x1, y1, x2, y2] = d.box;
      if (d.cls === C.PERSON_CLASS && bikes.some((b) => rider(d.box, b))) continue;   // a rider
      if ((y1 + y2) / 2 / H > this.hood) { res.tracks.push([d, "hood"]); continue; }
      if (!this.vehicle.has(d.cls) && !this.vru.has(d.cls)) continue;
      const box = [x1 / W, y1 / H, x2 / W, y2 / H];
      const u = this.lateral(d.cls, box, aspect);
      const h = this.hist.get(d.tid) || [];
      h.push([t, ...box, u]);
      while (t - h[0][0] > C.HISTORY_SEC) h.shift();
      this.hist.set(d.tid, h);
      this.cls.set(d.tid, d.cls); this.lastSeen.set(d.tid, t);
      if (!this.firstSeen.has(d.tid)) this.firstSeen.set(d.tid, t);
      live.push(d);
    }
    for (const [tid, seen] of [...this.lastSeen]) {
      if (t - seen > C.TRACK_TTL) for (const m of [this.lastSeen, this.hist, this.cls, this.rates, this.firstSeen, this.critHist]) m.delete(tid);
    }
    for (const q of this.critHist.values()) while (q.length && t - q[0] > 1.0) q.shift();
    const yaw = this.yawRate(t, live), turning = Math.abs(yaw) > C.YAW_TURN, slack = C.YAW_SLACK * Math.abs(yaw);

    let closeCars = 0;
    for (const d of live) {
      if (!this.vehicle.has(d.cls)) continue;
      const b = this.hist.get(d.tid).at(-1), sv = (b[3] - b[1]) * aspect > C.CROSS_MIN_ASPECT * (b[4] - b[2]);
      const dd = this.distance(d.cls, b[3] - b[1], b[4] - b[2], aspect, sv);
      if (dd !== null && dd < C.DENSE_DIST) closeCars++;
    }
    this.dense = closeCars >= C.DENSE_CARS;
    const raw = new Set(), threats = [];
    let lead = null;
    for (const d of live) {
      const h = this.hist.get(d.tid), [, x1, y1, x2, y2, u] = h.at(-1);
      const w = x2 - x1, by = y2, vehicle = this.vehicle.has(d.cls);
      const sideView = (x2 - x1) * aspect > C.CROSS_MIN_ASPECT * (y2 - y1);
      const dist = this.distance(d.cls, x2 - x1, y2 - y1, aspect, sideView);
      const ttc = this.ttc(h, t, sideView || !vehicle);
      const du = this.latSpeed(h, t, yaw, vehicle, aspect);
      const inside = x1 > C.EDGE_MARGIN && x2 < 1 - C.EDGE_MARGIN, ahead = by >= this.top;
      const whole = x1 > C.SIDE_MARGIN && x2 < 1 - C.SIDE_MARGIN;
      const inCorr = Math.abs((x1 + x2) / 2 - C.CORRIDOR_CENTER_X) <= this.half(by) * C.PATH_CORRIDOR_FRAC;
      const inLane = u !== null && Math.abs(u) <= C.IN_LANE && ahead && inCorr;
      const inPath = u !== null && Math.abs(u) <= (vehicle ? C.IMPACT_U_CAR : C.IMPACT_U) && ahead && inCorr;
      const toward = u !== null && du !== null && u !== 0 ? -Math.sign(u) * du : 0;
      const age = t - this.firstSeen.get(d.tid);
      const closingSpeed = dist !== null && ttc ? dist / ttc : 0;   // fast approach, not just a small TTC
      const found = [];
      if (vehicle && inLane && inside && (lead === null || y2 > this.hist.get(lead.tid).at(-1)[4])) lead = d;

      // cut-in / crossing / drift
      if (u !== null && du !== null && ahead && whole && !inLane && Math.abs(u) <= C.CUTIN_MAX_U && age >= C.CUTIN_MIN_AGE) {
        const tIn = toward > 0 ? (Math.abs(u) - C.IN_LANE) / toward : null;
        const closingIn = (ttc !== null && ttc <= C.CUTIN_TTC && closingSpeed >= C.MIN_CLOSING) || (dist !== null && dist <= C.CRIT_DIST);
        if (vehicle && closingIn && dist <= C.CUTIN_MAX_DIST && toward >= C.CUTIN_MIN_SPEED + slack && tIn !== null && tIn <= C.CUTIN_HORIZON
            && (!turning || (ttc !== null && ttc < C.TTC_CRITICAL))) {
          const crit = ttc !== null && ttc < C.CUTIN_CRIT_TTC && dist <= C.CUTIN_CRIT_DIST;
          found.push([sideView ? "crossing" : "cutin", crit ? 3 : 2]);
        } else if (vehicle && !sideView && dist <= C.DRIFT_MAX_DIST && toward >= C.DRIFT_MIN_SPEED + slack && tIn !== null && tIn <= C.DRIFT_HORIZON && !turning) {
          found.push(["drift", 1]);
        }
      }
      // pedestrians and two-wheelers: only if we are driving towards them
      if (!vehicle && dist !== null && ahead) {
        let lvl = 0;
        const approaching = ttc !== null && ttc <= C.PED_TTC && closingSpeed >= C.PED_MIN_CLOSING;
        if (inPath && dist <= C.VRU_NEAR && approaching) lvl = ttc < C.TTC_CRITICAL && dist <= C.PED_RED_DIST ? 3 : 2;
        else if (u !== null && du !== null && !inLane && toward >= C.VRU_MIN_SPEED && Math.abs(u) <= C.VRU_STEP_U && age >= C.CUTIN_MIN_AGE) {
          const tIn = (Math.abs(u) - C.IN_LANE) / toward;
          if (ttc !== null && ttc <= C.CUTIN_TTC && dist <= C.VRU_DIST && tIn <= C.VRU_HORIZON && this.steadyU(h, t, 0.5, 9))
            lvl = ttc < C.CUTIN_CRIT_TTC ? 3 : 2;
        }
        if (!lvl && age <= C.VRU_SUDDEN_AGE && dist <= C.VRU_SUDDEN_DIST && inside && approaching && u !== null && Math.abs(u) <= C.VRU_SUDDEN_U) lvl = 2;
        if (lvl) { found.push(["ped", lvl]); res.peds.push(d); }
      }
      // collision course: u + du * TTC inside our track
      if (ttc !== null && u !== null && ahead && dist !== null && dist <= C.COURSE_MAX_DIST && ttc <= C.COURSE_TTC
          && closingSpeed >= (vehicle ? C.MIN_CLOSING : C.PED_MIN_CLOSING)
          && !(vehicle && inPath && !sideView) && !found.length && !(vehicle && turning && ttc >= C.TTC_CRITICAL) && (whole || !vehicle)) {
        const uImp = u + (du || 0) * ttc;
        const movingIn = toward >= C.COURSE_MIN_TOWARD + slack || (sideView && inPath);
        if (Math.abs(uImp) <= C.IMPACT_U && (du !== null || Math.abs(u) <= C.IMPACT_U) && (movingIn || (!vehicle && (toward >= C.COURSE_MIN_TOWARD || inPath)))) {
          const kind = !vehicle ? "ped" : sideView ? "crossing" : "cutin";
          const crit = ttc < C.CUTIN_CRIT_TTC && dist <= C.CUTIN_CRIT_DIST;
          if (vehicle || dist <= C.PED_COURSE_DIST) {
            found.push([kind, crit ? 3 : 2]);
            if (kind === "ped") res.peds.push(d);
          }
        }
      }
      // car in our path closing in fast
      if (vehicle && inPath && inside && !sideView && ttc !== null && w >= C.TTC_MIN_WIDTH) {
        const closing = dist ? dist / ttc : 0;
        if (closing > C.LEAD_MAX_CLOSING) { if (d === lead) res.oncoming = true; }
        else if (ttc < C.TTC_WARNING && closing >= C.MIN_CLOSING) {
          const crit = ttc < C.TTC_CRITICAL && w >= C.CRITICAL_MIN_WIDTH && closing >= C.BRAKE_MIN_CLOSING;
          if (crit) { const q = this.critHist.get(d.tid) || []; q.push(t); this.critHist.set(d.tid, q); }
          found.push(crit ? ["ttc", 3] : ["closing", 2]);
        }
      }
      if (this.dense) for (const f of found) if (["cutin", "crossing", "drift"].includes(f[0])) f[1] = Math.max(1, f[1] - 1);
      for (const [kind, level] of found) {
        threats.push({ det: d, kind, level, u, du, ttc, dist, path: this.predict(h, u, du) });
        raw.add(kind);
      }
      res.tracks.push([d, (u !== null ? `u${u >= 0 ? "+" : ""}${u.toFixed(1)}` : "-") + (du !== null && Math.abs(du) >= 0.2 ? ` ${du.toFixed(1)}/s` : "")
        + found.map(([k]) => " " + k.toUpperCase()).join("")]);
    }

    for (const tid of this.crashes(t)) this.crashAt.set(tid, t);
    for (const [k, v] of [...this.crashAt]) if (t - v > C.CRASH_HOLD) this.crashAt.delete(k);
    if (this.crashAt.size) {
      raw.add("crash");
      res.crashed = live.filter((d) => this.crashAt.has(d.tid));
      for (const d of res.crashed) threats.push({ det: d, kind: "crash", level: d.box[2] - d.box[0] >= C.CRASH_CRIT_WIDTH * W ? 3 : 2 });
    }

    if (lead) {
      res.lead = lead;
      const h = this.hist.get(lead.tid), last = h.at(-1), w = last[3] - last[1];
      res.distance = this.distance(lead.cls, w, last[4] - last[2], aspect, false);
      const side = w * aspect > C.CROSS_MIN_ASPECT * (last[4] - last[2]);
      res.ttc = res.oncoming || side ? null : this.ttc(h, t, false);
      const [rate, jerk] = side ? [0, 0] : this.brake(lead.tid, h, t);
      if (!res.oncoming && res.distance !== null && res.distance <= C.BRAKE_MAX_DIST && rate >= C.BRAKE_RATE && jerk >= C.BRAKE_JERK && res.ttc !== null && res.ttc < C.BRAKE_MAX_TTC) {
        raw.add("hardbrake");
        threats.push({ det: lead, kind: "hardbrake", level: res.ttc !== null && res.ttc < C.BRAKE_CRIT_TTC ? 3 : 2, ttc: res.ttc, dist: res.distance });
      }
      if (w >= C.CLOSE_WIDTH && res.ttc !== null && res.ttc < C.CLOSE_MAX_TTC) {
        raw.add("close"); threats.push({ det: lead, kind: "close", level: 1, ttc: res.ttc, dist: res.distance });
      }
      if (C.WEAVE && this.weaving(h, t)) { raw.add("weave"); threats.push({ det: lead, kind: "weave", level: 2, ttc: res.ttc, dist: res.distance }); }
    }

    this.reputation(t, live, threats, raw, aspect);
    res.raw = raw; res.threats = threats;
    res.alerts = this.confirm(t, raw, threats);
    res.top = res.alerts.reduce((a, b) => (a === null || b.level > a.level ? b : a), null);
    res.level = res.top ? res.top.level : 0;
    res.risk = Math.max(0, ...threats.map((th) => th.level)) / 3;
    return res;
  }

  // remember cars that made risky manoeuvres / keep changing lanes; warn while such a car is near (behavior._reputation)
  reputation(t, live, threats, raw, aspect) {
    const C = this.C, MARK = new Set(["hardbrake", "cutin", "crossing", "drift", "weave", "skid"]);
    for (const th of threats) {
      if (!MARK.has(th.kind) || !this.prevActive.has(th.kind) || th.level < 2) continue;
      const m = this.marks.get(th.det.tid) || [];
      if (!m.length || t - m.at(-1)[0] > C.MARK_GAP || m.at(-1)[1] !== th.kind) m.push([t, th.kind]);
      this.marks.set(th.det.tid, m);
    }
    const threatened = new Set(threats.map((th) => th.det.tid));
    if (this.dense) return;
    for (const d of live) {
      if (!this.vehicle.has(d.cls)) continue;
      const h = this.hist.get(d.tid), u = h.at(-1)[5];
      if (u !== null && Math.abs(u - Math.round(u)) <= C.LANE_DEEP) {
        const lane = Math.round(u), prev = this.laneIdx.get(d.tid);
        if (!prev) this.laneIdx.set(d.tid, [lane, t]);
        else if (lane !== prev[0]) {
          if (t - prev[1] >= C.LANE_STAY) { const m = this.marks.get(d.tid) || []; m.push([t, "lanechange"]); this.marks.set(d.tid, m); }
          this.laneIdx.set(d.tid, [lane, t]);
        }
      }
      const m = (this.marks.get(d.tid) || []).filter(([tt, k]) => t - tt <= C.MARK_WINDOW && C.MARK_COUNTS.includes(k));
      const score = m.reduce((a, [, k]) => a + (k === "lanechange" ? C.LANECHANGE_WEIGHT : 1), 0);
      if (score < 1 || threatened.has(d.tid)) continue;
      const [, x1, y1, x2, y2] = h.at(-1);
      const sv = (x2 - x1) * aspect > C.CROSS_MIN_ASPECT * (y2 - y1);
      const dist = this.distance(d.cls, x2 - x1, y2 - y1, aspect, sv);
      if (dist === null || dist > C.MARK_DIST || y2 < this.top) continue;
      threats.push({ det: d, kind: "aggressive", level: score >= C.MARK_YELLOW ? 2 : 1, u, dist, count: m.length });
      raw.add("aggressive");
    }
    for (const k of [...this.marks.keys()]) if (!this.hist.has(k)) { this.marks.delete(k); this.laneIdx.delete(k); }
  }

  confirm(t, raw, threats) {
    const C = this.C;
    for (const k of raw) { if (!this.since.has(k)) this.since.set(k, t); this.lastRaw.set(k, t); }
    for (const k of [...this.since.keys()]) if (t - this.lastRaw.get(k) > C.RAW_GAP) this.since.delete(k);
    const active = new Set([...this.since.keys()].filter((k) => t - this.lastRaw.get(k) <= C.ALERT_HOLD + EPS
      && this.lastRaw.get(k) - this.since.get(k) >= C.CONFIRM_SEC[k] - EPS));
    if (active.has("ttc")) active.delete("closing");
    if (active.has("cutin")) active.delete("drift");
    this.prevActive = new Set(active);
    this.lastLevel = this.lastLevel || new Map();
    return [...active].map((k) => {
      const lv = threats.filter((th) => th.kind === k).map((th) => th.level);
      if (lv.length) this.lastLevel.set(k, Math.max(...lv));       // held alerts keep their last real level
      return { kind: k, level: this.lastLevel.get(k) ?? C.ALERTS[k][0], screen: C.ALERTS[k][1], voice: C.ALERTS[k][2] };
    });
  }

  lateral(cls, box, aspect) {
    const C = this.C, [x1, y1, x2, y2] = box, cx = (x1 + x2) / 2;
    if (y2 < this.top) return null;
    const meters = this.vehicle.has(cls) ? (cx - C.CORRIDOR_CENTER_X) / Math.max(x2 - x1, 1e-6) * C.VEHICLE_WIDTH_M
      : (cx - C.CORRIDOR_CENTER_X) * aspect / Math.max(y2 - y1, 1e-6) * C.PERSON_HEIGHT_M;
    return meters / C.LANE_WIDTH_M;
  }

  distance(cls, w, h, aspect, sideView) {
    const C = this.C;
    if (this.vehicle.has(cls) && cls !== 3) {
      if (!sideView) return C.VEHICLE_WIDTH_M * this.focal() / Math.max(w, 1e-6);
      return ([5, 7].includes(cls) ? C.TRUCK_HEIGHT_M : C.CAR_HEIGHT_M) * this.focal() * aspect / Math.max(h, 1e-6);
    }
    return C.PERSON_HEIGHT_M * this.focal() * aspect / Math.max(h, 1e-6);
  }

  yawRate(t, live) {
    const C = this.C, speeds = [];
    for (const d of live) {
      const h = this.hist.get(d.tid);
      if (h.length >= 2 && h.at(-1)[0] - h.at(-2)[0] <= 0.3) {
        const dt = Math.max(h.at(-1)[0] - h.at(-2)[0], 1e-3);
        speeds.push(((h.at(-1)[1] + h.at(-1)[3]) - (h.at(-2)[1] + h.at(-2)[3])) / 2 / dt);
      }
    }
    this.yawQ.push([t, speeds.length >= C.YAW_MIN_TRACKS ? median(speeds) : 0]);
    while (t - this.yawQ[0][0] > C.KIN_WINDOW) this.yawQ.shift();
    return mean(this.yawQ.map((p) => p[1]));
  }

  latSpeed(h, t, yaw, vehicle, aspect) {
    const C = this.C, pts = h.filter((p) => t - p[0] <= C.KIN_WINDOW && p[5] !== null);
    if (pts.length < 4 || pts.at(-1)[0] - pts[0][0] < C.KIN_MIN_SPAN) return null;
    const ts = pts.map((p) => p[0]);
    if (!vehicle) {
      const hh = pts.at(-1)[4] - pts.at(-1)[2];
      return polyfit1(ts, pts.map((p) => p[5]))[0] - yaw * aspect / Math.max(hh, 1e-6) * C.PERSON_HEIGHT_M / C.LANE_WIDTH_M;
    }
    const cx = pts.map((p) => (p[1] + p[3]) / 2), w = pts.map((p) => p[3] - p[1]);
    const dcx = polyfit1(ts, cx)[0], g = polyfit1(ts, w.map(Math.log))[0];
    const X = (cx.at(-1) - C.CORRIDOR_CENTER_X) / w.at(-1);
    return ((dcx - yaw) / w.at(-1) - X * g) * C.VEHICLE_WIDTH_M / C.LANE_WIDTH_M;
  }

  steadyU(h, t, span, limit) {
    const pts = h.filter((p) => t - p[0] <= span && p[5] !== null).map((p) => p[5]);
    return pts.length >= 4 && Math.max(...pts.map(Math.abs)) <= limit && h.at(-1)[0] - h[0][0] >= span;
  }

  predict(h, u, du) {
    if (du === null || u === null) return null;
    const C = this.C, [, x1, , x2, y2] = h.at(-1);
    return [(x1 + x2) / 2 + du * C.PREDICT_SEC * (x2 - x1) * C.LANE_WIDTH_M / C.VEHICLE_WIDTH_M, y2];
  }

  rate(h, t, byHeight) {
    const C = this.C, pts = h.filter((p) => t - p[0] <= C.TTC_WINDOW);
    if (pts.length < 4 || pts.at(-1)[0] - pts[0][0] < C.TTC_MIN_SPAN) return null;
    const ts = pts.map((p) => p[0]), ws = pts.map((p) => Math.max(byHeight ? p[4] - p[2] : p[3] - p[1], 1e-6));
    return { rate: polyfit1(ts, ws.map(Math.log))[0], ts, ws };
  }

  ttc(h, t, byHeight) {
    const C = this.C, r = this.rate(h, t, byHeight);
    if (!r || (!byHeight && r.ws.at(-1) < C.TTC_MIN_WIDTH) || r.rate <= C.TTC_MIN_RATE) return null;
    return Math.max(1 / r.rate - (r.ts.at(-1) - mean(r.ts)), 0.1);
  }

  brake(tid, h, t) {
    const C = this.C, r = this.rate(h, t, false);
    if (!r) return [0, 0];
    const q = this.rates.get(tid) || [];
    q.push([t, r.rate]);
    while (t - q[0][0] > C.BRAKE_JERK_SEC) q.shift();
    this.rates.set(tid, q);
    return [r.rate, r.rate - Math.min(...q.map((p) => p[1]))];
  }

  weaving(h, t) {
    const C = this.C, pts = h.filter((p) => t - p[0] <= C.WEAVE_WINDOW);
    if (pts.length < 10 || pts.at(-1)[0] - pts[0][0] < C.WEAVE_MIN_SPAN) return false;
    const ts = pts.map((p) => p[0]), ws = pts.map((p) => p[3] - p[1]), xs = pts.map((p) => (p[1] + p[3]) / 2);
    const width = mean(ws);
    if (width < C.WEAVE_MIN_WIDTH) return false;
    const [a, b] = polyfit1(ts, xs);
    const amp = std(xs.map((x, i) => x - (a * ts[i] + b))) / width;
    const fps = (ts.length - 1) / (ts.at(-1) - ts[0]), k = Math.max(3, Math.floor(0.3 * fps));
    const smooth = [];
    for (let i = 0; i + k <= xs.length; i++) smooth.push(mean(xs.slice(i, i + k)));
    const moving = smooth.slice(1).map((v, i) => (v - smooth[i]) * fps / width).filter((v) => Math.abs(v) > C.WEAVE_MIN_SPEED).map(Math.sign);
    let rev = 0;
    for (let i = 1; i < moving.length; i++) if (moving[i] !== moving[i - 1]) rev++;
    return amp >= C.WEAVE_REL_STD && rev >= C.WEAVE_MIN_REVERSALS;
  }

  crashes(t) {
    const C = this.C, pairs = new Map();
    for (const [tid, h] of this.hist) {
      if (this.lastSeen.get(tid) !== t || !this.vehicle.has(this.cls.get(tid))) continue;
      const old = h.find((p) => t - p[0] <= C.CRASH_WINDOW);
      if (!old || old === h.at(-1) || t - old[0] < C.CRASH_WINDOW / 2) continue;
      pairs.set(tid, [old, h.at(-1)]);
    }
    if (!pairs.size) return [];
    const shifts = [...pairs.values()].map(([o, n]) => ((n[2] + n[4]) - (o[2] + o[4])) / 2);
    const ego = shifts.length >= 3 ? median(shifts) : 0;
    const out = [];
    for (const [tid, [o, n]] of pairs) {
      const dt = n[0] - o[0], w = n[3] - n[1], h = n[4] - n[2], w0 = o[3] - o[1], h0 = o[4] - o[2];
      const cut = Math.min(o[1], n[1]) < C.EDGE_MARGIN || Math.max(o[3], n[3]) > 1 - C.EDGE_MARGIN || (n[2] + n[4]) / 2 > this.hood;
      const young = t - (this.firstSeen.get(tid) ?? t) < C.CRASH_MIN_AGE;
      if (cut || young || w < C.CRASH_MIN_WIDTH || h <= 0 || h0 <= 0) continue;
      const rise = -(((n[2] + n[4]) - (o[2] + o[4])) / 2 - ego) / h / dt;
      const taller = Math.log((h / w) / (h0 / w0)) / dt, scale = Math.log(w / w0) / dt;
      if (rise >= C.CRASH_RISE && taller >= C.CRASH_ASPECT && Math.abs(scale) < C.CRASH_MAX_SCALE) out.push(tid);
    }
    return out;
  }
}
