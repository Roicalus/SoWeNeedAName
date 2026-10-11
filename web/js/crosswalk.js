// Port of crosswalk.py: zebra crossing from the image (bright unsaturated stripes alternating with grey
// asphalt at a regular period across the road ahead). Keep the constants in sync with crosswalk.py.
const K = { ROI_TOP: 0.48, ROI_BOTTOM: 0.85, WORK_W: 640, DELTA: 18, MIN_STRIPES: 6, MIN_PAINT: 140, MAX_SAT: 60,
            CENTER: [0.42, 0.58], PERIOD_CV: 0.35, MIN_SPAN: 0.30, MIN_BAND: 0.025, MIN_BOTTOM: 0.55 };

export class Crosswalk {
  constructor() {
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
  }

  // source: video/canvas, boxes: detections in source pixels -> [x1, y1, x2, y2] fractions or null
  detect(source, w, h, boxes = []) {
    const W = K.WORK_W, H = Math.max(1, Math.round(h * W / w)), k = W / w;
    this.canvas.width = W; this.canvas.height = H;
    this.ctx.drawImage(source, 0, 0, W, H);
    const px = this.ctx.getImageData(0, 0, W, H).data;
    const y0 = Math.floor(K.ROI_TOP * H), y1 = Math.floor(K.ROI_BOTTOM * H), win = (W >> 3) | 1, half = win >> 1;
    const gray = new Float32Array(W), sat = new Uint8Array(W * H), rows = [];
    for (let i = 0, j = 0; i < W * H; i++, j += 4) {
      const r = px[j], g = px[j + 1], b = px[j + 2], mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      sat[i] = mx ? Math.round(255 * (mx - mn) / mx) : 0;
    }
    const masked = (x, y) => boxes.some(([bx1, by1, bx2, by2]) => x >= bx1 * k && x <= bx2 * k && y >= by1 * k && y <= by2 * k);
    for (let y = y0; y < y1; y += 2) {
      const pre = new Float32Array(W + 1);
      for (let x = 0; x < W; x++) {
        const j = 4 * (y * W + x);
        gray[x] = 0.299 * px[j] + 0.587 * px[j + 1] + 0.114 * px[j + 2];
        pre[x + 1] = pre[x] + gray[x];
      }
      const mask = new Uint8Array(W);
      for (let x = 0; x < W; x++) {
        const a = Math.max(0, x - half), b = Math.min(W, x + half + 1);
        const bg = (pre[b] - pre[a]) / (b - a);
        mask[x] = gray[x] - bg > K.DELTA && gray[x] >= K.MIN_PAINT && sat[y * W + x] <= K.MAX_SAT && !masked(x, y) ? 1 : 0;
      }
      const z = zebraRow(mask, W);
      if (z) rows.push([y, z[0], z[1]]);
    }
    if (!rows.length) return null;
    const bands = [];
    let cur = [rows[0]];
    for (const r of rows.slice(1)) { if (r[0] - cur.at(-1)[0] <= 0.03 * H) cur.push(r); else { bands.push(cur); cur = [r]; } }
    bands.push(cur);
    const band = bands.reduce((a, b) => (b.at(-1)[0] - b[0][0] > a.at(-1)[0] - a[0][0] ? b : a));
    if (band.at(-1)[0] - band[0][0] < K.MIN_BAND * H || band.length < 4 || band.at(-1)[0] < K.MIN_BOTTOM * H) return null;
    const med = (a) => { const s = [...a].sort((p, q) => p - q); return s[s.length >> 1]; };
    const xa = med(band.map((r) => r[1])), xb = med(band.map((r) => r[2]));
    if (xa > K.CENTER[0] * W || xb < K.CENTER[1] * W) return null;
    const gs = [];
    for (let y = band[0][0]; y <= band.at(-1)[0]; y += 2) for (let x = xa; x < xb; x += 4) gs.push(sat[y * W + x]);
    if (!gs.length || med(gs) > K.MAX_SAT) return null;
    return [xa / W, band[0][0] / H, xb / W, band.at(-1)[0] / H];
  }
}

function zebraRow(mask, W) {
  const s = [], e = [];
  for (let x = 0, on = 0; x <= W; x++) {
    const v = x < W ? mask[x] : 0;
    if (v && !on) s.push(x);
    if (!v && on) e.push(x);
    on = v;
  }
  const keep = s.map((a, i) => e[i] - a).map((w) => w >= 2 && w <= W * 0.12);
  const S = s.filter((_, i) => keep[i]), E = e.filter((_, i) => keep[i]);
  if (S.length < K.MIN_STRIPES) return null;
  const c = S.map((a, i) => (a + E[i]) / 2);
  let best = null;
  for (let i = 0; i < S.length - 1;) {
    let j = i + 1;
    while (j < S.length) {
      const per = c.slice(i, j + 1).slice(1).map((v, q) => v - c[i + q]);
      const m = per.reduce((a, b) => a + b, 0) / per.length, sd = Math.sqrt(per.reduce((a, b) => a + (b - m) ** 2, 0) / per.length);
      const srt = [...per].sort((a, b) => a - b), md = srt[srt.length >> 1];
      if (sd / Math.max(m, 1e-6) > K.PERIOD_CV || per.at(-1) > 3 * md) break;
      j++;
    }
    if (j - i >= K.MIN_STRIPES && (!best || j - i > best[1] - best[0])) best = [i, j];
    i = Math.max(i + 1, j - 1);
  }
  if (!best) return null;
  const [i, j] = best, x0 = S[i], x1 = E[j - 1];
  if (x1 - x0 < K.MIN_SPAN * W) return null;
  let paint = 0;
  for (let q = i; q < j; q++) paint += E[q] - S[q];
  const fill = paint / Math.max(x1 - x0, 1);
  return fill >= 0.25 && fill <= 0.75 ? [x0, x1] : null;
}
