// YOLO11n in the browser (onnxruntime-web): WebGPU when the device has it, WASM otherwise.
// Same pre/post-processing as detector.py: letterbox, class filter, NMS per group (vehicles / people / bicycles).

const ort = globalThis.ort;

export class Detector {
  constructor(C) {
    this.C = C;
    this.vehicle = new Set(Object.keys(C.VEHICLE_CLASSES).map(Number));
    this.classes = [...Object.keys(C.VEHICLE_CLASSES).map(Number), C.PERSON_CLASS, C.BICYCLE_CLASS];
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
  }

  async load(url, size) {
    this.size = size;
    this.canvas.width = this.canvas.height = size;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    const tries = "gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"];
    for (const ep of tries) {
      try {
        this.session = await ort.InferenceSession.create(url, { executionProviders: [ep], graphOptimizationLevel: "all" });
        this.backend = ep;
        break;
      } catch (e) { console.warn(`[detector] ${ep} failed`, e); }
    }
    if (!this.session) throw new Error("Could not start the model on this device");
    await this.detect(document.createElement("canvas"), 16, 9);       // warm-up (shader compilation)
    return this.backend;
  }

  // source: video or canvas; returns [[x1, y1, x2, y2, conf, cls]] in source pixels
  async detect(source, w, h) {
    const C = this.C, S = this.size, r = S / Math.max(w, h);
    const nw = Math.round(w * r), nh = Math.round(h * r), dx = (S - nw) >> 1, dy = (S - nh) >> 1;
    this.ctx.fillStyle = "rgb(114,114,114)";
    this.ctx.fillRect(0, 0, S, S);
    if (w > 16) this.ctx.drawImage(source, dx, dy, nw, nh);
    const px = this.ctx.getImageData(0, 0, S, S).data, n = S * S;
    const input = new Float32Array(3 * n);
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      input[i] = px[j] / 255; input[n + i] = px[j + 1] / 255; input[2 * n + i] = px[j + 2] / 255;
    }
    const out = await this.session.run({ [this.session.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, S, S]) });
    const y = out[this.session.outputNames[0]], data = y.data, A = y.dims[2];   // [1, 84, A]
    const boxes = [];
    for (let a = 0; a < A; a++) {
      let best = -1, conf = 0;
      for (const c of this.classes) { const s = data[(4 + c) * A + a]; if (s > conf) { conf = s; best = c; } }
      if (conf < C.TRACK_LOW_CONF) continue;
      const cx = data[a], cy = data[A + a], bw = data[2 * A + a], bh = data[3 * A + a];
      boxes.push([
        Math.min(Math.max((cx - bw / 2 - dx) / r, 0), w), Math.min(Math.max((cy - bh / 2 - dy) / r, 0), h),
        Math.min(Math.max((cx + bw / 2 - dx) / r, 0), w), Math.min(Math.max((cy + bh / 2 - dy) / r, 0), h), conf, best]);
    }
    return nms(boxes, C.NMS_IOU, (c) => (this.vehicle.has(c) ? 0 : c === C.PERSON_CLASS ? 1 : 2));
  }
}

function nms(boxes, thr, group) {
  boxes.sort((a, b) => b[4] - a[4]);
  const keep = [];
  for (const b of boxes) {
    let ok = true;
    for (const k of keep) {
      if (group(k[5]) !== group(b[5])) continue;
      const x1 = Math.max(k[0], b[0]), y1 = Math.max(k[1], b[1]), x2 = Math.min(k[2], b[2]), y2 = Math.min(k[3], b[3]);
      const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      const u = (k[2] - k[0]) * (k[3] - k[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
      if (inter / (u + 1e-9) > thr) { ok = false; break; }
    }
    if (ok) keep.push(b);
  }
  return keep;
}
