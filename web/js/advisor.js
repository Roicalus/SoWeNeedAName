// Port of advisor.py: colour, side and spoken cue, without crying wolf. Keep in sync with advisor.py.
//   green  (1) - glow + a calm phrase at most every HINT_GAP[1] s
//   yellow (2) - glow + chime + phrase at most every HINT_GAP[2] s
//   red    (3) - strong glow + urgent chime + phrase at once (no budget)
const SIDE_WORDS = { left: "on the left", right: "on the right", ahead: "ahead" };

export class Advisor {
  constructor(C) {
    this.C = C;
    this.A = C.ADVISOR;
    this.lastLevel = { 1: -1e9, 2: -1e9 };
    this.lastKind = new Map();
  }

  sideOf(box, W) {
    if (!box) return "ahead";
    const dx = (box[0] + box[2]) / 2 / W - 0.5;
    return Math.abs(dx) < this.A.AHEAD_DX ? "ahead" : dx < 0 ? "left" : "right";
  }

  phrase(kind, side) {
    let text = this.C.ALERTS[kind][2];
    if (this.A.SIDED.includes(kind) && (side === "left" || side === "right")) text = text.replace(/[!.]+$/, "") + " " + SIDE_WORDS[side];
    return text;
  }

  update(t, res, W) {
    const C = this.C, adv = { level: 0, side: null, glow: 0, cue: null, kinds: res.alerts.map((a) => a.kind).sort() };
    const top = res.top;
    if (!top) return adv;
    const ths = res.threats.filter((th) => th.kind === top.kind);
    const box = ths.length ? ths.reduce((a, b) => (b.level > a.level ? b : a)).det.box : res.lead ? res.lead.box : null;
    Object.assign(adv, { level: top.level, side: this.sideOf(box, W), glow: { 1: 0.35, 2: 0.7, 3: 1 }[top.level] });
    if (t - (this.lastKind.get(top.kind) ?? -1e9) < C.VOICE_COOLDOWN[top.kind]) return adv;
    if (top.level < 3) {
      if (top.level === 1 && !C.SPEAK_GREEN) return adv;
      const last = Math.max(...Object.keys(this.lastLevel).map(Number).filter((lv) => lv >= top.level).map((lv) => this.lastLevel[lv]));
      if (t - last < C.HINT_GAP[top.level]) return adv;
    }
    this.lastKind.set(top.kind, t);
    for (const lv of Object.keys(this.lastLevel)) if (Number(lv) <= top.level) this.lastLevel[lv] = t;
    adv.cue = { kind: top.kind, level: top.level, phrase: this.phrase(top.kind, adv.side), side: adv.side };
    return adv;
  }
}
