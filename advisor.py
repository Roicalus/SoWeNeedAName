"""How to tell the driver: colour, side and a spoken cue - without crying wolf.

BehaviorEngine decides WHAT is happening. Advisor decides HOW to say it:

  GREEN  (1, attention)    - soft glow on the side of the hazard; a calm phrase at most every
                             HINT_GAP[1] seconds (pedestrian at a crosswalk, car drifting, too close).
  YELLOW (2, possible risk) - glow + chime + short phrase, at most every HINT_GAP[2] seconds
                             (cut-in, car ahead braking hard, pedestrian heading into our lane...).
  RED    (3, danger)       - strong glow + urgent chime + phrase immediately, no budget
                             (Brake!, crash ahead, wrong-way car, cut-in at < 2 s).
The same phrase never repeats faster than config.VOICE_COOLDOWN[kind]. Frequent and false hints are
the fastest way to teach a driver to ignore the assistant, so only the most important alert speaks.
The module is pure (no sound, no window): the app, evaluate.py and debug_video.py share it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config as C

AHEAD_DX = 0.15         # |centre - 0.5| below this (frame fraction): "ahead", not left/right
SIDE_WORDS = {"left": "on the left", "right": "on the right", "ahead": "ahead"}
SIDED = {"ped", "ped_xwalk", "vru_near", "cutin", "drift", "crossing", "aggressive"}   # phrases that get "on the left/right"


@dataclass
class Advice:
    level: int = 0                # 0 clear, 1 green, 2 yellow, 3 red
    side: str | None = None       # "left" | "right" | "ahead"
    glow: float = 0.0             # glow strength 0..1
    cue: dict | None = None       # what to say now: {"kind", "level", "phrase", "side"}
    kinds: list = field(default_factory=list)


def side_of(box, W):
    if box is None:
        return "ahead"
    dx = (box[0] + box[2]) / 2 / W - 0.5
    return "ahead" if abs(dx) < AHEAD_DX else ("left" if dx < 0 else "right")


def phrase(kind, side):
    text = C.ALERTS[kind][2]
    if kind in SIDED and side in ("left", "right"):
        text = text.rstrip("!.") + " " + SIDE_WORDS[side]
    return text


class Advisor:
    def __init__(self):
        self.last_level_hint = {1: -1e9, 2: -1e9}
        self.last_kind: dict[str, float] = {}

    def update(self, t, res, W) -> Advice:
        adv = Advice(kinds=sorted(a.kind for a in res.alerts))
        top = res.top
        if top is None:
            return adv
        # the object behind the top alert: its most severe threat of that kind
        ths = [th for th in res.threats if th.kind == top.kind]
        box = max(ths, key=lambda th: th.level).det.box if ths else (res.lead.box if res.lead else None)
        adv.level, adv.side = top.level, side_of(box, W)
        adv.glow = {1: 0.35, 2: 0.7, 3: 1.0}[top.level]
        if top.kind in C.SILENT_KINDS or t - self.last_kind.get(top.kind, -1e9) < C.VOICE_COOLDOWN[top.kind]:
            return adv
        if top.level < 3:
            if top.level == 1 and not C.SPEAK_GREEN:
                return adv
            if t - max(self.last_level_hint[lv] for lv in self.last_level_hint if lv >= top.level) < C.HINT_GAP[top.level]:
                return adv
        self.last_kind[top.kind] = t
        for lv in self.last_level_hint:
            if lv <= top.level:
                self.last_level_hint[lv] = t     # a louder cue also uses up the quieter budgets
        adv.cue = {"kind": top.kind, "level": top.level, "phrase": phrase(top.kind, adv.side), "side": adv.side}
        return adv
