// Audio cues: a chime panned towards the hazard (left / right / centre), then a short phrase.
//   green - one soft low tone, yellow - two rising tones, red - three fast high tones (always centred).
// Browsers only allow audio after a user gesture, so unlock() is called from the first tap.

export class Sound {
  constructor() {
    this.ctx = null;
    this.enabled = true;
    this.voice = null;
  }

  unlock() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === "suspended") this.ctx.resume();
    if ("speechSynthesis" in window) {
      const pick = () => { this.voice = speechSynthesis.getVoices().find((v) => v.lang.startsWith("en")); };
      pick();
      speechSynthesis.onvoiceschanged = pick;
      speechSynthesis.speak(new SpeechSynthesisUtterance(""));          // iOS: unlock speech too
    }
  }

  tone(freq, start, dur, gain, pan) {
    const c = this.ctx, o = c.createOscillator(), g = c.createGain(), p = c.createStereoPanner();
    o.type = "sine";
    o.frequency.value = freq;
    g.gain.setValueAtTime(0, start);
    g.gain.linearRampToValueAtTime(gain, start + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
    p.pan.value = pan;
    o.connect(g).connect(p).connect(c.destination);
    o.start(start);
    o.stop(start + dur + 0.02);
  }

  // cue: {kind, level, phrase, side}
  play(cue) {
    if (!this.enabled || !this.ctx) return;
    const pan = cue.side === "left" ? -0.8 : cue.side === "right" ? 0.8 : 0;
    const t = this.ctx.currentTime + 0.01;
    if (cue.level >= 3) {
      for (let i = 0; i < 3; i++) this.tone(1480, t + i * 0.11, 0.08, 0.45, 0);
    } else if (cue.level === 2) {
      this.tone(880, t, 0.1, 0.25, pan);
      this.tone(1320, t + 0.13, 0.14, 0.25, pan);
    } else {
      this.tone(660, t, 0.2, 0.12, pan);
    }
    if ("speechSynthesis" in window && cue.phrase) {
      if (cue.level >= 3) speechSynthesis.cancel();                      // urgent: interrupt anything else
      else if (speechSynthesis.speaking) return;                         // never queue hints behind each other
      const u = new SpeechSynthesisUtterance(cue.phrase);
      u.lang = "en-US";
      if (this.voice) u.voice = this.voice;
      u.rate = cue.level >= 3 ? 1.15 : 1.0;
      u.volume = cue.level >= 2 ? 1 : 0.7;
      setTimeout(() => speechSynthesis.speak(u), cue.level >= 3 ? 280 : 350);
    }
  }
}
