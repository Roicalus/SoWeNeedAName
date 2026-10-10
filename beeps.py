"""Short chimes that play right away, before the spoken phrase.

A chime is shorter than speech: the driver reacts to it before the phrase is over.
    chime.wav     - green (attention): one soft low tone
    beep.wav      - yellow (possible risk): two short rising tones
    beep_crit.wav - red (danger, "Brake!"): three fast high tones
Re-create: python beeps.py
"""
import os
import wave

import numpy as np

RATE = 22050      # same as the voice phrases in sounds/ - they can be concatenated as is


def tone(freq, dur, vol=0.6):
    t = np.arange(int(RATE * dur)) / RATE
    env = np.minimum(1, np.minimum(t, dur - t) / 0.008)          # soft edges, no clicks
    # base tone + a bit of the second harmonic: sounds like a signal, not a squeak
    return vol * env * (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)) / 1.3


def gap(dur):
    return np.zeros(int(RATE * dur))


def make(path, parts):
    pcm = (np.concatenate(parts) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def main(out="sounds"):
    os.makedirs(out, exist_ok=True)
    make(os.path.join(out, "chime.wav"), [tone(660, 0.16, 0.35), gap(0.05)])
    make(os.path.join(out, "beep.wav"), [tone(880, 0.09), gap(0.04), tone(1320, 0.12), gap(0.05)])
    make(os.path.join(out, "beep_crit.wav"), [tone(1760, 0.07), gap(0.04)] * 3 + [gap(0.03)])
    print("sounds/chime.wav, sounds/beep.wav, sounds/beep_crit.wav")


if __name__ == "__main__":
    main()
