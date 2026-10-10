"""Короткие звуковые сигналы «внимание, опасность» — звучат сразу, до голосовой фразы.

Сигнал короче и быстрее речи: водитель реагирует на него раньше, чем дослушает фразу.
    beep.wav      — опасность (уровни 1–2): два коротких тона по возрастающей
    beep_crit.wav — критично (уровень 3, «Тормозите!»): три частых высоких тона
Пересоздать: python beeps.py
"""
import os
import wave

import numpy as np

RATE = 22050      # как у голосовых фраз в sounds/ — их можно склеивать без пересчёта


def tone(freq, dur, vol=0.6):
    t = np.arange(int(RATE * dur)) / RATE
    env = np.minimum(1, np.minimum(t, dur - t) / 0.008)          # мягкие края, без щелчков
    # основной тон + немного второй гармоники — звучит как сигнал, а не как писк
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
    make(os.path.join(out, "beep.wav"), [tone(880, 0.09), gap(0.04), tone(1320, 0.12), gap(0.05)])
    make(os.path.join(out, "beep_crit.wav"), [tone(1760, 0.07), gap(0.04)] * 3 + [gap(0.03)])
    print("sounds/beep.wav, sounds/beep_crit.wav")


if __name__ == "__main__":
    main()
