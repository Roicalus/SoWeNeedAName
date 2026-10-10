"""Голосовые предупреждения в отдельном потоке, чтобы не тормозить обработку видео.

Перед фразой звучит короткий сигнал (sounds/beep.wav, для критичного — beep_crit.wav, см. beeps.py):
водитель реагирует на него раньше, чем дослушает фразу.
Если в папке sounds/ лежит <тип>.wav (см. make_sounds.py), играем его — это быстрее и надёжнее.
Иначе говорим через pyttsx3.
"""
import itertools
import os
import queue
import shutil
import subprocess
import threading
import time

import config as C

try:
    import winsound
except ImportError:      # не Windows
    winsound = None

# проигрыватель .wav вне Windows: macOS — afplay, Linux — aplay
PLAYER = shutil.which("afplay") or shutil.which("aplay")


def play_wav(path):
    """Проиграть .wav и дождаться конца. False — если играть нечем или файла нет."""
    if not os.path.exists(path):
        return False
    if winsound:
        winsound.PlaySound(path, winsound.SND_FILENAME)
    elif PLAYER:
        subprocess.run([PLAYER, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        return False
    return True


def beep_file(level, sounds_dir="sounds"):
    return os.path.join(sounds_dir, "beep_crit.wav" if level >= C.BEEP_CRIT_LEVEL else "beep.wav")


def make_engine():
    import pyttsx3
    engine = pyttsx3.init()
    for v in engine.getProperty("voices"):
        if "ru" in v.id.lower() or "irina" in v.name.lower() or "russian" in v.name.lower():
            engine.setProperty("voice", v.id)
            break
    engine.setProperty("rate", 190)
    return engine


class Voice:
    def __init__(self, enabled=True, sounds_dir="sounds"):
        self.enabled = enabled
        self.sounds_dir = sounds_dir
        self.q = queue.PriorityQueue()
        self.seq = itertools.count()
        self.last = {}
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, alert):
        if not self.enabled or alert is None:
            return
        now = time.monotonic()
        if now - self.last.get(alert.kind, -1e9) < C.VOICE_COOLDOWN[alert.kind]:
            return
        self.last[alert.kind] = now
        print(f"[{time.strftime('%H:%M:%S')}] {alert.screen}: {alert.voice}", flush=True)
        # сначала самые опасные, при равенстве — в порядке поступления
        self.q.put((-alert.level, next(self.seq), now, alert))

    def _run(self):
        engine = None
        while True:
            _, _, born, alert = self.q.get()
            if time.monotonic() - born > C.VOICE_MAX_DELAY:
                continue
            wav = os.path.join(self.sounds_dir, f"{alert.kind}.wav")
            try:
                if C.BEEP:
                    play_wav(beep_file(alert.level, self.sounds_dir))
                if not play_wav(wav):
                    engine = engine or make_engine()
                    engine.say(alert.voice)
                    engine.runAndWait()
            except Exception as e:     # голос не должен ронять программу
                print(f"[voice] {e}")
