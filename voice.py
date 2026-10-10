"""Spoken cues in a separate thread so video processing never waits for sound.

The Advisor (advisor.py) decides what to say and when; this module only plays it:
a chime for the level (beeps.py: chime / beep / beep_crit), then the phrase. Pre-recorded
sounds/<kind>_<side>.wav or sounds/<kind>.wav (make_sounds.py) play faster and more reliably than
speech synthesis; without them pyttsx3 speaks the text. Stale phrases (waited > VOICE_MAX_DELAY) are
dropped - a late warning is worse than none.
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
except ImportError:      # not Windows
    winsound = None

# .wav player outside Windows: macOS - afplay, Linux - aplay
PLAYER = shutil.which("afplay") or shutil.which("aplay")
CHIMES = {1: "chime.wav", 2: "beep.wav", 3: "beep_crit.wav"}


def play_wav(path):
    """Play a .wav and wait for it to end. False if there is nothing to play it with or no file."""
    if not os.path.exists(path):
        return False
    if winsound:
        winsound.PlaySound(path, winsound.SND_FILENAME)
    elif PLAYER:
        subprocess.run([PLAYER, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        return False
    return True


def chime_file(level, sounds_dir="sounds"):
    return os.path.join(sounds_dir, CHIMES.get(level, "beep.wav"))


def phrase_file(cue, sounds_dir="sounds"):
    """Pre-recorded phrase for a cue: <kind>_<side>.wav, else <kind>.wav, else None."""
    for name in (f"{cue['kind']}_{cue['side']}.wav", f"{cue['kind']}.wav"):
        p = os.path.join(sounds_dir, name)
        if os.path.exists(p):
            return p
    return None


def make_engine():
    import pyttsx3
    engine = pyttsx3.init()
    want = "en" if C.LANG == "en" else C.LANG
    for v in engine.getProperty("voices"):
        name = (v.name + " " + v.id).lower()
        if (want == "en" and ("english" in name or "zira" in name or "david" in name or "en-us" in name)) or \
                (want == "ru" and ("russian" in name or "irina" in name)):
            engine.setProperty("voice", v.id)
            engine.chosen = v.name          # getProperty("voice") reports the change only after runAndWait
            break
    engine.setProperty("rate", 185)
    return engine


class Voice:
    def __init__(self, enabled=True, sounds_dir="sounds"):
        self.enabled = enabled
        self.sounds_dir = sounds_dir
        self.q = queue.PriorityQueue()
        self.seq = itertools.count()
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, cue):
        """cue - Advice.cue from advisor.py (or None)."""
        if not self.enabled or cue is None:
            return
        print(f"[{time.strftime('%H:%M:%S')}] {C.LEVEL_NAMES[cue['level']].upper()}: {cue['phrase']}", flush=True)
        # most dangerous first, then in arrival order
        self.q.put((-cue["level"], next(self.seq), time.monotonic(), cue))

    def _run(self):
        engine = None
        while True:
            _, _, born, cue = self.q.get()
            if time.monotonic() - born > C.VOICE_MAX_DELAY:
                continue
            try:
                if C.BEEP:
                    play_wav(chime_file(cue["level"], self.sounds_dir))
                wav = phrase_file(cue, self.sounds_dir)
                if not (wav and play_wav(wav)):
                    engine = engine or make_engine()
                    engine.say(cue["phrase"])
                    engine.runAndWait()
            except Exception as e:     # sound must never crash the app
                print(f"[voice] {e}")
