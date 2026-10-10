"""Record the alert phrases into sounds/<kind>.wav and sounds/<kind>_<side>.wav (speech synthesis).

Pre-recorded .wav files play faster and more reliably than synthesis on the fly. Replace them with
real voice recordings (same file names) for a better demo or another language.
    python make_sounds.py
"""
import os

import config as C
from advisor import SIDED, phrase
from voice import make_engine


def main():
    os.makedirs("sounds", exist_ok=True)
    engine = make_engine()
    print("Voice:", getattr(engine, "chosen", "system default"))
    jobs = []
    for kind in C.ALERTS:
        jobs.append((f"{kind}.wav", phrase(kind, "ahead")))
        if kind in SIDED:
            for side in ("left", "right"):
                jobs.append((f"{kind}_{side}.wav", phrase(kind, side)))
    for name, text in jobs:
        path = os.path.join("sounds", name)
        engine.save_to_file(text, path)
        print(f"{path}: {text}")
    engine.runAndWait()


if __name__ == "__main__":
    main()
