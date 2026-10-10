"""Записывает фразы предупреждений в sounds/<тип>.wav.

Готовые .wav играют быстрее и надёжнее, чем синтез на лету. Можно заменить их своими
записями голоса (тот же формат имён) — звучит гораздо лучше для демо.
"""
import os

import config as C
from voice import make_engine


def main():
    os.makedirs("sounds", exist_ok=True)
    engine = make_engine()
    print("Голос:", engine.getProperty("voice"))
    for kind, (_, _, phrase) in C.ALERTS.items():
        path = os.path.join("sounds", f"{kind}.wav")
        engine.save_to_file(phrase, path)
        print(f"{path}: {phrase}")
    engine.runAndWait()


if __name__ == "__main__":
    main()
