"""Второй пилот — умный видеорегистратор-помощник водителя.

Запуск:  python main.py data/videos/ролик.mov
         python main.py 0            (веб-камера)
Клавиши: Esc/q — выход, пробел — пауза, c — показать/скрыть полосу, v — вкл/выкл голос.

Видео показывается плавно в реальном времени, а нейросеть работает в отдельном потоке
и всегда берёт самый свежий кадр. Так устроены и настоящие регистраторы: анализ может
быть реже кадров, но никогда не отстаёт от дороги.
"""
import argparse
import threading
import time

import cv2

import config as C
import ui
from behavior import BehaviorEngine, FrameResult
from detector import Detector
from recorder import EventRecorder
from voice import Voice


class Analyzer:
    """Фоновый поток: детекция + анализ поведения над самым свежим кадром."""

    def __init__(self, model):
        self.det = Detector(model)
        self.engine = BehaviorEngine()
        self.cond = threading.Condition()
        self.pending = None               # (t, кадр), ещё не обработанный
        self.result = FrameResult()
        self.rate = None                  # анализов в секунду
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, t, frame):
        with self.cond:
            self.pending = (t, frame)     # старый необработанный кадр просто заменяется
            self.cond.notify()

    def reset(self):
        with self.cond:
            self.pending = None
            self.det.reset()
            self.engine = BehaviorEngine()
            self.result = FrameResult()

    def _run(self):
        while True:
            with self.cond:
                while self.pending is None:
                    self.cond.wait()
                (t, frame), self.pending = self.pending, None
                det, engine = self.det, self.engine
            tic = time.perf_counter()
            res = engine.update(t, frame.shape[1], frame.shape[0], det(frame), frame)
            dt = max(time.perf_counter() - tic, 1e-6)
            self.rate = 1 / dt if self.rate is None else 0.9 * self.rate + 0.1 / dt
            self.result = res


def main():
    ap = argparse.ArgumentParser(description="Второй пилот — умный видеорегистратор")
    ap.add_argument("source", help="путь к видео или номер камеры (0)")
    ap.add_argument("--model", default=C.MODEL, help="путь к .onnx модели")
    ap.add_argument("--start", type=float, default=0, help="начать с N-й секунды видео")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-record", action="store_true", help="не сохранять ролики событий")
    ap.add_argument("--loop", action="store_true", help="крутить видео по кругу")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--debug", action="store_true", help="печатать время видео и скорость анализа")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    is_file = isinstance(src, str)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть источник: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 1 <= fps <= 120:
        fps = 30.0
    if is_file and args.start:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)

    analyzer = Analyzer(args.model)
    voice = Voice(enabled=not args.no_voice)
    rec = None if args.no_record else EventRecorder(fps)

    win = "Co-pilot"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(win, 960, 540)

    show_corridor, paused = True, False
    i, t, t0 = 0, 0.0, time.monotonic()
    shown = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                if is_file and args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    analyzer.reset()
                    i, t0 = 0, time.monotonic()
                    continue
                break
            i += 1
            t = i / fps if is_file else time.monotonic() - t0
            if args.debug and i % int(fps) == 0:
                print(f"видео {t:6.1f} с | прошло {time.monotonic() - t0:6.1f} с | анализ {analyzer.rate or 0:4.1f}/с", flush=True)
            analyzer.submit(t, frame.copy())             # копия: поверх frame сейчас будет рисоваться интерфейс

            res = analyzer.result
            voice.say(res.top)
            ui.draw(frame, res, show_corridor, analyzer.rate, recording=bool(rec and rec.recording))
            if rec:
                rec.push(frame, trigger=res.level >= C.RECORD_LEVEL, tag=res.top.kind if res.top else "event")
            shown = frame

        cv2.imshow(win, shown)

        delay = 30
        if not paused:
            delay = 1
            if is_file:                                  # не обгонять реальное время
                ahead = t - (time.monotonic() - t0)
                if ahead < -0.5:                         # сильно отстали (медленный декодер) — пропускаем кадры
                    skip = int(-ahead * fps)
                    for _ in range(skip):
                        cap.grab()
                    i += skip
                delay = max(1, int(ahead * 1000))
        key = cv2.waitKey(delay) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord(" "):
            paused = not paused
            if not paused:
                t0 = time.monotonic() - t                # продолжить без рывка
        elif key == ord("c"):
            show_corridor = not show_corridor
        elif key == ord("v"):
            voice.enabled = not voice.enabled
            print(f"Голос: {'вкл' if voice.enabled else 'выкл'}")
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    if rec:
        rec.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
