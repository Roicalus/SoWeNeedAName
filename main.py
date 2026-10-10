"""Co-pilot - a driver assistant that watches the road and warns before things get dangerous.

Run:   python main.py data/videos/clip.mp4
       python main.py 0            (camera)
Keys:  Esc/q - quit, space - pause, c - show/hide the lane, v - voice on/off.

The video is shown smoothly in real time while the networks run in a separate thread and always
take the newest frame. Real dashcams and ADAS units work the same way: analysis may be less frequent
than frames, but it never lags behind the road.
"""
import argparse
import os
import threading
import time

import cv2

import config as C
import crosswalk
import ui
from advisor import Advisor
from behavior import BehaviorEngine, FrameResult
from detector import Detector
from egomotion import EgoMotion
from lanes import LaneDetector
from recorder import EventRecorder
from voice import Voice


class Analyzer:
    """Background thread: detection + lanes + crosswalk + own motion + behaviour on the newest frame."""

    def __init__(self, model=None, can_speed=None):
        self.det = Detector(model)
        self.lanes = LaneDetector() if os.path.exists(C.LANE_MODEL) else None
        if self.lanes is None:
            print(f"No lane model {C.LANE_MODEL} - own lane from a trapezoid (see lanes.py)")
        self.can_speed = can_speed
        self._new()
        self.cond = threading.Condition()
        self.pending = None               # (t, frame) not processed yet
        self.rate = None                  # analyses per second
        threading.Thread(target=self._run, daemon=True).start()

    def _new(self):
        self.engine, self.ego, self.advisor = BehaviorEngine(), EgoMotion(), Advisor()
        self.result, self.advice, self.n = FrameResult(), None, 0

    def submit(self, t, frame):
        with self.cond:
            self.pending = (t, frame)     # an older unprocessed frame is simply replaced
            self.cond.notify()

    def reset(self):
        with self.cond:
            self.pending = None
            self.det.reset()
            self._new()

    def step(self, t, frame):
        self.n += 1
        dets = self.det(frame)
        boxes = [d.box for d in dets]
        # markings change slowly - on weak hardware they need not be searched on every analysis
        lanes = self.lanes.detect(frame) if self.lanes and self.n % C.LANE_EVERY == 0 else None
        ego = self.ego.update(t, frame, boxes, can_speed=self.can_speed)
        kw = {}
        if C.CROSSWALK and self.n % C.CROSSWALK_EVERY == 0:
            kw["crosswalk"] = crosswalk.detect(frame, boxes)
        res = self.engine.update(t, frame.shape[1], frame.shape[0], dets, frame, lanes, ego, **kw)
        return res, self.advisor.update(t, res, frame.shape[1])

    def _run(self):
        while True:
            with self.cond:
                while self.pending is None:
                    self.cond.wait()
                (t, frame), self.pending = self.pending, None
            tic = time.perf_counter()
            res, advice = self.step(t, frame)
            dt = max(time.perf_counter() - tic, 1e-6)
            self.rate = 1 / dt if self.rate is None else 0.9 * self.rate + 0.1 / dt
            self.result, self.advice = res, advice


def main():
    ap = argparse.ArgumentParser(description="Co-pilot - road hazard assistant")
    ap.add_argument("source", help="video path or camera number (0)")
    ap.add_argument("--model", default=None, help="detector .onnx (default config.MODEL)")
    ap.add_argument("--start", type=float, default=0, help="start at second N of the video")
    ap.add_argument("--can-speed", type=float, help="own speed from the vehicle bus, km/h (else from video)")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-record", action="store_true", help="do not save event clips")
    ap.add_argument("--loop", action="store_true", help="loop the video")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--debug", action="store_true", help="print video time and analysis rate")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    is_file = isinstance(src, str)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 1 <= fps <= 120:
        fps = 30.0
    if is_file and args.start:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)

    analyzer = Analyzer(args.model, args.can_speed / 3.6 if args.can_speed is not None else None)
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
    shown, last_advice = None, None

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
                print(f"video {t:6.1f} s | wall {time.monotonic() - t0:6.1f} s | analysis {analyzer.rate or 0:4.1f}/s",
                      flush=True)
            analyzer.submit(t, frame.copy())             # copy: the UI is drawn over frame now

            res, advice = analyzer.result, analyzer.advice
            if advice is not None and advice is not last_advice:
                voice.say(advice.cue)
                last_advice = advice
            ui.draw(frame, res, show_corridor, analyzer.rate, recording=bool(rec and rec.recording), advice=advice)
            if rec:
                rec.push(frame, trigger=res.level >= C.RECORD_LEVEL, tag=res.top.kind if res.top else "event")
            shown = frame

        cv2.imshow(win, shown)

        delay = 30
        if not paused:
            delay = 1
            if is_file:                                  # do not run ahead of real time
                ahead = t - (time.monotonic() - t0)
                if ahead < -0.5:                         # far behind (slow decoder) - skip frames
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
                t0 = time.monotonic() - t                # resume without a jump
        elif key == ord("c"):
            show_corridor = not show_corridor
        elif key == ord("v"):
            voice.enabled = not voice.enabled
            print(f"Voice: {'on' if voice.enabled else 'off'}")
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    if rec:
        rec.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
