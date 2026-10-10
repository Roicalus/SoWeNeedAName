"""Автозапись событий: держим в памяти последние секунды и при опасности сохраняем ролик."""
import os
import time
from collections import deque

import cv2

import config as C


class EventRecorder:
    def __init__(self, fps, out_dir=C.EVENTS_DIR):
        self.fps = fps
        self.out_dir = out_dir
        self.buf = deque(maxlen=int(C.RECORD_PRE * fps))   # кадры в JPEG, чтобы не съесть память
        self.writer = None
        self.left = 0
        self.last_path = None

    @property
    def recording(self):
        return self.writer is not None

    def push(self, frame, trigger=False, tag="event"):
        if self.writer is None:
            self.buf.append(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1])
            if trigger:
                self._start(frame.shape, tag)
            return
        self.writer.write(frame)
        self.left = int(C.RECORD_POST * self.fps) if trigger else self.left - 1
        if self.left <= 0:
            self.close()

    def _start(self, shape, tag):
        os.makedirs(self.out_dir, exist_ok=True)
        self.last_path = os.path.join(self.out_dir, f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}.mp4")
        h, w = shape[:2]
        self.writer = cv2.VideoWriter(self.last_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        for jpg in self.buf:
            self.writer.write(cv2.imdecode(jpg, cv2.IMREAD_COLOR))
        self.buf.clear()
        self.left = int(C.RECORD_POST * self.fps)
        print(f"[rec] сохраняю событие: {self.last_path}")

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
