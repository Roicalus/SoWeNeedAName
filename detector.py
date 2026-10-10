"""Детекция и трекинг: YOLO (ONNX Runtime, без PyTorch) + собственный трекер.

ONNX нужен, потому что Smart App Control в Windows блокирует библиотеки PyTorch.
Бонус: тот же .onnx файл потом переносится на Raspberry Pi / чип регистратора.
"""
import cv2
import numpy as np
import onnxruntime as ort

import config as C
from behavior import Detection
from tracker import Tracker


class Detector:
    def __init__(self, model=C.MODEL):
        so = ort.SessionOptions()
        so.intra_op_num_threads = C.ORT_THREADS
        self.sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name, self.size = inp.name, inp.shape[2]
        self.dtype = np.float16 if "float16" in inp.type else np.float32
        self.classes = np.array(list(C.VEHICLE_CLASSES) + [C.PERSON_CLASS])
        self.tracker = Tracker()

    def detect(self, frame):
        """Возвращает массив N x 6: [x1, y1, x2, y2, уверенность, класс COCO] в пикселях кадра."""
        h, w = frame.shape[:2]
        r = self.size / max(h, w)
        nw, nh = round(w * r), round(h * r)
        dx, dy = (self.size - nw) // 2, (self.size - nh) // 2
        img = np.full((self.size, self.size, 3), 114, np.uint8)           # letterbox
        img[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True).astype(self.dtype)

        out = self.sess.run(None, {self.input_name: blob})[0][0].T.astype(np.float32)   # 8400 x (4 + 80)
        scores = out[:, 4:][:, self.classes]
        best = scores.argmax(1)
        conf = scores[np.arange(len(scores)), best]
        keep = conf >= C.TRACK_LOW_CONF
        if not keep.any():
            return np.zeros((0, 6))
        cx, cy, bw, bh = out[keep, :4].T
        conf, cls = conf[keep], self.classes[best[keep]]

        x1 = np.clip((cx - bw / 2 - dx) / r, 0, w)
        y1 = np.clip((cy - bh / 2 - dy) / r, 0, h)
        x2 = np.clip((cx + bw / 2 - dx) / r, 0, w)
        y2 = np.clip((cy + bh / 2 - dy) / r, 0, h)

        groups = (cls == C.PERSON_CLASS).astype(np.int32)   # NMS отдельно для машин и для людей
        xywh = np.stack([x1, y1, x2 - x1, y2 - y1], 1)
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), conf.tolist(), groups.tolist(),
                                      C.TRACK_LOW_CONF, C.NMS_IOU)
        idx = np.array(idx, dtype=int).reshape(-1)
        return np.stack([x1, y1, x2, y2, conf, cls], 1)[idx]

    def __call__(self, frame):
        return [Detection(box, cls, tid) for box, cls, tid in self.tracker.update(self.detect(frame))]

    def reset(self):
        """Новый ролик — сбрасываем трекер."""
        self.tracker = Tracker()
