"""Дообучение YOLO11s на BDD100K (ночь, плотный трафик) с сохранением COCO-разметки классов.

Зачем: COCO-модель ночью почти не видит машины (фары, засветка, тёмные силуэты). BDD100K — 80 тыс.
кадров с регистраторов США, ~40 % ночных. Учим на нём, но выход оставляем 4 + 80 каналов
и COCO-номера классов, поэтому новый .onnx — прямая замена в detector.py (ничего менять не нужно).

Данные (официальный сайт BDD100K не работает, берём открытые копии с Hugging Face, без логина):
    кадры:        https://huggingface.co/datasets/Xing1210/bdd100k  (один zip 5.7 ГБ, все 100 тыс. кадров)
    рамки:        https://huggingface.co/datasets/dronefreak/BDD100K
                  (только metadata.jsonl по шардам: рамки x,y,w,h в пикселях, 10 классов BDD;
                   train 59 384 / valid 10 479 / test 10 000 = официальный val BDD)
    время суток:  https://huggingface.co/datasets/dronefreak/BDD100K-Period-Classification
                  (метка daytime / night / dawn or dusk для каждого кадра, по имени файла)
Покадровая загрузка с HF упирается в лимит 3000 запросов / 5 мин, поэтому кадры берём одним zip.

Обучение идёт в отдельном окружении (PyTorch нужен только здесь, сам проект работает на ONNX Runtime):
    python3 -m venv .venv-train
    .venv-train/bin/pip install torch torchvision "ultralytics>=8.3,<8.4" onnx onnxslim onnxruntime huggingface_hub

Всё сразу (скачать -> подготовить -> обучить -> оценить -> экспорт в models/yolo11s_bdd.onnx):
    .venv-train/bin/python tools/train_detector.py all --train-n 10000 --epochs 8 --device mps
По шагам:
    .venv-train/bin/python tools/train_detector.py download
    .venv-train/bin/python tools/train_detector.py prep --train-n 0 --night-share 0.4   # 0 = весь train
    .venv-train/bin/python tools/train_detector.py train --epochs 30 --batch 32 --device 0   # 0 = CUDA
    .venv-train/bin/python tools/train_detector.py eval --weights data/bdd_det/weights/yolo11s.pt \
        runs/bdd/yolo11s_bdd/weights/best.pt
    .venv-train/bin/python tools/train_detector.py export
    .venv/bin/python tools/train_detector.py video --onnx models/yolo11s_fp32.onnx models/yolo11s_bdd.onnx

Отложенная проверка — официальный val BDD (10 000 кадров, в обучение не попадает), отдельно день и ночь.
"""
import argparse
import glob
import json
import os
import random
import shutil
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "bdd_det")
ZIP = os.path.join(DATA, "zip", "bdd100k_images_100k.zip")
YOLO = os.path.join(DATA, "yolo")          # подготовленный набор: images/<split>/*.jpg + labels/<split>/*.txt
META = os.path.join(DATA, "meta")          # metadata.jsonl: рамки (det_*) и время суток (period_*)
WEIGHTS = os.path.join(DATA, "weights", "yolo11s.pt")
HF = "https://huggingface.co/datasets/"
ZIP_URL = HF + "Xing1210/bdd100k/resolve/main/bdd100k_images_100k.zip"
DET_URL = HF + "dronefreak/BDD100K/resolve/main/data/images/{split}/shard_{s:03d}/metadata.jsonl"
PERIOD_URL = HF + "dronefreak/BDD100K-Period-Classification/resolve/main/data/images/{split}/shard_{s:03d}/metadata.jsonl"
COCO_PT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
DET_SHARDS = {"train": 8, "valid": 2, "test": 2}
PERIOD_SHARDS = {"train": 9, "valid": 2}
IMG_W, IMG_H = 1280, 720                   # все кадры BDD100K 1280x720
RUNS = os.path.join(ROOT, "runs", "bdd")
RUN_NAME = "yolo11s_bdd"

# класс BDD (как в data.yaml у dronefreak) -> класс COCO; остальные (train, светофор, знак) выбрасываем
BDD_TO_COCO = {
    0: 0,   # person -> person
    1: 0,   # rider (велосипедист/мотоциклист) -> person
    2: 2,   # car -> car
    3: 7,   # truck -> truck
    4: 5,   # bus -> bus
    6: 3,   # motor -> motorcycle
    7: 1,   # bike -> bicycle
}
VEHICLES = [2, 3, 5, 7]          # те же, что C.VEHICLE_CLASSES в config.py
EVAL_CLASSES = [0, 1, 2, 3, 5, 7]


def coco_names():
    """80 имён COCO в порядке номеров (берём из самой модели, чтобы не ошибиться)."""
    from ultralytics import YOLO as Y
    return Y(WEIGHTS).names


def fetch(url, dst):
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print("скачиваю", url)
        urllib.request.urlretrieve(url, dst + ".part")
        os.replace(dst + ".part", dst)


# ---------------------------------------------------------------- скачивание
def download():
    """zip с кадрами (5.7 ГБ), разметку и время суток (несколько jsonl), веса COCO yolo11s.pt."""
    t = time.time()
    fetch(COCO_PT_URL, WEIGHTS)
    for split, n in DET_SHARDS.items():
        for s in range(n):
            fetch(DET_URL.format(split=split, s=s), os.path.join(META, f"det_{split}_{s:03d}.jsonl"))
    for split, n in PERIOD_SHARDS.items():
        for s in range(n):
            fetch(PERIOD_URL.format(split=split, s=s), os.path.join(META, f"period_{split}_{s:03d}.jsonl"))
    fetch(ZIP_URL, ZIP)
    print(f"скачано за {time.time() - t:.0f} с")


def timeofday():
    """имя файла -> 'daytime' / 'night' / 'dawn or dusk' / 'unknown'."""
    tod = {}
    for f in glob.glob(os.path.join(META, "period_*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            tod[d["file_name"]] = d["label"]
    return tod


def boxes(split):
    """имя файла -> список (класс BDD, x, y, w, h в пикселях) для train / valid / test (dronefreak)."""
    res = {}
    for f in sorted(glob.glob(os.path.join(META, f"det_{split}_*.jsonl"))):
        for line in open(f):
            d = json.loads(line)
            res[d["file_name"]] = list(zip(d["objects"]["categories"], d["objects"]["bbox"]))
    return res


# ---------------------------------------------------------------- подготовка
def yolo_lines(objs):
    """Рамки BDD -> строки YOLO с номерами COCO (центр, ширина, высота в долях кадра)."""
    out = []
    for c, (x, y, w, h) in objs:
        if c in BDD_TO_COCO and w > 1 and h > 1:
            out.append(f"{BDD_TO_COCO[c]} {(x + w / 2) / IMG_W:.6f} {(y + h / 2) / IMG_H:.6f} "
                       f"{w / IMG_W:.6f} {h / IMG_H:.6f}")
    return out


def add_split(zf, members, names, split, labels, tod):
    """Распаковывает кадры в yolo/images/<split>/ и пишет метки в yolo/labels/<split>/."""
    di, dl = os.path.join(YOLO, "images", split), os.path.join(YOLO, "labels", split)
    os.makedirs(di, exist_ok=True)
    os.makedirs(dl, exist_ok=True)
    res = []
    for name in names:
        img = os.path.join(di, name)
        if not os.path.exists(img):
            with zf.open(members[name]) as src, open(img, "wb") as dst:
                shutil.copyfileobj(src, dst)
        lines = yolo_lines(labels[name])
        with open(os.path.join(dl, name[:-4] + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        res.append((img, tod.get(name, "unknown")))
    return res


def write_list(path, items):
    with open(path, "w") as f:
        f.write("\n".join(items) + "\n")
    return path


def write_yaml(path, train, val, names):
    with open(path, "w") as f:
        f.write(f"path: {YOLO}\ntrain: {train}\nval: {val}\nnc: {len(names)}\nnames:\n")
        for i in range(len(names)):
            f.write(f"  {i}: {names[i]}\n")
    return path


def prep(train_n, night_share, val_n, train_val_n, seed):
    """Собирает списки кадров. Ночь добираем повтором ночных кадров в train.txt (оверсэмплинг)."""
    rnd = random.Random(seed)
    tod = timeofday()
    for f in glob.glob(os.path.join(YOLO, "*.cache")) + glob.glob(os.path.join(YOLO, "labels", "*.cache")):
        os.remove(f)
    if os.path.exists(os.path.join(YOLO, "labels")):
        shutil.rmtree(os.path.join(YOLO, "labels"))
    # пул для обучения — официальный train BDD (у dronefreak это train + valid), проверка — официальный val
    pool = {**boxes("train"), **boxes("valid")}
    test_boxes = boxes("test")
    zf = zipfile.ZipFile(ZIP)
    members = {os.path.basename(n): n for n in zf.namelist() if n.endswith(".jpg")}
    tr = sorted(n for n in pool if n in members)
    te = sorted(n for n in test_boxes if n in members)
    rnd.shuffle(tr)
    rnd.shuffle(te)
    if train_n:
        tr = tr[:train_n]
    if val_n:
        te = te[:val_n]
    t = time.time()
    train = add_split(zf, members, tr, "train", pool, tod)
    test = add_split(zf, members, te, "val", test_boxes, tod)
    print(f"распаковано за {time.time() - t:.0f} с")

    night = [p for p, t in train if t == "night"]
    lst = [p for p, _ in train]
    share = len(night) / max(1, len(lst))
    if night and share < night_share < 1:  # добавить e ночных: (n + e) / (T + e) = night_share
        extra = round((night_share * len(lst) - len(night)) / (1 - night_share))
        lst += [rnd.choice(night) for _ in range(extra)]
    rnd.shuffle(lst)
    n_night = sum(tod.get(os.path.basename(p)) == "night" for p in lst)
    print(f"train: {len(train)} кадров (ночь {share:.0%}), в списке с оверсэмплингом {len(lst)} "
          f"(ночь {n_night / len(lst):.0%})")

    names = coco_names()
    write_list(os.path.join(YOLO, "train.txt"), lst)
    groups = {"all": [p for p, _ in test],
              "day": [p for p, t in test if t == "daytime"],
              "night": [p for p, t in test if t == "night"],
              "dawn": [p for p, t in test if t == "dawn or dusk"]}
    for g, items in groups.items():
        write_list(os.path.join(YOLO, f"val_{g}.txt"), items)
        write_yaml(os.path.join(YOLO, f"eval_{g}.yaml"), "train.txt", f"val_{g}.txt", names)
        print(f"val {g}: {len(items)}")
    # маленькая проверка во время обучения (полная оценка потом командой eval)
    write_list(os.path.join(YOLO, "val_mini.txt"), groups["all"][:train_val_n])
    write_yaml(os.path.join(YOLO, "train.yaml"), "train.txt", "val_mini.txt", names)


# ---------------------------------------------------------------- обучение
def train(epochs, batch, device, workers, imgsz, weights, resume, hours):
    from ultralytics import YOLO as Y
    t = time.time()
    model = Y(os.path.join(RUNS, RUN_NAME, "weights", "last.pt") if resume else weights)
    model.train(data=os.path.join(YOLO, "train.yaml"), epochs=epochs, imgsz=imgsz, batch=batch,
                device=device, workers=workers, project=RUNS, name=RUN_NAME, exist_ok=True,
                resume=resume, cos_lr=True, close_mosaic=max(1, epochs // 5), plots=False,
                patience=100, amp=True, time=hours or None,
                warmup_epochs=min(3.0, epochs * 0.1))   # аугментации по умолчанию; от готовых весов — короткий разогрев
    print(f"обучение: {(time.time() - t) / 60:.1f} мин")
    return os.path.join(RUNS, RUN_NAME, "weights", "best.pt")


# ---------------------------------------------------------------- оценка
def iou(a, b):
    """IoU матрица для рамок x1y1x2y2: a (N x 4), b (M x 4)."""
    import numpy as np
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), 2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def match(pred, gt, thr=0.5):
    """Жадное сопоставление по убыванию уверенности (как в COCO). pred: K x 5 (x1y1x2y2, conf)."""
    import numpy as np
    tp = np.zeros(len(pred), bool)
    if len(pred) and len(gt):
        m = iou(pred[:, :4], gt)
        used = np.zeros(len(gt), bool)
        for i in np.argsort(-pred[:, 4]):
            j = np.where(~used & (m[i] >= thr))[0]
            if len(j):
                j = j[m[i, j].argmax()]
                used[j] = tp[i] = True
    return tp


def ap50(tps, confs, n_gt):
    import numpy as np
    from ultralytics.utils.metrics import compute_ap
    if n_gt == 0:
        return float("nan")
    tp, conf = np.concatenate(tps), np.concatenate(confs)
    o = np.argsort(-conf)
    tpc = np.cumsum(tp[o])
    rec, prec = tpc / n_gt, tpc / np.arange(1, len(tpc) + 1)
    return compute_ap(rec.tolist(), prec.tolist())[0]


def gt_boxes(img_path):
    """Метки YOLO (COCO-номера) -> {группа: рамки x1y1x2y2 в пикселях}."""
    import numpy as np
    lab = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)[:-4] + ".txt"
    rows = [list(map(float, l.split())) for l in open(lab) if l.strip()]
    a = np.array(rows).reshape(-1, 5)
    xyxy = np.stack([(a[:, 1] - a[:, 3] / 2) * IMG_W, (a[:, 2] - a[:, 4] / 2) * IMG_H,
                     (a[:, 1] + a[:, 3] / 2) * IMG_W, (a[:, 2] + a[:, 4] / 2) * IMG_H], 1)
    return {"vehicle": xyxy[np.isin(a[:, 0], VEHICLES)], "person": xyxy[a[:, 0] == 0]}


def evaluate(weights_list, device, batch):
    """AP50 «машина» (car/motorcycle/bus/truck как один класс — так их видит detector.py) и «человек»,
    отдельно для дневных и ночных кадров отложенной проверки (официальный val BDD)."""
    import numpy as np
    import torch
    import torchvision
    from ultralytics import YOLO as Y
    groups = {g: set(open(os.path.join(YOLO, f"val_{g}.txt")).read().split()) for g in ("day", "night")}
    images = open(os.path.join(YOLO, "val_all.txt")).read().split()
    rows = []
    for w in weights_list:
        model = Y(w)
        t = time.time()
        acc = {(g, k): ([], [], [0]) for g in ("all", "day", "night") for k in ("vehicle", "person")}
        for i in range(0, len(images), batch):
            chunk = images[i:i + batch]
            for path, r in zip(chunk, model.predict(chunk, imgsz=640, conf=0.001, iou=0.7, device=device,
                                                    classes=VEHICLES + [0], max_det=300, verbose=False)):
                b = r.boxes
                xyxy, conf, cls = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()
                gt = gt_boxes(path)
                for k in ("vehicle", "person"):
                    sel = np.isin(cls, VEHICLES) if k == "vehicle" else cls == 0
                    p = np.concatenate([xyxy[sel], conf[sel, None]], 1)
                    if k == "vehicle" and len(p):     # NMS для всех машин вместе, как в detector.py
                        keep = torchvision.ops.nms(torch.from_numpy(p[:, :4]), torch.from_numpy(p[:, 4]), 0.7)
                        p = p[keep.numpy()]
                    tp = match(p, gt[k])
                    for g in ("all", "day", "night"):
                        if g == "all" or path in groups[g]:
                            acc[g, k][0].append(tp)
                            acc[g, k][1].append(p[:, 4])
                            acc[g, k][2][0] += len(gt[k])
        name = os.path.relpath(w, ROOT)
        print(f"{name}: {time.time() - t:.0f} с на {len(images)} кадров")
        for g in ("day", "night", "all"):
            rows.append((name, g, len(images) if g == "all" else len(groups[g]),
                         ap50(*acc[g, "vehicle"][:2], acc[g, "vehicle"][2][0]),
                         ap50(*acc[g, "person"][:2], acc[g, "person"][2][0])))
    print(f"\n{'модель':45s} {'набор':6s} {'кадров':>7s} {'AP50 машины':>12s} {'AP50 люди':>10s}")
    for r in rows:
        print(f"{r[0]:45s} {r[1]:6s} {r[2]:7d} {r[3]:12.3f} {r[4]:10.3f}")
    return rows


# ---------------------------------------------------------------- экспорт
def export(weights, out, opset):
    """ONNX как у Ultralytics по умолчанию: 1 x 3 x 640 x 640 -> 1 x 84 x 8400, fp32, без NMS."""
    from ultralytics import YOLO as Y
    p = Y(weights).export(format="onnx", imgsz=640, opset=opset, dynamic=False, half=False,
                          simplify=True, nms=False, batch=1)
    shutil.copy(p, out)
    print("сохранено:", out)


# ---------------------------------------------------------------- проверка на ролике
def video(onnx_paths, path, step, conf):
    """Сколько кадров (каждый step-й) с хотя бы одной машиной conf >= conf и шириной < 0.9 кадра."""
    import cv2
    import numpy as np
    import onnxruntime as ort
    for mp in onnx_paths:
        sess = ort.InferenceSession(mp, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        dtype = np.float16 if "float16" in inp.type else np.float32
        cap = cv2.VideoCapture(path)
        i = n = hit = 0
        ms = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if (i - 1) % step:
                continue
            h, w = frame.shape[:2]
            r = 640 / max(h, w)
            nw, nh = round(w * r), round(h * r)
            dx, dy = (640 - nw) // 2, (640 - nh) // 2
            img = np.full((640, 640, 3), 114, np.uint8)
            img[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh))
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True).astype(dtype)
            t = time.perf_counter()
            out = sess.run(None, {inp.name: blob})[0][0].T.astype(np.float32)
            ms.append((time.perf_counter() - t) * 1000)
            s = out[:, 4:][:, VEHICLES].max(1)
            bw = out[:, 2] / r
            n += 1
            hit += bool(((s >= conf) & (bw < 0.9 * w)).any())
        print(f"{os.path.basename(mp)}: {hit}/{n} кадров с машиной, {np.median(ms[3:]):.1f} мс/кадр")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["download", "prep", "train", "eval", "export", "video", "all"])
    ap.add_argument("--train-n", type=int, default=10000, help="кадров в train (0 = весь официальный train, 69 863)")
    ap.add_argument("--night-share", type=float, default=0.4, help="минимальная доля ночи в train")
    ap.add_argument("--val-n", type=int, default=0, help="кадров в отложенной проверке (0 = все 10 000)")
    ap.add_argument("--train-val-n", type=int, default=500, help="кадров для проверки во время обучения")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hours", type=float, default=0,
                    help="ограничить обучение по времени (ч): Ultralytics сам пересчитает число эпох и расписание lr")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps", help="mps (Mac), 0 (CUDA), cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="продолжить прерванное обучение")
    ap.add_argument("--weights", nargs="*", help="веса для eval/export (по умолчанию best.pt)")
    ap.add_argument("--onnx", nargs="*", default=[os.path.join(ROOT, "models", "yolo11s_fp32.onnx"),
                                                   os.path.join(ROOT, "models", "yolo11s_bdd.onnx")])
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "yolo11s_bdd.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--video-path", default=os.path.join(ROOT, "data", "videos", "ca_wrong_way_pd.webm"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    best = os.path.join(RUNS, RUN_NAME, "weights", "best.pt")

    if a.cmd in ("download", "all"):
        download()
    if a.cmd in ("prep", "all"):
        prep(a.train_n, a.night_share, a.val_n, a.train_val_n, a.seed)
    if a.cmd in ("train", "all"):
        best = train(a.epochs, a.batch, a.device, a.workers, a.imgsz, WEIGHTS, a.resume, a.hours)
    if a.cmd in ("eval", "all"):
        evaluate(a.weights or [WEIGHTS, best], a.device, a.batch)
    if a.cmd in ("export", "all"):
        export((a.weights or [best])[0], a.out, a.opset)
    if a.cmd in ("video", "all"):
        video(a.onnx, a.video_path, 3, 0.3)


if __name__ == "__main__":
    main()
