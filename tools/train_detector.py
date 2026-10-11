"""Fine-tune YOLO11 on BDD100K (night, dense traffic, US dashcams) while keeping COCO class ids.

Why: the COCO model barely sees cars at night (headlights, glare, dark silhouettes) and misses many
small far objects in dashcam footage. BDD100K is 80k dashcam frames, ~40% of them at night. We train on
it but keep the 4 + 80 output channels and COCO class numbers, so the new .onnx is a drop-in
replacement for detector.py (nothing else changes).

Data (the official BDD100K site is down; open Hugging Face copies, no login):
    frames:      https://huggingface.co/datasets/Xing1210/bdd100k  (one 5.7 GB zip, all 100k frames)
    boxes:       https://huggingface.co/datasets/dronefreak/BDD100K
                 (metadata.jsonl shards: boxes x,y,w,h in pixels, 10 BDD classes;
                  train 59 384 / valid 10 479 / test 10 000 = official BDD val)
    time of day: https://huggingface.co/datasets/dronefreak/BDD100K-Period-Classification
Class mapping BDD -> COCO: person, rider -> person; car; truck; bus; train -> train; motor -> motorcycle;
bike -> bicycle; traffic light -> traffic light. Traffic signs are dropped (COCO has only stop sign).

Training needs PyTorch (only here - the app itself runs on ONNX Runtime):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install ultralytics onnx onnxslim

Everything (download -> prepare -> train -> evaluate -> export models/<arch>_bdd.onnx):
    python tools/train_detector.py all --train-n 0 --epochs 30 --device 0
Step by step:
    python tools/train_detector.py download
    python tools/train_detector.py prep --train-n 0 --night-share 0.4        # 0 = the whole train split
    python tools/train_detector.py train --arch yolo11s --epochs 30 --batch 32 --device 0
    python tools/train_detector.py eval --weights data/bdd_det/weights/yolo11s.pt runs/bdd/yolo11s_bdd/weights/best.pt
    python tools/train_detector.py export --arch yolo11s
    python tools/train_detector.py video --onnx models/yolo11s_fp32.onnx models/yolo11s_bdd.onnx

The held-out check is the official BDD val (10 000 frames, never trained on), day and night separately.
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
# prepared set: images/<split>/*.jpg + labels/<split>/*.txt. Keep it on an SSD (BDD_YOLO_DIR or --yolo):
# 70k small files read randomly from a hard disk starve the GPU (~30 images/s instead of hundreds)
YOLO = os.environ.get("BDD_YOLO_DIR", os.path.join(DATA, "yolo"))
META = os.path.join(DATA, "meta")          # metadata.jsonl: boxes (det_*) and time of day (period_*)
HF = "https://huggingface.co/datasets/"
ZIP_URL = HF + "Xing1210/bdd100k/resolve/main/bdd100k_images_100k.zip"
DET_URL = HF + "dronefreak/BDD100K/resolve/main/data/images/{split}/shard_{s:03d}/metadata.jsonl"
PERIOD_URL = HF + "dronefreak/BDD100K-Period-Classification/resolve/main/data/images/{split}/shard_{s:03d}/metadata.jsonl"
PT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{arch}.pt"
DET_SHARDS = {"train": 8, "valid": 2, "test": 2}
PERIOD_SHARDS = {"train": 9, "valid": 2}
IMG_W, IMG_H = 1280, 720                   # all BDD100K frames are 1280x720
RUNS = os.path.join(ROOT, "runs", "bdd")

# BDD class (as in dronefreak's data.yaml) -> COCO class; the rest (traffic sign) is dropped
BDD_TO_COCO = {
    0: 0,   # person -> person
    1: 0,   # rider (cyclist / motorcyclist) -> person
    2: 2,   # car -> car
    3: 7,   # truck -> truck
    4: 5,   # bus -> bus
    5: 6,   # train (incl. tram) -> train
    6: 3,   # motor -> motorcycle
    7: 1,   # bike -> bicycle
    8: 9,   # traffic light -> traffic light (junction context)
}
VEHICLES = [2, 3, 5, 7]          # same as C.VEHICLE_CLASSES in config.py


def weights(arch):
    return os.path.join(DATA, "weights", f"{arch}.pt")


def coco_names(arch="yolo11s"):
    """80 COCO names in id order (taken from the model itself)."""
    from ultralytics import YOLO as Y
    fetch(PT_URL.format(arch=arch), weights(arch))
    return Y(weights(arch)).names


def fetch(url, dst):
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print("downloading", url)
        urllib.request.urlretrieve(url, dst + ".part")
        os.replace(dst + ".part", dst)


# ---------------------------------------------------------------- download
def download(arch):
    """Frames zip (5.7 GB), boxes and time of day (several jsonl), COCO weights."""
    t = time.time()
    fetch(PT_URL.format(arch=arch), weights(arch))
    for split, n in DET_SHARDS.items():
        for s in range(n):
            fetch(DET_URL.format(split=split, s=s), os.path.join(META, f"det_{split}_{s:03d}.jsonl"))
    for split, n in PERIOD_SHARDS.items():
        for s in range(n):
            fetch(PERIOD_URL.format(split=split, s=s), os.path.join(META, f"period_{split}_{s:03d}.jsonl"))
    fetch(ZIP_URL, ZIP)
    print(f"downloaded in {time.time() - t:.0f} s")


def timeofday():
    """file name -> 'daytime' / 'night' / 'dawn or dusk' / 'unknown'."""
    tod = {}
    for f in glob.glob(os.path.join(META, "period_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            tod[d["file_name"]] = d["label"]
    return tod


def boxes(split):
    """file name -> [(BDD class, x, y, w, h in pixels)] for train / valid / test (dronefreak)."""
    res = {}
    for f in sorted(glob.glob(os.path.join(META, f"det_{split}_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            res[d["file_name"]] = list(zip(d["objects"]["categories"], d["objects"]["bbox"]))
    return res


# ---------------------------------------------------------------- preparation
def yolo_lines(objs):
    """BDD boxes -> YOLO lines with COCO ids (centre, width, height as frame fractions)."""
    out = []
    for c, (x, y, w, h) in objs:
        if c in BDD_TO_COCO and w > 1 and h > 1:
            out.append(f"{BDD_TO_COCO[c]} {(x + w / 2) / IMG_W:.6f} {(y + h / 2) / IMG_H:.6f} "
                       f"{w / IMG_W:.6f} {h / IMG_H:.6f}")
    return out


def add_split(zf, members, names, split, labels, tod):
    """Extracts frames to yolo/images/<split>/ and writes labels to yolo/labels/<split>/."""
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
        with open(os.path.join(dl, name[:-4] + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        res.append((img, tod.get(name, "unknown")))
    return res


def write_list(path, items):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(items) + "\n")
    return path


def write_yaml(path, train, val, names):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"path: {YOLO}\ntrain: {train}\nval: {val}\nnc: {len(names)}\nnames:\n")
        for i in range(len(names)):
            f.write(f"  {i}: {names[i]}\n")
    return path


def prep(train_n, night_share, val_n, train_val_n, seed):
    """Builds the frame lists. Night is topped up by repeating night frames in train.txt (oversampling)."""
    rnd = random.Random(seed)
    tod = timeofday()
    for f in glob.glob(os.path.join(YOLO, "*.cache")) + glob.glob(os.path.join(YOLO, "labels", "*.cache")):
        os.remove(f)
    if os.path.exists(os.path.join(YOLO, "labels")):
        shutil.rmtree(os.path.join(YOLO, "labels"))
    # training pool = official BDD train (dronefreak's train + valid), check = official val
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
    print(f"extracted in {time.time() - t:.0f} s")

    night = [p for p, t in train if t == "night"]
    lst = [p for p, _ in train]
    share = len(night) / max(1, len(lst))
    if night and share < night_share < 1:  # add e night frames: (n + e) / (T + e) = night_share
        extra = round((night_share * len(lst) - len(night)) / (1 - night_share))
        lst += [rnd.choice(night) for _ in range(extra)]
    rnd.shuffle(lst)
    n_night = sum(tod.get(os.path.basename(p)) == "night" for p in lst)
    print(f"train: {len(train)} frames (night {share:.0%}), list with oversampling {len(lst)} "
          f"(night {n_night / len(lst):.0%})")

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
    # a small check during training (the full evaluation is the eval command)
    write_list(os.path.join(YOLO, "val_mini.txt"), groups["all"][:train_val_n])
    write_yaml(os.path.join(YOLO, "train.yaml"), "train.txt", "val_mini.txt", names)


# ---------------------------------------------------------------- training
def train(arch, epochs, batch, device, workers, imgsz, resume, hours, init=None, name=None):
    from ultralytics import YOLO as Y
    t = time.time()
    run = name or f"{arch}_bdd"
    fetch(PT_URL.format(arch=arch), weights(arch))
    model = Y(os.path.join(RUNS, run, "weights", "last.pt") if resume else (init or weights(arch)))
    extra = {}
    if init:   # continue from already fine-tuned weights: a short run with a lower start lr and no long warm-up
        extra = {"optimizer": "SGD", "lr0": 0.004, "warmup_epochs": 0.3}
    model.train(data=os.path.join(YOLO, "train.yaml"), epochs=epochs, imgsz=imgsz, batch=batch,
                device=device, workers=workers, project=RUNS, name=run, exist_ok=True,
                resume=resume, cos_lr=True, close_mosaic=max(1, epochs // 6), plots=False,
                patience=100, amp=True, time=hours or None,
                **({"warmup_epochs": min(3.0, epochs * 0.1)} | extra))   # default augmentations
    print(f"training: {(time.time() - t) / 60:.1f} min")
    return os.path.join(RUNS, run, "weights", "best.pt")


# ---------------------------------------------------------------- evaluation
def iou(a, b):
    """IoU matrix for x1y1x2y2 boxes: a (N x 4), b (M x 4)."""
    import numpy as np
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), 2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def match(pred, gt, thr=0.5):
    """Greedy matching by decreasing confidence (as in COCO). pred: K x 5 (x1y1x2y2, conf)."""
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
    """YOLO labels (COCO ids) -> {group: x1y1x2y2 boxes in pixels}."""
    import numpy as np
    lab = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)[:-4] + ".txt"
    rows = [list(map(float, l.split())) for l in open(lab, encoding="utf-8") if l.strip()]
    a = np.array(rows).reshape(-1, 5)
    xyxy = np.stack([(a[:, 1] - a[:, 3] / 2) * IMG_W, (a[:, 2] - a[:, 4] / 2) * IMG_H,
                     (a[:, 1] + a[:, 3] / 2) * IMG_W, (a[:, 2] + a[:, 4] / 2) * IMG_H], 1)
    return {"vehicle": xyxy[np.isin(a[:, 0], VEHICLES)], "person": xyxy[a[:, 0] == 0]}


def evaluate(weights_list, device, batch):
    """AP50 "vehicle" (car/motorcycle/bus/truck as one class - the way detector.py sees them) and
    "person", separately for day and night frames of the held-out set (official BDD val)."""
    import numpy as np
    import torch
    import torchvision
    from ultralytics import YOLO as Y
    groups = {g: set(open(os.path.join(YOLO, f"val_{g}.txt"), encoding="utf-8").read().split()) for g in ("day", "night")}
    images = open(os.path.join(YOLO, "val_all.txt"), encoding="utf-8").read().split()
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
                    if k == "vehicle" and len(p):     # NMS over all vehicles together, as in detector.py
                        keep = torchvision.ops.nms(torch.from_numpy(p[:, :4]), torch.from_numpy(p[:, 4]), 0.7)
                        p = p[keep.numpy()]
                    tp = match(p, gt[k])
                    for g in ("all", "day", "night"):
                        if g == "all" or path in groups[g]:
                            acc[g, k][0].append(tp)
                            acc[g, k][1].append(p[:, 4])
                            acc[g, k][2][0] += len(gt[k])
        name = os.path.relpath(w, ROOT)
        print(f"{name}: {time.time() - t:.0f} s for {len(images)} frames")
        for g in ("day", "night", "all"):
            rows.append((name, g, len(images) if g == "all" else len(groups[g]),
                         ap50(*acc[g, "vehicle"][:2], acc[g, "vehicle"][2][0]),
                         ap50(*acc[g, "person"][:2], acc[g, "person"][2][0])))
    print(f"\n{'model':45s} {'set':6s} {'frames':>7s} {'AP50 vehicle':>13s} {'AP50 person':>12s}")
    for r in rows:
        print(f"{r[0]:45s} {r[1]:6s} {r[2]:7d} {r[3]:13.3f} {r[4]:12.3f}")
    out = []
    for w in weights_list:
        name = os.path.relpath(w, ROOT)
        get = {(r[1], k): r[3 if k == "v" else 4] for r in rows if r[0] == name for k in ("v", "p")}
        out.append({"model": "fine-tuned on BDD100K (ours)" if os.sep + "runs" + os.sep in os.sep + name else "COCO weights (before)",
                    "day_vehicle": get[("day", "v")], "night_vehicle": get[("night", "v")],
                    "day_person": get[("day", "p")], "night_person": get[("night", "p")]})
    with open(os.path.join(ROOT, "logs", "detector_ap.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return rows


# ---------------------------------------------------------------- export
def export(weights_path, out, opset):
    """ONNX as Ultralytics exports by default: 1 x 3 x 640 x 640 -> 1 x 84 x 8400, fp32, no NMS."""
    from ultralytics import YOLO as Y
    p = Y(weights_path).export(format="onnx", imgsz=640, opset=opset, dynamic=False, half=False,
                               simplify=True, nms=False, batch=1)
    shutil.copy(p, out)
    print("saved:", out)


# ---------------------------------------------------------------- check on a video
def video(onnx_paths, path, step, conf):
    """How many frames (every step-th) have at least one car with conf >= conf and width < 0.9 frame."""
    import cv2
    import numpy as np
    import onnxruntime as ort
    for mp in onnx_paths:
        sess = ort.InferenceSession(mp, providers=ort.get_available_providers())
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
        print(f"{os.path.basename(mp)}: {hit}/{n} frames with a car, {np.median(ms[3:]):.1f} ms/frame")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["download", "prep", "train", "eval", "export", "video", "all"])
    ap.add_argument("--arch", default="yolo11s", help="yolo11n / yolo11s / yolo11m ...")
    ap.add_argument("--train-n", type=int, default=10000, help="frames in train (0 = all official train, 69 863)")
    ap.add_argument("--night-share", type=float, default=0.4, help="minimum share of night in train")
    ap.add_argument("--val-n", type=int, default=0, help="frames in the held-out check (0 = all 10 000)")
    ap.add_argument("--train-val-n", type=int, default=500, help="frames for the check during training")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hours", type=float, default=0,
                    help="time limit for training (h): Ultralytics recomputes epochs and the lr schedule")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="0 (CUDA), mps (Mac), cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="continue an interrupted training")
    ap.add_argument("--weights", nargs="*", help="weights for eval/export (default best.pt)")
    ap.add_argument("--onnx", nargs="*")
    ap.add_argument("--out", help="exported ONNX (default models/<arch>_bdd.onnx)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--video-path", default=os.path.join(ROOT, "data", "videos", "krakow_peds.mp4"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="start training from these weights (e.g. last.pt of an interrupted run)")
    ap.add_argument("--name", help="run name (default <arch>_bdd)")
    ap.add_argument("--yolo", help="prepared dataset folder (default BDD_YOLO_DIR or data/bdd_det/yolo)")
    a = ap.parse_args()
    if a.yolo:
        global YOLO
        YOLO = a.yolo
    best = os.path.join(RUNS, f"{a.arch}_bdd", "weights", "best.pt")
    out = a.out or os.path.join(ROOT, "models", f"{a.arch}_bdd.onnx")

    if a.cmd in ("download", "all"):
        download(a.arch)
    if a.cmd in ("prep", "all"):
        prep(a.train_n, a.night_share, a.val_n, a.train_val_n, a.seed)
    if a.cmd in ("train", "all"):
        best = train(a.arch, a.epochs, a.batch, a.device, a.workers, a.imgsz, a.resume, a.hours, a.init, a.name)
    if a.cmd in ("eval", "all"):
        evaluate(a.weights or [weights(a.arch), best], a.device, a.batch)
    if a.cmd in ("export", "all"):
        export((a.weights or [best])[0], out, a.opset)
    if a.cmd in ("video", "all"):
        video(a.onnx or [os.path.join(ROOT, "models", "yolo11s_fp32.onnx"), out], a.video_path, 3, 0.3)


if __name__ == "__main__":
    main()
