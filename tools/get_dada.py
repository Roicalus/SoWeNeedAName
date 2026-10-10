"""Скачать выборку роликов DADA-2000 (реальные аварии с регистраторов) и разметку для evaluate.py.

DADA-2000: https://github.com/JWFangit/LOTVS-DADA — 1962 ролика аварий, в основном плотный городской
трафик Китая, у каждого размечены кадры «начало опасной ситуации» и «авария».
Оригинал на Google Drive часто упирается в квоту скачиваний, поэтому берём открытую копию по кадрам:
    кадры:    https://huggingface.co/datasets/yumuxie/dada   (<тип>/<номер>/images/0001.jpg …)
    разметка: https://huggingface.co/datasets/DBatinic/DADA2000 (CSV)

    python tools/get_dada.py --per-group 40
    python tools/get_dada.py --per-group 40 --night-share 0.3

Чтобы не качать ~400 кадров на ролик по одному запросу, берётся каждый --step-й кадр
(по умолчанию 3 → 10 к/с, как частота анализа) и только окно от --pre секунд до начала опасности
до --post секунд после аварии. Ролик собирается с частотой 30/step к/с, время в разметке — от его начала.

Ролики группируются по тому, что должно сработать у нас (GROUPS ниже). Из каждой группы берётся
--per-group роликов, половина помечается split=test — на них только проверяем, пороги не подбираем.
Результат: data/dada/videos/*.mp4, data/dada/labels.csv
  (file,start,end,type,group,light,split,text — start = начало опасности, end = момент аварии, с).
"""
import argparse
import csv
import glob
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2

ANN_URL = "https://huggingface.co/datasets/DBatinic/DADA2000/resolve/main/DADA2000_video_annotations.csv"
FRAMES_REPO = "yumuxie/dada"
FPS = 30   # DADA-2000 записан с частотой 30 к/с

# группа -> (тип для evaluate.py, ключевые слова в описании ролика)
GROUPS = {
    "lead_stop": ("brake", ["lead vehicle stops", "lead vehicle decelerates", "lead motorcycle stops"]),
    "cut_in":    ("cutin", ["changes lanes", "changes lines", "overtakes"]),
    "crossing":  ("crossing", ["vehicle crosses the road", "turns left across path", "turns right across path",
                               "goes straight crossing paths", "turns around"]),
    "vru":       ("ped", ["pedestrian crosses", "cyclist crosses", "motorbike crosses"]),
    "loss":      ("crash", ["out of control", "controls loss", "object crash"]),
}


def group_of(text):
    text = text.lower()
    if text.startswith("ego-car"):          # виноват сам водитель (наш манёвр) — не наш сценарий
        return None
    for g, (_, keys) in GROUPS.items():
        if any(k in text for k in keys):
            return g
    return None


def main():
    ap = argparse.ArgumentParser(description="Выборка роликов DADA-2000")
    ap.add_argument("--per-group", type=int, default=40)
    ap.add_argument("--night-share", type=float, default=0.25, help="доля ночных роликов в каждой группе")
    ap.add_argument("--out", default="data/dada")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=3, help="брать каждый N-й кадр")
    ap.add_argument("--pre", type=float, default=6.0, help="секунд до начала опасности")
    ap.add_argument("--post", type=float, default=1.0, help="секунд после аварии")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "videos"), exist_ok=True)
    ann = os.path.join(args.out, "annotations.csv")
    if not os.path.exists(ann):
        urllib.request.urlretrieve(ANN_URL, ann)
    with open(ann, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    rnd = random.Random(args.seed)
    picked = []
    for g in GROUPS:
        cand = [r for r in rows if group_of(r["texts"].replace("[CLS]", "").replace("[SEP]", "").strip()) == g
                and r["whether an accident occurred (1/0)"] == "1"]
        night = [r for r in cand if r["light(day,night)1-2"] == "2"]
        day = [r for r in cand if r["light(day,night)1-2"] != "2"]
        rnd.shuffle(night)
        rnd.shuffle(day)
        n_night = min(len(night), round(args.per_group * args.night_share))
        sel = night[:n_night] + day[:args.per_group - n_night]
        rnd.shuffle(sel)
        for k, r in enumerate(sel):
            picked.append((g, "test" if k % 2 else "tune", r))
        print(f"{g:10s} кандидатов {len(cand):4d} (ночь {len(night)}), взято {len(sel)}")

    cache = os.path.join(args.out, "frames")
    def window(r):
        first = max(1, int(r["abnormal start frame"]) - int(args.pre * FPS))
        last = min(int(r["total frames"]), int(r["accident frame"]) + int(args.post * FPS))
        return first, last

    files = []
    for _, _, r in picked:
        first, last = window(r)
        files += [f"{int(r['type'])}/{r['video']}/images/{k:04d}.jpg" for k in range(first, last + 1, args.step)]
    todo = [f for f in files if not os.path.exists(os.path.join(cache, f))]
    print(f"Кадров {len(files)}, скачать {len(todo)} ({len(picked)} роликов; лимит HF ~10 файлов/с)…", flush=True)

    def fetch(rel):
        dst = os.path.join(cache, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        url = f"https://huggingface.co/datasets/{FRAMES_REPO}/resolve/main/{rel}"
        for _ in range(20):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = r.read()
                with open(dst + ".part", "wb") as f:
                    f.write(data)
                os.replace(dst + ".part", dst)
                return True
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return False
                # 429: лимит Hugging Face для анонимных запросов — 3000 файлов за 5 минут; ждём окна
                wait = 30
                m = re.search(r"t=(\d+)", e.headers.get("ratelimit", "") or "")
                if m:
                    wait = int(m.group(1)) + 1
                time.sleep(wait)
            except Exception:
                time.sleep(2)
        return False

    with ThreadPoolExecutor(8) as pool:
        for k, ok in enumerate(pool.map(fetch, todo), 1):
            if k % 1000 == 0:
                print(f"  {k}/{len(todo)}", flush=True)

    labels = []
    for g, split, r in picked:
        src = os.path.join(cache, str(int(r["type"])), r["video"], "images")
        first, last = window(r)
        frames = [os.path.join(src, f"{k:04d}.jpg") for k in range(first, last + 1, args.step)]
        frames = [p for p in frames if os.path.exists(p)]
        if not frames:
            print("нет кадров:", src)
            continue
        name = f"dada_{int(r['type']):02d}_{r['video']}.mp4"
        dst = os.path.join(args.out, "videos", name)
        if not os.path.exists(dst):
            first = cv2.imread(frames[0])
            h, w = first.shape[:2]
            vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"avc1"), FPS / args.step, (w, h))
            for p in frames:
                vw.write(cv2.imread(p))
            vw.release()
        text = r["texts"].replace("[CLS]", "").replace("[SEP]", "").strip()
        # время от первого кадра нашего ролика (кадры в разметке нумеруются с 1)
        labels.append({"file": name, "start": round((int(r["abnormal start frame"]) - first) / FPS, 2),
                       "end": round((int(r["accident frame"]) - first) / FPS, 2), "type": GROUPS[g][0],
                       "group": g, "light": "night" if r["light(day,night)1-2"] == "2" else "day",
                       "split": split, "text": text})

    with open(os.path.join(args.out, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(labels[0]))
        w.writeheader()
        w.writerows(labels)
    print(f"{len(labels)} роликов → {args.out}/videos, разметка {args.out}/labels.csv")


if __name__ == "__main__":
    main()
