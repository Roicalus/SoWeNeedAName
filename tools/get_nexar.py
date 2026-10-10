"""Скачать ролики аварий и почти-аварий из Nexar Dashcam Collision Prediction Dataset + разметку.

Датасет: https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction
1500 роликов по ~40 с, 1280x720, 30 к/с, в основном США (много плотного городского трафика, Нью-Йорк).
У «positive» размечены момент аварии (time_of_event) и момент, когда опасность уже видна (time_of_alert).

Доступ закрыт соглашением (Nexar Dataset License 2025: только с указанием авторства, без продажи):
  1. войти на huggingface.co и нажать «Agree and access» на странице датасета;
  2. создать токен (Settings → Access Tokens, тип Read);
  3. запустить:
        HF_TOKEN=hf_... python tools/get_nexar.py --n 30
        HF_TOKEN=hf_... python tools/get_nexar.py --n 30 --scene urban --negative 10

Результат: data/nexar/*.mp4 и data/nexar/labels.csv (формат evaluate.py: file,start,end,type).
Опасный интервал — от time_of_alert до time_of_event. «negative» (обычная езда) попадают в папку
без строк в labels.csv — evaluate.py считает их спокойными, любое предупреждение = ложное.
Проверка:  python evaluate.py data/nexar --labels data/nexar/labels.csv
           python debug_video.py data/nexar
"""
import argparse
import csv
import io
import os
import random
import sys
import urllib.request

REPO = "https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/resolve/main"


def get(path, token):
    req = urllib.request.Request(f"{REPO}/{path}", headers={"Authorization": f"Bearer {token}"})
    try:
        return urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            sys.exit("Нет доступа: примите условия на странице датасета и проверьте HF_TOKEN.")
        raise


def metadata(split, token):
    rows = list(csv.DictReader(io.TextIOWrapper(get(f"train/{split}/metadata.csv", token), "utf-8")))
    for r in rows:
        r["_split"] = split
    return rows


def download(row, out, token):
    name = os.path.basename(row["file_name"])
    dst = os.path.join(out, f"nexar_{row['_split'][:3]}_{name}")
    if not os.path.exists(dst):
        with get(f"train/{row['_split']}/{name}", token) as r, open(dst + ".part", "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(dst + ".part", dst)
    return dst


def main():
    ap = argparse.ArgumentParser(description="Скачать ролики Nexar с авариями")
    ap.add_argument("--n", type=int, default=30, help="сколько роликов с авариями/почти-авариями")
    ap.add_argument("--negative", type=int, default=0, help="сколько роликов обычной езды (для ложных)")
    ap.add_argument("--scene", help="фильтр по сцене, например urban")
    ap.add_argument("--light", help="фильтр по освещению, например Normal или Night")
    ap.add_argument("--out", default="data/nexar")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Нужен HF_TOKEN (см. начало файла).")
    os.makedirs(args.out, exist_ok=True)

    def pick(rows, n):
        if args.scene:
            rows = [r for r in rows if args.scene.lower() in r.get("scene", "").lower()]
        if args.light:
            rows = [r for r in rows if args.light.lower() in r.get("light_conditions", "").lower()]
        random.Random(args.seed).shuffle(rows)
        return rows[:n]

    rows = pick(metadata("positive", token), args.n)
    if args.negative:
        rows += pick(metadata("negative", token), args.negative)
    print("Колонки разметки:", ", ".join(k for k in rows[0] if not k.startswith("_")) if rows else "-")

    labels = []
    for k, r in enumerate(rows, 1):
        dst = download(r, args.out, token)
        info = " ".join(r.get(c, "") for c in ("scene", "light_conditions", "weather"))
        print(f"[{k}/{len(rows)}] {os.path.basename(dst)}  {info}")
        if r["_split"] == "positive" and r.get("time_of_event"):
            event = float(r["time_of_event"])
            alert = float(r["time_of_alert"]) if r.get("time_of_alert") else event - 1.0
            labels.append({"file": os.path.basename(dst), "start": round(alert, 2), "end": round(event, 2),
                           "type": "any"})

    path = os.path.join(args.out, "labels.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "start", "end", "type"])
        w.writeheader()
        w.writerows(labels)
    with open(os.path.join(args.out, "SOURCE.txt"), "w", encoding="utf-8") as f:
        f.write("Nexar Dashcam Collision Prediction Dataset — (c) 2025 Nexar Inc., Nexar Dataset License 2025.\n"
                "D. C. Moura, S. Zvitia, Nexar Collision Dataset, Hugging Face, 2025.\n")
    print(f"\n{len(rows)} роликов в {args.out}, разметка: {path}")


if __name__ == "__main__":
    main()
