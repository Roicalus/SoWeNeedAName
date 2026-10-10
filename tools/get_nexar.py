"""Download crash and near-miss videos from the Nexar Dashcam Collision Prediction Dataset + labels.

Dataset: https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction
1500 videos of ~40 s, 1280x720, 30 fps, mostly the US (much dense city traffic, New York).
"positive" videos have the crash moment (time_of_event) and the moment the danger is visible
(time_of_alert) labelled.

Access is gated by an agreement (Nexar Dataset License 2025: attribution, no resale):
  1. sign in on huggingface.co and press "Agree and access" on the dataset page;
  2. create a token (Settings -> Access Tokens, type Read);
  3. run:
        HF_TOKEN=hf_... python tools/get_nexar.py --n 30
        HF_TOKEN=hf_... python tools/get_nexar.py --n 30 --scene urban --negative 10

Output: data/nexar/*.mp4 and data/nexar/labels.csv (file,start,end,type).
The danger interval is time_of_alert .. time_of_event. "negative" videos (normal driving) are saved
without label rows - evaluate.py treats them as calm, any yellow/red alert = false.
Check:  python evaluate.py data/nexar --labels data/nexar/labels.csv
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
            sys.exit("No access: accept the terms on the dataset page and check HF_TOKEN.")
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
    ap = argparse.ArgumentParser(description="Download Nexar crash videos")
    ap.add_argument("--n", type=int, default=30, help="crash / near-miss videos")
    ap.add_argument("--negative", type=int, default=0, help="normal driving videos (for false alarms)")
    ap.add_argument("--scene", help="scene filter, e.g. urban")
    ap.add_argument("--light", help="light filter, e.g. Normal or Night")
    ap.add_argument("--out", default="data/nexar")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is required (see the top of this file).")
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
    print("Label columns:", ", ".join(k for k in rows[0] if not k.startswith("_")) if rows else "-")

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
        f.write("Nexar Dashcam Collision Prediction Dataset - (c) 2025 Nexar Inc., Nexar Dataset License 2025.\n"
                "D. C. Moura, S. Zvitia, Nexar Collision Dataset, Hugging Face, 2025.\n")
    print(f"\n{len(rows)} videos in {args.out}, labels: {path}")


if __name__ == "__main__":
    main()
