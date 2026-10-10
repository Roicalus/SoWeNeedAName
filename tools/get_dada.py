"""Download DADA-2000 clips (real crashes filmed by dashcams) and labels for replay.py / train_risk.py.

DADA-2000: https://github.com/JWFangit/LOTVS-DADA - 1962 crash videos, mostly dense urban traffic in
China; every video has the frames "danger starts" and "crash" labelled.
The original on Google Drive often hits the download quota, so we use an open per-frame copy:
    frames: https://huggingface.co/datasets/yumuxie/dada   (<type>/<id>/images/0001.jpg ...)
    labels: https://huggingface.co/datasets/DBatinic/DADA2000 (CSV)

    python tools/get_dada.py --all                  # every crash caused by another road user (~1700)
    python tools/get_dada.py --per-group 40         # a sample per group

To avoid ~400 requests per video we take every --step-th frame (default 3 -> 10 fps, the analysis
rate) and only the window from --pre seconds before the danger to --post seconds after the crash.
The clip is written at 30/step fps; label times are from the clip start.
Clips are built as they arrive and labels.csv is appended after each one, so an interrupted download
is usable right away and the same command resumes it (existing frames and clips are skipped).
Requests reuse keep-alive connections (two TLS handshakes per frame made it ~5x slower);
anonymous Hugging Face limit is 3000 requests / 5 min - set HF_TOKEN for a higher one.

Videos are grouped by what should fire in our system (GROUPS below). split=test for every second
video of a group: only evaluated on, never used to tune thresholds or train.
Output: data/dada/videos/*.mp4, data/dada/labels.csv
  (file,start,end,type,group,light,weather,scene,split,text - start = danger begins, end = crash, s).
"""
import argparse
import csv
import os
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import requests

ANN_URL = "https://huggingface.co/datasets/DBatinic/DADA2000/resolve/main/DADA2000_video_annotations.csv"
FRAMES_REPO = "yumuxie/dada"
FPS = 30   # DADA-2000 is recorded at 30 fps
FIELDS = ["file", "start", "end", "type", "group", "light", "weather", "scene", "split", "text"]
WEATHER = {"1": "sunny", "2": "rainy", "3": "snowy", "4": "foggy"}
SCENE = {"1": "highway", "2": "tunnel", "3": "mountain", "4": "urban", "5": "rural"}

# group -> (threat type, keywords in the video description); order matters - first match wins
GROUPS = {
    "lead_stop": ("brake", ["lead vehicle stops", "lead vehicle decelerates", "lead motorcycle stops",
                            "backes up", "backs up"]),
    "cut_in":    ("cutin", ["changes lanes", "changes lines", "overtakes", "turns on a road with the same",
                            "following vehicle makes a maneuver", "dodges"]),
    "crossing":  ("crossing", ["vehicle crosses the road", "turns left across path", "turns right across path",
                               "crossing paths", "turns around", "runs red light", "turns on a road with the opposite"]),
    "vru":       ("ped", ["pedestrian", "cyclist", "motorbike crosses", "motorbike retrogrades"]),
    "oncoming":  ("wrongway", ["meet on the road", "retrograde"]),
    "loss":      ("crash", ["out of control", "controls loss", "object crash"]),
}


def clean(text):
    """Description without service tokens and with normal spaces (the CSV contains non-breaking \\xa0)."""
    text = text.replace("[CLS]", " ").replace("[SEP]", " ")
    return re.sub(r"\s+", " ", text).strip()


def group_of(text):
    text = clean(text).lower()
    if text.startswith("ego-car"):          # our own driver caused it - not our scenario
        return None
    for g, (_, keys) in GROUPS.items():
        if any(k in text for k in keys):
            return g
    return "other"


_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        token = os.environ.get("HF_TOKEN")
        if token:
            _local.s.headers["Authorization"] = f"Bearer {token}"
    return _local.s


def fetch(rel, cache):
    dst = os.path.join(cache, rel)
    if os.path.exists(dst):
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"https://huggingface.co/datasets/{FRAMES_REPO}/resolve/main/{rel}"
    for _ in range(30):
        try:
            r = session().get(url, timeout=60)
            if r.status_code == 404:
                return False
            if r.status_code == 429:      # Hugging Face rate limit - wait for the next window
                m = re.search(r"t=(\d+)", r.headers.get("ratelimit", ""))
                time.sleep(int(m.group(1)) + 1 if m else 30)
                continue
            r.raise_for_status()
            with open(dst + ".part", "wb") as f:
                f.write(r.content)
            os.replace(dst + ".part", dst)
            return True
        except Exception:
            _local.__dict__.pop("s", None)
            time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser(description="DADA-2000 clips")
    ap.add_argument("--all", action="store_true", help="every crash caused by another road user")
    ap.add_argument("--per-group", type=int, default=40)
    ap.add_argument("--night-share", type=float, default=0.25, help="share of night videos per group (without --all)")
    ap.add_argument("--out", default="data/dada")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=3, help="take every N-th frame")
    ap.add_argument("--pre", type=float, default=6.0, help="seconds before the danger starts")
    ap.add_argument("--post", type=float, default=1.0, help="seconds after the crash")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16, help="videos downloaded in parallel")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "videos"), exist_ok=True)
    ann = os.path.join(args.out, "annotations.csv")
    if not os.path.exists(ann):
        urllib.request.urlretrieve(ANN_URL, ann)
    with open(ann, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    rnd = random.Random(args.seed)
    by_group = {}
    for r in rows:
        g = group_of(r["texts"])
        if g is None or r["whether an accident occurred (1/0)"] != "1":
            continue
        if int(r["accident frame"]) <= int(r["abnormal start frame"]):
            continue
        by_group.setdefault(g, []).append(r)
    picked = {}
    for g, cand in sorted(by_group.items()):
        rnd.shuffle(cand)
        if not args.all:
            night = [r for r in cand if r["light(day,night)1-2"] == "2"]
            day = [r for r in cand if r["light(day,night)1-2"] != "2"]
            n_night = min(len(night), round(args.per_group * args.night_share))
            cand = night[:n_night] + day[:args.per_group - n_night]
            rnd.shuffle(cand)
        picked[g] = [(g, "test" if k % 2 else "tune", r) for k, r in enumerate(cand)]
        print(f"{g:10s} picked {len(cand):4d} (night {sum(r['light(day,night)1-2'] == '2' for r in cand)})")
    # interleave groups: an interrupted download is still balanced
    order = []
    while any(picked.values()):
        for g in list(picked):
            if picked[g]:
                order.append(picked[g].pop(0))

    cache = os.path.join(args.out, "frames")
    labels_path = os.path.join(args.out, "labels.csv")
    done = set()
    if os.path.exists(labels_path):
        with open(labels_path, encoding="utf-8") as f:
            done = {r["file"] for r in csv.DictReader(f)}
    else:
        with open(labels_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def plan(item):
        g, split, r = item
        first = max(1, int(r["abnormal start frame"]) - int(args.pre * FPS))
        last = min(int(r["total frames"]), int(r["accident frame"]) + int(args.post * FPS))
        rels = [f"{int(r['type'])}/{r['video']}/images/{i:04d}.jpg" for i in range(first, last + 1, args.step)]
        return f"dada_{int(r['type']):02d}_{r['video']}.mp4", first, rels

    def build(item, name, first, rels):
        g, split, r = item
        frames = [os.path.join(cache, p) for p in rels if os.path.exists(os.path.join(cache, p))]
        if len(frames) < len(rels) * 0.9:
            print("too few frames, skipped:", name, len(frames), "/", len(rels))
            return False
        dst = os.path.join(args.out, "videos", name)
        img = cv2.imread(frames[0])
        h, w = img.shape[:2]
        vw = cv2.VideoWriter(dst + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS / args.step, (w, h))
        for p in frames:
            im = cv2.imread(p)
            vw.write(im if im is not None and im.shape[:2] == (h, w) else img)
        vw.release()
        os.replace(dst + ".tmp.mp4", dst)
        row = {"file": name, "start": round((int(r["abnormal start frame"]) - first) / FPS, 2),
               "end": round((int(r["accident frame"]) - first) / FPS, 2), "type": GROUPS.get(g, ("any",))[0],
               "group": g, "light": "night" if r["light(day,night)1-2"] == "2" else "day",
               "weather": WEATHER.get(r["weather(sunny,rainy,snowy,foggy)1-4"], "?"),
               "scene": SCENE.get(r["scenes(highway,tunnel,mountain,urban,rural)1-5"], "?"),
               "split": split, "text": clean(r["texts"])}
        with open(labels_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        return True

    todo = [item for item in order if plan(item)[0] not in done]
    print(f"to download: {len(todo)} videos", flush=True)
    tic, n_new = time.time(), 0
    with ThreadPoolExecutor(args.threads) as pool:
        for b in range(0, len(todo), args.batch):
            batch = [(item, *plan(item)) for item in todo[b:b + args.batch]]
            list(pool.map(lambda rel: fetch(rel, cache), [rel for *_, rels in batch for rel in rels]))
            for item, name, first, rels in batch:
                n_new += build(item, name, first, rels)
            el = time.time() - tic
            left = len(todo) - b - len(batch)
            print(f"[{b + len(batch)}/{len(todo)}] +{n_new} videos in {el / 60:.0f} min "
                  f"(~{el / (b + len(batch)) * left / 60:.0f} min left)", flush=True)
    print(f"done: {args.out}/videos, labels {labels_path}")


if __name__ == "__main__":
    main()
