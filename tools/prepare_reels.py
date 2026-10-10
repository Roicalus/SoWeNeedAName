"""Turn phone-filmed social-media driving clips (data/reels/*.mp4) into labelled dashcam-like clips.

The reels are vertical (720x1280), filmed by hand from the driver's seat: the road is a horizontal
band, the bottom half is the dashboard and steering wheel, and some have map overlays and cuts.
For each reel we crop a 16:9 band around the road (so the horizon and hood land where a dashcam would
put them) and cut one clip per hand-labelled dangerous window - from 4 s before it to 1 s after -
in the same format as DADA (tools/get_dada.py): start = danger begins, end = closest moment.

Labels were set by looking at frame sheets: close passes and squeezing between cars at high speed,
a car merging right next to us, closing in fast on the car ahead. In these clips the risky driver is
mostly the camera car itself, so they test "closing / cut-in / too close" in dense traffic, at night
and in Tashkent - conditions DADA and BDD barely have. They are weak labels (+-0.5 s).

    python tools/prepare_reels.py
Output: data/reels/clips/*.mp4, data/reels/labels.csv
"""
import csv
import os

import cv2

SRC = "data/reels"
OUT = os.path.join(SRC, "clips")
PRE, POST = 4.0, 1.0
W_OUT = 1280

# reel id -> (crop top, crop bottom as fraction of height, split, [(danger start, closest moment)], usable until s)
REELS = {
    "Dat_jZDMD0r": (0.24, 0.56, "tune", [(0.8, 1.7), (9.0, 10.0), (20.1, 21.7)], 22.3),   # Tashkent, night, weaving
    "Dawv0tICpWS": (0.32, 0.64, "test", [(1.8, 3.4), (8.2, 9.4)], 13.0),                 # fog, night, close merges
    "Dcoffe6P_Lc": (0.33, 0.65, "tune", [(0.5, 2.0), (4.5, 6.0), (8.5, 10.0)], 10.3),     # US highway, very fast
    "DdB7zmiKZu1": (0.25, 0.57, "tune", [(3.7, 5.6), (18.5, 24.1), (37.0, 38.9), (48.1, 51.8), (68.5, 70.3)], 74.0),
    "DdjrhrVP3Bt": (0.35, 0.67, "test", [(1.0, 3.0), (8.0, 10.0)], 10.3),                 # US highway, map at 5.5-7
    "DeGzFlmO0W0": (0.30, 0.62, "test", [(2.5, 3.5), (6.0, 7.5), (10.0, 10.8)], 11.4),    # Tashkent, day
    "DeISBY4Am4D": (0.23, 0.55, "tune", [(4.6, 5.8), (18.6, 19.7)], 23.2),                # night
}
FIELDS = ["file", "start", "end", "type", "group", "light", "weather", "scene", "split", "text"]
NIGHT = {"Dat_jZDMD0r", "Dawv0tICpWS", "DeISBY4Am4D"}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for rid, (top, bottom, split, windows, usable) in REELS.items():
        src = os.path.join(SRC, f"reel_{rid}.mp4")
        if not os.path.exists(src):
            print("missing", src)
            continue
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        H, W = frames[0].shape[:2]
        y0, y1 = int(top * H), int(bottom * H)
        h_out = round(W_OUT * (y1 - y0) / W)
        for k, (start, end) in enumerate(windows):
            a, b = max(0.0, start - PRE), min(usable, end + POST)
            name = f"reel_{rid}_{k}.mp4"
            vw = cv2.VideoWriter(os.path.join(OUT, name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_OUT, h_out))
            for f in frames[int(a * fps):int(b * fps)]:
                vw.write(cv2.resize(f[y0:y1], (W_OUT, h_out), interpolation=cv2.INTER_AREA))
            vw.release()
            rows.append({"file": name, "start": round(start - a, 2), "end": round(end - a, 2), "type": "any",
                         "group": "reel", "light": "night" if rid in NIGHT else "day", "weather": "?",
                         "scene": "urban", "split": split, "text": "aggressive driving in dense traffic (reel)"})
        print(rid, len(windows), "clips")
    with open(os.path.join(SRC, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "clips ->", OUT)


if __name__ == "__main__":
    main()
