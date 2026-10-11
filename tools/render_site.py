"""Render a clip for the site: driver view (with sound) + "how the system sees it", poster, entry in reels.json.

Runs debug_video.py twice (--demo and the full debug view), makes the files streamable in a browser
(H.264 + AAC, moov atom first) and adds / replaces the card in web/data/reels.json.

    python tools/render_site.py data/videos/krakow_peds.mp4 --id krakow --title "City · pedestrians" \
        --caption "Kraków: people step onto a zebra in front of us." --start 60 --dur 30
    python tools/render_site.py web/samples/crash_brake.mp4 --id crash --title "Car ahead brakes · crash"

Output: web/renders/<id>_driver.mp4, <id>_system.mp4, <id>.jpg. Commit them to show them on the site.
Best run on the PC with the lane model and a GPU: the renders then match the published numbers.
"""
import argparse
import json
import os
import subprocess
import sys

import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def render(src, extra, demo, out_dir):
    cmd = [sys.executable, os.path.join(ROOT, "debug_video.py"), src, "--out", out_dir] + extra + (["--demo"] if demo else [])
    subprocess.run(cmd, cwd=ROOT, check=True)
    stem = os.path.splitext(os.path.basename(src))[0]
    return os.path.join(out_dir, f"{stem}_{'demo' if demo else 'debug'}.mp4")


def web_copy(src, dst):
    """Streamable copy: moov atom first so the browser starts playing before the whole file arrives."""
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", src, "-c", "copy", "-movflags", "+faststart", dst], check=True)


def main():
    ap = argparse.ArgumentParser(description="Render a clip for the site")
    ap.add_argument("video")
    ap.add_argument("--id", required=True, help="short name, used for file names and to replace a card")
    ap.add_argument("--title", required=True)
    ap.add_argument("--caption", default="")
    ap.add_argument("--start", type=float, help="start at second N")
    ap.add_argument("--dur", type=float, help="render N seconds")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--driver-only", action="store_true", help="skip the system view")
    args = ap.parse_args()

    extra = ["--width", str(args.width)]
    if args.start is not None:
        extra += ["--start", str(args.start)]
    if args.dur is not None:
        extra += ["--dur", str(args.dur)]
    tmp = os.path.join(ROOT, "debug", "site")
    out = os.path.join(WEB, "renders")
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    views = {}
    for name, demo in [("driver", True)] + ([] if args.driver_only else [("system", False)]):
        path = render(args.video, extra, demo, tmp)
        dst = os.path.join(out, f"{args.id}_{name}.mp4")
        web_copy(path, dst)
        views[name] = f"renders/{args.id}_{name}.mp4"
        print(f"{dst}: {os.path.getsize(dst) / 1e6:.1f} MB")

    poster = os.path.join(out, f"{args.id}.jpg")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", "1", "-i", os.path.join(out, f"{args.id}_driver.mp4"),
                    "-frames:v", "1", "-q:v", "4", poster], check=True)

    index = os.path.join(WEB, "data", "reels.json")
    reels = json.load(open(index, encoding="utf-8")) if os.path.exists(index) else []
    entry = {"id": args.id, "title": args.title, "caption": args.caption, "poster": f"renders/{args.id}.jpg", "views": views}
    reels = [r for r in reels if r.get("id") != args.id] + [entry]
    with open(index, "w", encoding="utf-8") as f:
        json.dump(reels, f, ensure_ascii=False, indent=2)
    print(f"{index}: {len(reels)} clips")


if __name__ == "__main__":
    main()
