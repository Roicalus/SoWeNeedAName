"""Download N BDD100K videos (40 s, 720p, New York / San Francisco, day and night) without the whole archive.

The official BDD100K server (dl.cv.ethz.ch) no longer answers. Hugging Face has a byte-exact copy of
bdd100k_videos.zip (1.8 TB) split into 100 GiB parts:
    https://huggingface.co/datasets/linxxx3/bdd100k_videos
Videos in the archive are not compressed (STORED) and lie one after another, so we walk the zip
local headers with byte-range requests and cut out only the files we need (~20 MB each).

    python tools/get_bdd_videos.py --n 50
    python tools/get_bdd_videos.py --n 50 --skip 200      # the next videos

BDD100K licence: research use (see https://github.com/bdd100k/bdd100k).
BDD has no danger labels: it is "normal" driving in dense traffic - used to count false alarms and
medium-risk situations (cut-ins, dense queues, pedestrians).
"""
import argparse
import os
import struct
import time
import urllib.request

URL = ("https://huggingface.co/datasets/linxxx3/bdd100k_videos/resolve/main/"
       "bdd100k_videos/bdd100k_videos.zip.part-00000")


def fetch(start, length):
    """A byte range of the archive; connection resets and the HF limit (429) are retried with a pause."""
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{start + length - 1}"})
    for attempt in range(20):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == length:
                return data
        except Exception as e:
            print(f"  retry ({e})", flush=True)
        time.sleep(min(60, 5 * (attempt + 1)))
    raise SystemExit("could not download an archive range")


def entries():
    """Walk the zip local headers: (name, data offset, size)."""
    pos = 0
    while True:
        head = fetch(pos, 1024)
        if head[:4] != b"PK\x03\x04":
            return
        _, flag, method, _, _, _, csize, _, nlen, elen = struct.unpack("<HHHHHIIIHH", head[4:30])
        name = head[30:30 + nlen].decode()
        extra = head[30 + nlen:30 + nlen + elen]
        j = 0
        while j + 4 <= len(extra):                     # zip64: the real size is in the extra field
            hid, hs = struct.unpack("<HH", extra[j:j + 4])
            if hid == 1 and csize == 0xFFFFFFFF:
                csize = struct.unpack("<Q", extra[j + 4 + 8:j + 4 + 16] if hs >= 16 else extra[j + 4:j + 12])[0]
            j += 4 + hs
        if flag & 8 or method != 0:
            raise SystemExit(f"{name}: unexpected format (flag={flag}, method={method})")
        data = pos + 30 + nlen + elen
        yield name, data, csize
        pos = data + csize


def main():
    ap = argparse.ArgumentParser(description="Download part of the BDD100K videos")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N videos")
    ap.add_argument("--out", default="data/bdd")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    got = seen = 0
    for name, start, size in entries():
        if not name.endswith(".mov"):
            continue
        seen += 1
        if seen <= args.skip:
            continue
        dst = os.path.join(args.out, os.path.basename(name))
        if not (os.path.exists(dst) and os.path.getsize(dst) == size):
            with open(dst + ".part", "wb") as f:
                for off in range(0, size, 8 << 20):
                    f.write(fetch(start + off, min(8 << 20, size - off)))
            os.replace(dst + ".part", dst)
        got += 1
        print(f"[{got}/{args.n}] {os.path.basename(name)}  {size / 1e6:.0f} MB", flush=True)
        if got >= args.n:
            break


if __name__ == "__main__":
    main()
