"""Скачать N роликов BDD100K (40 с, 720p, Нью-Йорк / Сан-Франциско, день и ночь) без всего архива.

Официальный сервер BDD100K (dl.cv.ethz.ch) больше не отвечает. На Hugging Face лежит побайтовая копия
архива bdd100k_videos.zip (1.8 ТБ), нарезанная на части по 100 ГиБ:
    https://huggingface.co/datasets/linxxx3/bdd100k_videos
Ролики в архиве не сжаты (метод STORED) и идут подряд, поэтому по заголовкам zip можно пройти
запросами по диапазону байт и вырезать только нужные файлы (~20 МБ каждый).

    python tools/get_bdd_videos.py --n 50
    python tools/get_bdd_videos.py --n 50 --skip 200      # следующие ролики

Лицензия BDD100K — для исследований (см. https://github.com/bdd100k/bdd100k).
Разметки опасностей у BDD нет: это «обычная» езда в плотном трафике — для подсчёта ложных
срабатываний и средних по риску ситуаций (подрезания, плотная колонна, пешеходы).
"""
import argparse
import os
import struct
import urllib.request

URL = ("https://huggingface.co/datasets/linxxx3/bdd100k_videos/resolve/main/"
       "bdd100k_videos/bdd100k_videos.zip.part-00000")


def fetch(start, length):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def entries():
    """Идём по локальным заголовкам zip: (имя, смещение данных, размер)."""
    pos = 0
    while True:
        head = fetch(pos, 1024)
        if head[:4] != b"PK\x03\x04":
            return
        _, flag, method, _, _, _, csize, _, nlen, elen = struct.unpack("<HHHHHIIIHH", head[4:30])
        name = head[30:30 + nlen].decode()
        extra = head[30 + nlen:30 + nlen + elen]
        j = 0
        while j + 4 <= len(extra):                     # zip64: настоящий размер в доп. поле
            hid, hs = struct.unpack("<HH", extra[j:j + 4])
            if hid == 1 and csize == 0xFFFFFFFF:
                csize = struct.unpack("<Q", extra[j + 4 + 8:j + 4 + 16] if hs >= 16 else extra[j + 4:j + 12])[0]
            j += 4 + hs
        if flag & 8 or method != 0:
            raise SystemExit(f"{name}: неожиданный формат (flag={flag}, method={method})")
        data = pos + 30 + nlen + elen
        yield name, data, csize
        pos = data + csize


def main():
    ap = argparse.ArgumentParser(description="Скачать часть роликов BDD100K")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--skip", type=int, default=0, help="пропустить первые N роликов")
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
        print(f"[{got}/{args.n}] {os.path.basename(name)}  {size / 1e6:.0f} МБ", flush=True)
        if got >= args.n:
            break


if __name__ == "__main__":
    main()
