"""Copy onnxruntime-web (the in-browser model runtime) into web/vendor/ort so the site works offline.

    python tools/get_ort.py

The page (web/index.html) uses this copy when it exists and the CDN otherwise. ~25 MB, MIT licence.
"""
import os
import urllib.request

VERSION = "1.22.0"
BASE = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{VERSION}/dist/"
FILES = ["ort.webgpu.min.js", "ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "vendor", "ort")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in FILES:
        path = os.path.join(OUT, name)
        tmp = path + ".part"
        with urllib.request.urlopen(BASE + name, timeout=60) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, path)
        print(f"{name:40s} {os.path.getsize(path) / 1e6:6.1f} MB")
    with open(os.path.join(OUT, "VERSION.txt"), "w", encoding="utf-8") as f:
        f.write(f"onnxruntime-web {VERSION} (MIT licence) from {BASE}\n")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
