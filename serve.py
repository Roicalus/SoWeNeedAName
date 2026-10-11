"""Run the Co-pilot site locally - no GitHub, no install, only Python's standard library.

    python serve.py                 # http://localhost:8099, opens the browser
    python serve.py --port 9000 --no-browser

The camera and WebGPU need a secure page; http://localhost counts as one, a LAN address does not.
For a fully offline demo run tools/get_ort.py once: the page then takes onnxruntime-web from
web/vendor/ort instead of the CDN (fonts fall back to system ones).
"""
import argparse
import functools
import http.server
import mimetypes
import os
import sys
import threading
import webbrowser

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
TYPES = {".wasm": "application/wasm", ".mjs": "text/javascript", ".js": "text/javascript",
         ".onnx": "application/octet-stream", ".json": "application/json", ".mp4": "video/mp4"}


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, **TYPES}

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")      # edits show up on reload
        super().end_headers()

    def log_message(self, fmt, *args):
        if not str(args[1] if len(args) > 1 else "").startswith(("2", "3")):    # only errors
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser(description="Serve the Co-pilot site on localhost")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    for ext, typ in TYPES.items():
        mimetypes.add_type(typ, ext)
    handler = functools.partial(Handler, directory=WEB)
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError:
        sys.exit(f"Port {args.port} is busy - is the site already running? Try: python serve.py --port 9000")
    url = f"http://localhost:{args.port}/"
    local = os.path.exists(os.path.join(WEB, "vendor", "ort", "ort.webgpu.min.js"))
    print(f"Co-pilot site: {url}")
    print("onnxruntime-web: " + ("local copy (works offline)" if local else "CDN (needs internet; run tools/get_ort.py for offline)"))
    print("Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
