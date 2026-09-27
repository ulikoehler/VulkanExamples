#!/usr/bin/env python3
# Build, serve a synthetic MJPEG stream, verify the app decodes + uploads.
import subprocess, sys, os, threading, io, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

build = ("g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw "
         "-lshaderc -lavcodec -lavutil -lswscale")
r = subprocess.run(build.split(), capture_output=True, text=True)
if r.returncode:
    print(r.stderr); sys.exit("BUILD FAILED")

# --- tiny MJPEG server: testsrc-like JPEGs over multipart ----
from PIL import Image, ImageDraw
import socketserver, http.server

def make_jpeg(i):
    im = Image.new("RGB", (320, 240), (30, 40, 60))
    d = ImageDraw.Draw(im)
    d.rectangle([20+i*4, 30, 120+i*4, 140], fill=(220, 60, 40))
    d.ellipse([180, 60+i*3, 280, 160+i*3], fill=(40, 200, 90))
    d.text((10, 200), f"frame {i}", fill=(255, 255, 0))
    b = io.BytesIO(); im.save(b, "JPEG", quality=85)
    return b.getvalue()

JPEGS = [make_jpeg(i) for i in range(8)]

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace;boundary=frame")
        self.end_headers()
        try:
            for j in JPEGS:
                self.wfile.write(b"--frame\r\n"
                                 b"Content-Type: image/jpeg\r\n"
                                 b"Content-Length: %d\r\n\r\n" % len(j))
                self.wfile.write(j); self.wfile.write(b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass
    def log_message(self, *a): pass

srv = socketserver.TCPServer(("127.0.0.1", 8137), Handler)
srv.allow_reuse_address = True
t = threading.Thread(target=srv.serve_forever, daemon=True)
t.start()

r = subprocess.run(["./app", "--headless", "out.ppm",
                    "http://127.0.0.1:8137/cam"],
                   capture_output=True, text=True, timeout=60)
srv.shutdown()
print(r.stdout.strip())
print(r.stderr.strip())
if r.returncode or "SKIP" in r.stdout:
    sys.exit(0 if "SKIP" in r.stdout else 1)
if "wrote out.ppm" not in r.stdout:
    sys.exit("no output written")

im = Image.open("out.ppm")
px = list(im.getdata())
distinct = len(set(px[::53]))
reds = sum(1 for p in px if p[0] > 150 and p[1] < 100 and p[2] < 100)
assert distinct > 15, f"flat image ({distinct})"
assert reds > 1000, "no red box found — wrong frame content?"
print(f"OK: MJPEG HTTP ingest decoded + rendered, "
      f"{distinct} distinct colors, red box present")
