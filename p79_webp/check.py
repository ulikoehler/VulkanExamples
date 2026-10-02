#!/usr/bin/env python3
# Verify GPU WebP lossless encoder: PIL must decode out.webp
# bit-exactly against the regenerated source pattern.
import subprocess, sys, os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
build = "g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc"
r = subprocess.run(build.split(), capture_output=True, text=True)
if r.returncode:
    print(r.stderr); sys.exit("BUILD FAILED")

r = subprocess.run(["./app", "out.webp"],
                   capture_output=True, text=True, timeout=60)
print(r.stdout.strip())
if r.returncode or "SKIP" in r.stdout:
    sys.exit(0 if "SKIP" in r.stdout else 1)

from PIL import Image
im = np.asarray(Image.open("out.webp").convert("RGBA"))
w, h = 640, 480
px = np.zeros((h, w, 4), np.uint8)
yy, xx = np.mgrid[0:h, 0:w]
px[..., 0] = xx * 255 // w
px[..., 1] = yy * 255 // h
px[..., 2] = np.where(((xx // 40) ^ (yy // 40)) & 1, 220, 40)
px[..., 3] = 255
px[:, w // 2:w // 2 + 8] = [255, 0, 0, 255]
assert np.array_equal(im, px), "PIL-decoded WebP != source pattern"
print("OK: GPU-encoded lossless WebP decodes bit-exact via PIL/libwebp")
