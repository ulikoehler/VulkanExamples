#!/usr/bin/env python3
# Visual comparison: input PNGs vs GPU-decoded PPMs.
# Writes compare.png - one row per DEFLATE variant:
# [ input PNG | GPU-decoded | |diff| x8 ].
import subprocess, sys, os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists("app"):
    r = subprocess.run("g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc".split(),
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr); sys.exit("BUILD FAILED")

from PIL import Image, ImageDraw, ImageFont

def ppm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P6"
        w, h = map(int, f.readline().split()); f.readline()
        return np.frombuffer(f.read(), np.uint8).reshape(h, w, 3)

# regenerate the same test set check.py uses
rng = np.random.RandomState(42)
yy, xx = np.mgrid[0:360, 0:640]
photo = np.stack([np.sin(xx / 23.) * 127 + 128,
                  np.cos(yy / 31.) * 127 + 128,
                  ((xx // 17) ^ (yy // 19)) * 40
                  + rng.randint(0, 30, (360, 640))], 2).astype(np.uint8)
tiny = np.zeros((8, 8, 3), np.uint8); tiny[2:6, 2:6] = [255, 0, 0]

tests = []
Image.fromarray(photo, "RGB").save("t_dyn.png");                  tests.append(("t_dyn",    "dynamic huffman"))
Image.fromarray(photo, "RGB").save("t_stored.png", compress_level=0); tests.append(("t_stored", "stored blocks"))
Image.fromarray(photo).convert("L").save("t_gray.png");           tests.append(("t_gray",   "gray ctype 0"))
Image.fromarray(photo, "RGB").save("t_l9.png", compress_level=9); tests.append(("t_l9",     "level 9"))
Image.fromarray(tiny).save("t_tiny.png");                         tests.append(("t_tiny",   "fixed huffman"))

font = ImageFont.load_default()
hdr, gap, lab_w = 20, 6, 130
W, H = 640, 360
cv = Image.new("RGB", (lab_w + W * 3 + gap * 5, hdr + H * len(tests) + gap * (len(tests) + 1)),
               (24, 24, 28))
dr = ImageDraw.Draw(cv)
for i, lab in enumerate(["input PNG", "GPU-decoded", "|diff| x8"]):
    dr.text((lab_w + gap + i * (W + gap) + 4, 4), lab, font=font, fill=(230, 230, 230))

worst = 0
for r_i, (name, desc) in enumerate(tests):
    r = subprocess.run(["./app", name + ".png", name + ".ppm"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode:
        print(r.stdout, r.stderr); sys.exit(1)
    ref = np.asarray(Image.open(name + ".png").convert("RGB"))
    got = ppm(name + ".ppm")
    diff = np.abs(ref.astype(np.int16) - got.astype(np.int16)).max(-1)
    worst = max(worst, int(diff.max()))
    y = hdr + gap + r_i * (H + gap)
    dr.text((4, y + H // 2 - 4), f"{name}\n{desc}", font=font, fill=(200, 200, 200))
    for c_i, arr in enumerate([ref, got, np.clip(diff * 8, 0, 255).astype(np.uint8)]):
        im = Image.fromarray(arr if arr.ndim == 3 else np.stack([arr] * 3, -1))
        if im.size != (W, H):
            im = im.resize((W, H), Image.NEAREST)
        cv.paste(im, (lab_w + gap + c_i * (W + gap), y))
cv.save("compare.png")
print(f"wrote compare.png ({len(tests)} variants, worst max diff {worst})")
