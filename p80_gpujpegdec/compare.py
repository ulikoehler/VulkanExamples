#!/usr/bin/env python3
# Visual comparison: input JPEGs vs GPU-decoded PPMs.
# Writes compare.png - one row per variant:
# [ input JPEG (decoded by PIL) | GPU-decoded | |diff| x8 ].
import subprocess, sys, os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists("app"):
    r = subprocess.run("g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc -ljpeg".split(),
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr); sys.exit("BUILD FAILED")

from PIL import Image, ImageDraw, ImageFont

def ppm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P6"
        w, h = map(int, f.readline().split()); f.readline()
        return np.frombuffer(f.read(), np.uint8).reshape(h, w, 3)

# regenerate the same corpus check.py uses (small subset, one per variant)
rng = np.random.RandomState(42)
yy, xx = np.mgrid[0:360, 0:640]
photo = np.stack([np.sin(xx / 23.) * 127 + 128,
                  np.cos(yy / 31.) * 127 + 128,
                  ((xx // 17) ^ (yy // 19)) * 40
                  + rng.randint(0, 30, (360, 640))], 2).astype(np.uint8)

tests = []
Image.fromarray(photo, "RGB").save("t_444.jpg", quality=92, subsampling=0)
tests.append(("t_444", "4:4:4"))
Image.fromarray(photo, "RGB").save("t_422.jpg", quality=90, subsampling=1)
tests.append(("t_422", "4:2:2"))
Image.fromarray(photo, "RGB").save("t_420.jpg", quality=90, subsampling=2)
tests.append(("t_420", "4:2:0"))
Image.fromarray(photo).convert("L").save("t_gray.jpg", quality=95)
tests.append(("t_gray", "gray"))
Image.fromarray(photo, "RGB").save("t_src.ppm")
subprocess.run(["cjpeg", "-restart", "1", "-quality", "88",
                "-outfile", "t_rst.jpg", "t_src.ppm"], capture_output=True)
tests.append(("t_rst", "restart int 1"))

font = ImageFont.load_default()
hdr, gap, lab_w = 20, 6, 130
W, H = 640, 360
cv = Image.new("RGB", (lab_w + W * 3 + gap * 5, hdr + H * len(tests) + gap * (len(tests) + 1)),
               (24, 24, 28))
dr = ImageDraw.Draw(cv)
for i, lab in enumerate(["libjpeg-decoded", "GPU-decoded", "|diff| x8"]):
    dr.text((lab_w + gap + i * (W + gap) + 4, 4), lab, font=font, fill=(230, 230, 230))

worst = 0
for r_i, (name, desc) in enumerate(tests):
    r = subprocess.run(["./app", name + ".jpg", name + ".ppm"],
                       capture_output=True, text=True)
    if r.returncode:
        print(name, "FAILED:", r.stderr); continue
    ref = np.asarray(Image.open(name + ".jpg").convert("RGB"))
    got = ppm(name + ".ppm")
    diff = np.abs(ref.astype(int) - got.astype(int)).astype(np.uint8)
    worst = max(worst, int(diff.max()))
    y = hdr + r_i * (H + gap) + gap
    dr.text((4, y + H // 2 - 4), f"{name}\n{desc}", font=font, fill=(200, 200, 200))
    cv.paste(Image.fromarray(ref), (lab_w + gap, y))
    cv.paste(Image.fromarray(got), (lab_w + gap + (W + gap), y))
    cv.paste(Image.fromarray(np.clip(diff * 8, 0, 255).astype(np.uint8)),
             (lab_w + gap + 2 * (W + gap), y))
    print(f"{name}: max |diff| = {diff.max()}")

cv.save("compare.png")
print(f"wrote compare.png  (worst pixel diff {worst})")
