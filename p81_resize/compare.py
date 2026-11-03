#!/usr/bin/env python3
# Visual comparison: per-filter GPU-resized output vs PIL.
# Writes compare.png - one row per filter:
# [ source (nearest-thumbnail) | PIL resized | GPU resized | |diff| x8 ].
import subprocess, sys, os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists("app"):
    r = subprocess.run("g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc".split(),
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr); sys.exit("BUILD FAILED")

from PIL import Image, ImageDraw, ImageFont

rng = np.random.RandomState(42)
yy, xx = np.mgrid[0:360, 0:640]
photo = np.stack([np.sin(xx / 23.) * 127 + 128,
                  np.cos(yy / 31.) * 127 + 128,
                  ((xx // 17) ^ (yy // 19)) * 40
                  + rng.randint(0, 30, (360, 640))], 2).astype(np.uint8)
Image.fromarray(photo, "RGB").save("t_src.ppm")

FILTERS = {"nearest": Image.NEAREST, "box": Image.BOX,
           "bilinear": Image.BILINEAR, "hamming": Image.HAMMING,
           "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}
OW, OH = 200, 113          # ~3.2x downscale

font = ImageFont.load_default()
hdr, gap, lab_w = 20, 6, 110
SW, SH = OW, OH
cv = Image.new("RGB", (lab_w + SW * 4 + gap * 6,
                       hdr + SH * len(FILTERS) + gap * (len(FILTERS) + 1)),
               (24, 24, 28))
dr = ImageDraw.Draw(cv)
for i, lab in enumerate(["source", "PIL", "GPU", "|diff| x8"]):
    dr.text((lab_w + gap + i * (SW + gap) + 4, 4), lab, font=font,
            fill=(230, 230, 230))

src_thumb = Image.open("t_src.ppm").resize((OW, OH), Image.NEAREST)
worst = 0
for r_i, (fname, resample) in enumerate(FILTERS.items()):
    subprocess.run(["./app", "t_src.ppm", "t_gpu.ppm",
                    f"{OW}x{OH}", fname], capture_output=True)
    ref = np.asarray(Image.open("t_src.ppm").resize((OW, OH), resample))
    got = np.asarray(Image.open("t_gpu.ppm"))
    diff = np.abs(ref.astype(int) - got.astype(int)).astype(np.uint8)
    worst = max(worst, int(diff.max()))
    y = hdr + r_i * (SH + gap) + gap
    dr.text((4, y + SH // 2 - 4), fname, font=font, fill=(200, 200, 200))
    cv.paste(src_thumb, (lab_w + gap, y))
    cv.paste(Image.fromarray(ref), (lab_w + gap + (SW + gap), y))
    cv.paste(Image.fromarray(got), (lab_w + gap + 2 * (SW + gap), y))
    cv.paste(Image.fromarray(np.clip(diff * 8, 0, 255).astype(np.uint8)),
             (lab_w + gap + 3 * (SW + gap), y))
    print(f"{fname}: max |diff| = {diff.max()}")

cv.save("compare.png")
print(f"wrote compare.png  (worst pixel diff {worst})")
