#!/usr/bin/env python3
# Visual comparison: per-conversion GPU output vs PIL.
# Writes compare.png - one row per conversion:
# [ source | PIL reference | GPU | |diff| x8 ].
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
yy, xx = np.mgrid[0:240, 0:320]
photo = np.stack([np.sin(xx / 23.) * 127 + 128,
                  np.cos(yy / 31.) * 127 + 128,
                  ((xx // 17) ^ (yy // 19)) * 40
                  + rng.randint(0, 30, (240, 320))], 2).astype(np.uint8)

CASES = ["rgb2gray", "rgb2ycbcr", "rgb2hsv", "rgb2bgr"]

def ref(im, conv):
    if conv == "rgb2gray":
        return np.asarray(im.convert("L"))
    if conv == "rgb2ycbcr":
        return np.asarray(im.convert("YCbCr"))
    if conv == "rgb2hsv":
        return np.asarray(im.convert("HSV"))
    if conv == "rgb2bgr":
        return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])

font = ImageFont.load_default()
hdr, gap, lab_w = 20, 6, 110
W, H = 320, 240
cv = Image.new("RGB", (lab_w + W * 4 + gap * 6,
                       hdr + H * len(CASES) + gap * (len(CASES) + 1)),
               (24, 24, 28))
dr = ImageDraw.Draw(cv)
for i, lab in enumerate(["source", "PIL", "GPU", "|diff| x8"]):
    dr.text((lab_w + gap + i * (W + gap) + 4, 4), lab, font=font,
            fill=(230, 230, 230))

im = Image.fromarray(photo, "RGB")
worst = 0
for r_i, conv in enumerate(CASES):
    np.ascontiguousarray(photo).tofile("t_src.raw")
    subprocess.run(["./app", "--raw", f"{W}x{H}x3", "t_src.raw",
                    "t_gpu.raw", conv], capture_output=True)
    b = ref(im, conv)
    a = np.fromfile("t_gpu.raw", np.uint8).reshape(b.shape)
    d = np.abs(a.astype(int) - b.astype(int)).astype(np.uint8)
    worst = max(worst, int(d.max()))
    y = hdr + r_i * (H + gap) + gap
    dr.text((4, y + H // 2 - 4), conv, font=font, fill=(200, 200, 200))
    cv.paste(im, (lab_w + gap, y))
    cv.paste(Image.fromarray(b.astype(np.uint8)), (lab_w + gap + (W + gap), y))
    cv.paste(Image.fromarray(a), (lab_w + gap + 2 * (W + gap), y))
    dd = d if d.ndim == 2 else d
    cv.paste(Image.fromarray(np.clip(dd * 8, 0, 255).astype(np.uint8)),
             (lab_w + gap + 3 * (W + gap), y))
    print(f"{conv}: max |diff| = {d.max()}")

cv.save("compare.png")
print(f"wrote compare.png  (worst pixel diff {worst})")
