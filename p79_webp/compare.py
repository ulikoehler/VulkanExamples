#!/usr/bin/env python3
# Visual comparison: source pattern vs GPU-encoded WebP.
# Writes compare.png = [ INPUT | DECODED | |diff| x8 ].
import subprocess, sys, os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists("app") or not os.path.exists("out.webp"):
    r = subprocess.run("g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc".split(),
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr); sys.exit("BUILD FAILED")
    r = subprocess.run(["./app", "out.webp"], capture_output=True, text=True, timeout=60)
    print(r.stdout.strip())
    if r.returncode:
        sys.exit(1)

from PIL import Image, ImageDraw, ImageFont

w, h = 640, 480
yy, xx = np.mgrid[0:h, 0:w]
src = np.zeros((h, w, 3), np.uint8)
src[..., 0] = xx * 255 // w
src[..., 1] = yy * 255 // h
src[..., 2] = np.where(((xx // 40) ^ (yy // 40)) & 1, 220, 40)
src[:, w // 2:w // 2 + 8] = [255, 0, 0]

dec = np.asarray(Image.open("out.webp").convert("RGB"))
diff = np.abs(src.astype(np.int16) - dec.astype(np.int16)).max(-1)
dvis = np.clip(diff * 8, 0, 255).astype(np.uint8)

font = ImageFont.load_default()
hdr, gap = 22, 6
cv = Image.new("RGB", (w * 3 + gap * 4, h + hdr + gap * 2), (24, 24, 28))
dr = ImageDraw.Draw(cv)
labels = ["input - procedural source",
          "decoded - out.webp via libwebp",
          f"|diff| x8 - max {diff.max()}, mean {diff.mean():.3f}"]
for i, (arr, lab) in enumerate(zip([src, dec, np.stack([dvis] * 3, -1)], labels)):
    x = gap + i * (w + gap)
    dr.text((x + 4, 6), lab, font=font, fill=(230, 230, 230))
    cv.paste(Image.fromarray(arr), (x, hdr + gap // 2))
cv.save("compare.png")
print(f"wrote compare.png (max diff {diff.max()}, mean {diff.mean():.4f})")
