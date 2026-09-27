#!/usr/bin/env python3
# Build + run, verify the cycle heatmap: right half must be
# measurably "hotter" (more red) than the left.
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
build = "g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc"
r = subprocess.run(build.split(), capture_output=True, text=True)
if r.returncode:
    print(r.stderr); sys.exit("BUILD FAILED")

r = subprocess.run(["./app", "--headless", "out.ppm"],
                   capture_output=True, text=True, timeout=60)
print(r.stdout.strip())
if r.returncode or "SKIP" in r.stdout:
    sys.exit(0 if "SKIP" in r.stdout else 1)

from PIL import Image
im = Image.open("out.ppm")
w, h = im.size
px = list(im.getdata())

def avg_red(region):
    ps = [px[y*w+x] for x,y in region]
    return sum(p[0] for p in ps) / len(ps)

# sample strips: left quarter (cheap) vs right quarter (expensive)
left  = avg_red([(x, y) for x in range(20, 120)
                         for y in range(40, h-40, 7)])
right = avg_red([(x, y) for x in range(w-140, w-20)
                         for y in range(h-100, h-20, 5)])
print(f"left(8 iters) avg red={left:.1f}  "
      f"right(40-240 iters) avg red={right:.1f}")
assert right > left + 30, \
    "clock readback didn't distinguish cheap vs expensive work"
print("OK: shader clock heatmap — expensive pixels measurably hotter")
