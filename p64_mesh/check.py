#!/usr/bin/env python3
# Verify mesh-shader output: 4 distinct colored quadrants.
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
# quadrant centers should be 4 distinct bright colors
cs = [px[(h//4 + dy)*(w) + (w//4 + dx)]
      for dy in (0, h//2) for dx in (0, w//2)]
distinct = len(set(cs))
assert distinct == 4, f"expected 4 quadrant colors, got {distinct}: {cs}"
black = sum(1 for p in px if max(p) < 30)
assert black > 1000, "no gaps — quads should have spacing"
print(f"OK: mesh shader emitted 4 procedural quads, colors {cs}")
