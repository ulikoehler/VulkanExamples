#!/usr/bin/env python3
# Build + run; verify the GPU-authored draw count produced a
# partially-filled grid (some cells culled, some drawn).
import subprocess, sys, os, re

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

m = re.search(r"drawCount=(\d+) \(of 256", r.stdout)
assert m, "no GPU draw count in output"
count = int(m.group(1))
assert 50 < count < 160, f"unexpected count {count}"

from PIL import Image
im = Image.open("out.ppm")
px = list(im.getdata())
nonblack = sum(1 for p in px if max(p) > 40)
assert 2000 < nonblack < 200000, \
    f"nonblack px {nonblack} — culled grid expected"
print(f"OK: GPU-driven drawIndirectCount, {count} draws, "
      f"{nonblack} lit px")
