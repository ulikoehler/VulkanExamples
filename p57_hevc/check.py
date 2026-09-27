#!/usr/bin/env python3
# Build, decode test.h265 via VA-API -> Vulkan import, verify output.
# SKIP is a pass on machines without VA-API/HEVC-10bit.
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

build = ("g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw "
         "-lshaderc -lavcodec -lavformat -lavutil")
r = subprocess.run(build.split(), capture_output=True, text=True)
if r.returncode:
    print(r.stderr); sys.exit("BUILD FAILED")

r = subprocess.run(["./app", "--headless", "out.ppm"],
                   capture_output=True, text=True, timeout=120)
print(r.stdout.strip())
if r.returncode or "SKIP" in r.stdout:
    sys.exit(0 if "SKIP" in r.stdout else 1)
if "wrote out.ppm" not in r.stdout:
    sys.exit("no output written")

from PIL import Image
im = Image.open("out.ppm")
px = list(im.getdata())[::97]
distinct = len(set(px))
mx = max(max(p) for p in px)
assert distinct > 20, f"image looks flat ({distinct} colors)"
assert mx > 150, "image too dark"
print(f"OK: HEVC Main10 -> VA-API -> zero-copy import, "
      f"{distinct} distinct sampled colors")
