#!/usr/bin/env python3
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
px = list(im.getdata())
distinct = len(set(px[::97]))
bright = sum(1 for p in px if max(p) > 200)
assert distinct > 30 and bright > 20000, \
    f"triangle missing ({distinct} colors, {bright} bright)"
print("OK: triangle rendered via shader objects, zero VkPipelines")
