#!/usr/bin/env python3
# Build, render headless, verify the pushed-descriptor texture.
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
build = "g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc"
r = subprocess.run(build.split(), capture_output=True, text=True)
if r.returncode:
    print(r.stderr); sys.exit("BUILD FAILED")

o = subprocess.run(["./app", "--headless", "out.ppm"], check=True,
                   capture_output=True, text=True).stdout
if "SKIP" in o:
    print(o.strip()); sys.exit(0)

from PIL import Image
from collections import Counter
im = Image.open("out.ppm")
c = Counter(im.getdata())
assert len(c) == 2, f"expected 2 colors, got {len(c)}"
cols = set(c)
assert (255, 0, 0) in cols and (0, 255, 0) in cols, cols
print(f"OK: push-descriptor texture sampled, {c[(255,0,0)]} red + "
      f"{c[(0,255,0)]} green px")
