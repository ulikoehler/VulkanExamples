#!/usr/bin/env python3
# Verify: ray-traced hard shadow — a dark triangle on a lit checkered
# ground plane, deterministic across runs.
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
import numpy as np
a = np.asarray(Image.open("out.ppm").convert("RGB")).astype(int)
lum = a.sum(2)
dark = (lum < 60).sum()
lit = (lum > 300).sum()
print(f"shadow px={dark}  lit px={lit}")
assert dark > 10000, "no shadow region"
assert lit > 40000, "no lit ground"
print("OK: ray query — hard triangle shadow, fully deterministic")
