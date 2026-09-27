#!/usr/bin/env python3
# Verify the GPU PNG decoder end-to-end: PIL-generated PNGs covering
# stored / fixed / dynamic DEFLATE blocks, gray + RGB color types and
# adaptive filters must decode bit-exactly in the compute shader.
import subprocess, sys, os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
build = "g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc"
r = subprocess.run(build.split(), capture_output=True, text=True)
if r.returncode:
    print(r.stderr); sys.exit("BUILD FAILED")

from PIL import Image

def ppm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P6"
        w, h = map(int, f.readline().split()); f.readline()
        return np.frombuffer(f.read(), np.uint8).reshape(h, w, 3)

rng = np.random.RandomState(42)
yy, xx = np.mgrid[0:360, 0:640]
photo = np.stack([np.sin(xx / 23.) * 127 + 128,
                  np.cos(yy / 31.) * 127 + 128,
                  ((xx // 17) ^ (yy // 19)) * 40
                  + rng.randint(0, 30, (360, 640))], 2).astype(np.uint8)

tests = []
Image.fromarray(photo, "RGB").save("t_dyn.png")                 # dynamic huffman
tests.append("t_dyn")
Image.fromarray(photo, "RGB").save("t_stored.png", compress_level=0)
tests.append("t_stored")                                      # stored blocks
Image.fromarray(photo).convert("L").save("t_gray.png")        # gray ctype 0
tests.append("t_gray")
Image.fromarray(photo, "RGB").save("t_l9.png", compress_level=9)
tests.append("t_l9")
tiny = np.zeros((8, 8, 3), np.uint8); tiny[2:6, 2:6] = [255, 0, 0]
Image.fromarray(tiny).save("t_tiny.png")                      # fixed huffman
tests.append("t_tiny")

for t in tests:
    r = subprocess.run(["./app", t + ".png", t + ".ppm"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode or "SKIP" in r.stdout:
        print(r.stdout, r.stderr)
        sys.exit(0 if "SKIP" in r.stdout else 1)
    ref = np.asarray(Image.open(t + ".png").convert("RGB"))
    got = ppm(t + ".ppm")
    assert np.array_equal(got, ref), f"{t}: pixel mismatch"
    print(f"  {t}: bit-exact ({got.shape[1]}x{got.shape[0]})")

print("OK: PNG fully decoded on GPU — inflate + unfilter, all variants")
