# SPDX-License-Identifier: CC0-1.0
import re, subprocess

o = subprocess.run(["./app", "--headless", "out.ppm"], check=True,
                   capture_output=True, text=True).stdout
# the export itself must have succeeded
assert re.search(r"exported dma-buf fd \d+", o), o
assert re.search(r"selected modifier 0x[0-9a-f]+", o), o

d = open("out.ppm", "rb").read()
m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(\d+)\s", d)
w, h = int(m.group(1)), int(m.group(2))
px = d[m.end():m.end() + w * h * 3]

def P(x, y):
    i = (y * w + x) * 3
    return px[i], px[i + 1], px[i + 2]

# image A was filled R/G checkerboard through ITS view of the memory;
# image B samples the SAME dma-buf. Both colors must appear.
cells = {P(x, y) for y in range(24, h, 48) for x in range(24, w, 48)}
assert cells == {(255, 51, 51), (51, 255, 51)}, cells
print("OK: checkerboard sampled through imported dma-buf image")
