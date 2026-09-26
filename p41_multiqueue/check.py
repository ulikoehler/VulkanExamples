# SPDX-License-Identifier: CC0-1.0
import re, subprocess

o = subprocess.run(["./app", "--headless", "out.ppm"], check=True,
                   capture_output=True, text=True).stdout
assert "timeline signaled" in o, o
fam = re.search(r"dedicated transfer queue fam (\d+) \(gfx fam (\d+)\)", o)
if fam:
    assert fam.group(1) != fam.group(2)

d = open("out.ppm", "rb").read()
m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(\d+)\s", d)
w, h = int(m.group(1)), int(m.group(2))
px = d[m.end():m.end() + w * h * 3]

def P(x, y):
    i = (y * w + x) * 3
    return px[i], px[i + 1], px[i + 2]

# checkerboard arrived via the transfer queue:
cells = {P(x, y) for y in range(0, h, 16) for x in range(0, w, 16)}
assert cells == {(51, 51, 51), (255, 255, 255)}, cells
print("OK: checkerboard uploaded via transfer queue, sampled by gfx")
