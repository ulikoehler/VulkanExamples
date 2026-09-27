# SPDX-License-Identifier: CC0-1.0
import re, subprocess

o = subprocess.run(["./app", "--headless", "out.ppm"], check=True,
                   capture_output=True, text=True).stdout
assert "wrote out.ppm" in o, o

d = open("out.ppm", "rb").read()
m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(\d+)\s", d)
w, h = int(m.group(1)), int(m.group(2))
px = d[m.end():m.end() + w * h * 3]
def P(x, y):
    i = (y * w + x) * 3
    return px[i], px[i + 1], px[i + 2]

def navy(c):  return c[2] > 60 and c[0] < 40
def isSrc(c): return not navy(c)

# FIT cell (left third): top of cell is navy letterbox, center is image
assert navy(P(w//6, int(0.10*h))), "FIT: expected letterbox band on top"
assert isSrc(P(w//6, h//2)),       "FIT: cell center should be image"
# FILL cell (middle third): reaches top of its cell — no bands
assert isSrc(P(w//2, int(0.10*h))), "FILL: should reach the cell top"
# COVER cell (right third): fills cell; left/right borders were
# cropped away -> at its left edge there is NO yellow source border
edge = P(int(0.68*w), h//2)
assert not (edge[0] > 200 and edge[1] > 200), \
    f"COVER: left border visible, crop failed? {edge}"
assert isSrc(P(int(0.83*w), h//2)), "COVER: cell center should be image"
print("OK: fit letterboxes, fill stretches, cover crops")
