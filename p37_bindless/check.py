# SPDX-License-Identifier: CC0-1.0
import re, subprocess, sys

subprocess.run(["./app", "--headless", "out.ppm"], check=True,
               capture_output=True)
d = open("out.ppm", "rb").read()
m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(\d+)\s", d)
w, h = int(m.group(1)), int(m.group(2))
px = d[m.end():m.end() + w * h * 3]

def P(x, y):
    i = (y * w + x) * 3
    return px[i], px[i + 1], px[i + 2]

# one fullscreen draw; fragment picks tex idx per quadrant:
# top-left=R, top-right=G, bottom-left=B, bottom-right=W
assert P(w // 4, h // 4) == (255, 51, 51), P(w // 4, h // 4)
assert P(3 * w // 4, h // 4) == (51, 255, 51), P(3 * w // 4, h // 4)
assert P(w // 4, 3 * h // 4) == (51, 51, 255), P(w // 4, 3 * h // 4)
assert P(3 * w // 4, 3 * h // 4) == (255, 255, 255), \
    P(3 * w // 4, 3 * h // 4)
print("OK: 4 quadrant colors from 1 draw via per-fragment index")
