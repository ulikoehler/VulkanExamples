# SPDX-License-Identifier: CC0-1.0
import re, subprocess

subprocess.run(["./app", "--headless", "out.ppm"], check=True,
               capture_output=True)
d = open("out.ppm", "rb").read()
m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(\d+)\s", d)
w, h = int(m.group(1)), int(m.group(2))
px = d[m.end():m.end() + w * h * 3]

def P(x, y):
    i = (y * w + x) * 3
    return px[i], px[i + 1], px[i + 2]

def near(a, b, tol=6):
    return all(abs(x - y) <= tol for x, y in zip(a, b))

# quadrants of the NV12 image: TL=red TR=green BL=blue BR=white
assert near(P(w // 4, h // 4), (255, 0, 0)), P(w // 4, h // 4)
assert near(P(3 * w // 4, h // 4), (0, 255, 0)), P(3 * w // 4, h // 4)
assert near(P(w // 4, 3 * h // 4), (0, 0, 255)), P(w // 4, 3 * h // 4)
assert near(P(3 * w // 4, 3 * h // 4), (255, 255, 255)), \
    P(3 * w // 4, 3 * h // 4)
print("OK: NV12 YCbCr converted to RGB inside the sampler")
