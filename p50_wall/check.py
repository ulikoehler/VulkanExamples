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

# nine tiles, nine distinct center colors (each has a different
# pattern + hue shift)
centers = set()
for gy in range(3):
    for gx in range(3):
        centers.add(P((gx * w // 3) + w // 6, (gy * h // 3) + h // 6))
assert len(centers) == 9, f"tiles not distinct: {centers}"

# zoomed tile must fill the frame: center and corners are tile pixels
o = subprocess.run(["./app", "--headless", "z.ppm", "--zoom", "4"],
                   check=True, capture_output=True, text=True).stdout
d = open("z.ppm", "rb").read()
m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(\d+)\s", d)
w, h = int(m.group(1)), int(m.group(2))
px = d[m.end():m.end() + w * h * 3]
bg = (13, 13, 20)  # clear color — must NOT appear after zoom
samples = {P(10, 10), P(w - 10, 10), P(10, h - 10),
           P(w - 10, h - 10), P(w // 2, h // 2)}
assert bg not in samples, f"clear color still visible: {samples}"
print("OK: 3x3 grid, 9 distinct tiles, zoom fills frame")
