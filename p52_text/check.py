# SPDX-License-Identifier: CC0-1.0
import re, subprocess, sys

o = subprocess.run(["./app", "--headless", "out.ppm"], check=True,
                   capture_output=True, text=True).stdout
m = re.search(r"shaped (\d+) glyphs, (\d+) vertices", o)
assert m and int(m.group(1)) > 10, o

d = open("out.ppm", "rb").read()
m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(\d+)\s", d)
w, h = int(m.group(1)), int(m.group(2))
px = d[m.end():m.end() + w * h * 3]

# text band: golden (r>180, g>120, b<120) pixels must exist and be
# well inside the text row — SDF edges produce intermediate values
def P(x, y):
    i = (y * w + x) * 3
    return px[i], px[i + 1], px[i + 2]

golden = sum(1 for y in range(150, 330, 4) for x in range(0, w, 4)
             if P(x, y)[0] > 180 and P(x, y)[1] > 120
             and P(x, y)[2] < 120)
assert golden > 200, f"only {golden} text pixels"
# smooth edges: intermediate grays along the boundary
mid = sum(1 for y in range(150, 330) for x in range(0, w, 3)
          if 60 < P(x, y)[0] < 170)
assert mid > 50, "edges look binary — SDF AA missing?"
print(f"OK: shaped text rendered, {golden} golden px, {mid} AA edge px")
