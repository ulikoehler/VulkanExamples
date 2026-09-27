# SPDX-License-Identifier: CC0-1.0
import re, subprocess, sys

o = subprocess.run(["./app", "--headless", "out.ppm"], check=True,
                   capture_output=True, text=True).stdout
if "SKIP" in o:
    print(o.strip()); sys.exit(0)
assert re.search(r"descriptor blob: \d+ bytes", o), o
assert "wrote out.ppm" in o

d = open("out.ppm", "rb").read()
m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(\d+)\s", d)
w, h = int(m.group(1)), int(m.group(2))
px = d[m.end():m.end() + w * h * 3]
def P(x, y):
    i = (y * w + x) * 3
    return px[i], px[i + 1], px[i + 2]

# 8x8 checker of red(255,60,0)/green(30,200,90)
red = sum(1 for y in range(4, h, 16) for x in range(4, w, 16)
          if P(x, y)[0] > 200)
grn = sum(1 for y in range(4, h, 16) for x in range(4, w, 16)
          if P(x, y)[1] > 150)
assert red > 40 and grn > 40, f"checker broken: red={red} grn={grn}"
print(f"OK: descriptor-buffer texture bound, red={red} grn={grn}")
