# SPDX-License-Identifier: CC0-1.0
import re, subprocess, sys

o = subprocess.run(["./app", "--headless", "out.ppm"], check=True,
                   capture_output=True, text=True).stdout
if "SKIP" in o:
    print(o.strip()); sys.exit(0)
assert "wrote out.ppm" in o, o

d = open("out.ppm", "rb").read()
m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(\d+)\s", d)
w, h = int(m.group(1)), int(m.group(2))
px = d[m.end():m.end() + w * h * 3]
def P(x, y):
    i = (y * w + x) * 3
    return px[i], px[i + 1], px[i + 2]

# gradient was (u,v,0.25); local-read draw wrote 1.0 - that.
# tl ~(0,0,64) -> ~ (255,255,191); br ~(255,255,64) -> ~(0,0,191)
tl, br = P(20, 20), P(w - 20, h - 20)
assert tl[0] > 200 and tl[1] > 200, f"top-left not inverted: {tl}"
assert br[0] < 60 and br[1] < 60,   f"bottom-right not inverted: {br}"
print(f"OK: in-pass feedback — tl={tl} br={br} (inverted gradient)")
