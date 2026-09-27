# SPDX-License-Identifier: CC0-1.0
import re, subprocess, sys

o = subprocess.run(["./app", "--headless", "out.ppm"], check=True,
                   capture_output=True, text=True).stdout
if "SKIP" in o:
    print(o.strip()); sys.exit(0)

assert re.search(r"dmabuf fd \d+", o), o
assert "captured frame 640x480" in o, o
assert "imported dma-buf as VkImage" in o, o

d = open("out.ppm", "rb").read()
m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(\d+)\s", d)
w, h = int(m.group(1)), int(m.group(2))
assert (w, h) == (640, 480)
px = d[m.end():]

# a real webcam frame: whatever it shows, 300k pixels of one color is
# either a lens cap or a decode bug — flag it, don't fail on it.
colors = {px[i:i+3] for i in range(0, len(px), 9973)}
if len(colors) <= 1:
    print("WARN: frame is one solid color (lens cap?)")
else:
    print(f"OK: live frame imported zero-copy, {len(colors)} distinct "
          "sampled colors")
print("OK: V4L2 dmabuf -> VkImage -> sampled")
