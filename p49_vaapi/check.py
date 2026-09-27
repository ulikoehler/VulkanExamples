# SPDX-License-Identifier: CC0-1.0
import re, subprocess, sys

o = subprocess.run(["./app", "--headless", "out.ppm"], check=True,
                   capture_output=True, text=True).stdout
if "SKIP" in o:
    print(o.strip()); sys.exit(0)

assert re.search(r"decoded \d+ frames, last is \d+x\d+ fmt=vaapi", o), o
assert re.search(r"drm desc: \d+ layers, \d+ objects", o), o
assert re.search(r"imported \d+ planes as VkImages", o), o

d = open("out.ppm", "rb").read()
m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(\d+)\s", d)
w, h = int(m.group(1)), int(m.group(2))
px = d[m.end():m.end() + w * h * 3]

# testsrc2 has saturated color bars — a zero-copy import that maps the
# wrong memory or misreads the tile layout yields far fewer colors.
colors = {px[i:i+3] for i in range(0, len(px), 9973)}
assert len(colors) > 20, f"only {len(colors)} distinct colors — wrong import"
print(f"OK: VA-API decoded, imported zero-copy, {len(colors)} distinct "
      "sampled colors")
