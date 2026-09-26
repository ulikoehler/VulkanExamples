# SPDX-License-Identifier: CC0-1.0
import re, subprocess

o = subprocess.run(["./app", "--headless", "out.ppm"],
                   capture_output=True, text=True).stdout
if "SKIP:" in o:
    print("OK: no host-image-copy device -> clean skip")
else:
    assert "host->image copy done" in o, o
    assert "roundtrip: exact" in o, o
    d = open("out.ppm", "rb").read()
    m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(\d+)\s", d)
    w, h = int(m.group(1)), int(m.group(2))
    px = d[m.end():m.end() + w * h * 3]
    cells = {tuple(px[(y * w + x) * 3:(y * w + x) * 3 + 3])
             for y in range(0, h, 16) for x in range(0, w, 16)}
    assert cells == {(0, 255, 0), (255, 0, 0)}, cells
    print("OK: R/G checkerboard via host image copy + readback")
