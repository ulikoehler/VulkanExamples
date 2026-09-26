# SPDX-License-Identifier: CC0-1.0
import re, subprocess, os

o = subprocess.run(["./app", "one_red.h264", "out.ppm"],
                   capture_output=True, text=True).stdout
# The Annex-B split + SPS parse must always work (pure CPU):
assert "64x64" in o and "idr=1" in o, o
if "SKIP:" in o:
    # Machine has no video-decode queue — capability detection worked.
    print("OK: parsed stream; no decode queue -> clean skip")
else:
    # A device decoded the frame: verify the readback is red.
    assert "decoded" in o, o
    d = open("out.ppm", "rb").read()
    m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(\d+)\s", d)
    w, h = int(m.group(1)), int(m.group(2))
    px = d[m.end():m.end() + w * h * 3]
    def P(x, y):
        i = (y * w + x) * 3
        return px[i], px[i + 1], px[i + 2]
    for xy in [(8, 8), (32, 32), (56, 56), (32, 8)]:
        r, g, b = P(*xy)
        assert r > 200 and g < 60 and b < 60, (xy, (r, g, b))
    print("OK: decoded IDR frame reads back as red")
