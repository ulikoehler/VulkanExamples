#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — run the app under Xvfb (or verify an existing shot.ppm)
# and assert the captured frame contains the triangle pixels:
#   - blue-ish apex region (top vertex = blue... actually vertex 2),
#   - red left / green right at the bottom edge,
#   - background elsewhere.
import subprocess
import sys


def load_ppm(path):
    d = open(path, "rb").read()
    hdr, data = d.split(b"255\n", 1)
    parts = hdr.split()
    assert parts[0] == b"P6"
    w, h = int(parts[1]), int(parts[2])
    assert len(data) == w * h * 3
    return w, h, data


def px(data, w, x, y):
    i = (y * w + x) * 3
    return data[i], data[i + 1], data[i + 2]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "shot.ppm"
    if len(sys.argv) < 2:
        subprocess.run(["xvfb-run", "-a", "./app"], check=True,
                       timeout=180)
    w, h, data = load_ppm(path)
    print(f"size {w}x{h}")
    BG = (13, 13, 20)

    def near(a, b, tol=40):
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    # background top-left
    assert near(px(data, w, 10, 10), BG), px(data, w, 10, 10)
    # apex: vertex at (0, -0.8) NDC is BLUE — in Vulkan's y-down
    # viewport NDC -1 maps to TOP? No: with a positive-height
    # viewport, NDC y=+1 is the BOTTOM. So -0.8 -> near top.
    apex = px(data, w, w // 2, int(h * 0.12))
    assert apex[2] > 150 and apex[0] < 120, f"apex {apex}"
    # bottom-left region near (-0.8, 0.7): red vertex
    bl = px(data, w, int(w * 0.20), int(h * 0.82))
    assert bl[0] > 150, f"bottom-left {bl}"
    # bottom-right near (0.8, 0.7): green vertex
    br = px(data, w, int(w * 0.80), int(h * 0.82))
    assert br[1] > 150, f"bottom-right {br}"
    print("OK: swapchain screenshot contains the rendered frame")


if __name__ == "__main__":
    main()
