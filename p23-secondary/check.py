#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify all 3 secondary command buffers executed:
# three colored quads (red/green/blue) side by side on dark bg.
import sys


def load_ppm(path):
    d = open(path, "rb").read()
    hdr, data = d.split(b"255\n", 1)
    parts = hdr.split()
    w, h = int(parts[1]), int(parts[2])
    assert len(data) == w * h * 3
    return w, h, data


def px(data, w, x, y):
    i = (y * w + x) * 3
    return data[i], data[i + 1], data[i + 2]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out.ppm"
    w, h, data = load_ppm(path)
    assert (w, h) == (384, 384)
    BG = (13, 13, 20)

    def near(a, b, tol=40):
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    y = h // 2
    # quads at NDC x = -0.6, 0, +0.6 -> px = (1+x)*w/2
    left = px(data, w, int(0.2 * w), y)
    mid = px(data, w, w // 2, y)
    right = px(data, w, int(0.8 * w), y)
    print("left/mid/right:", left, mid, right)
    assert left[0] > 150 and left[1] < 120, f"left {left}"
    assert mid[1] > 150 and mid[0] < 120, f"mid {mid}"
    assert right[2] > 150 and right[0] < 120, f"right {right}"
    # background above the quads
    assert near(px(data, w, w // 2, 20), BG), \
        px(data, w, w // 2, 20)
    print("OK: all 3 secondary buffers replayed inside one rendering")


if __name__ == "__main__":
    main()
