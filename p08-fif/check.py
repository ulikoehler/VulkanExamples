#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify post 8 produced 4 deterministic frames.
# Frame f rotates the quad by f*30deg; each corner keeps its color:
#   TL red, TR green, BR blue, BL yellow (NDC +Y down, -0.5 = upper).
# Corner radius = sqrt(0.5^2+0.5^2) ~ 0.707 -> always inside the NDC
# box, so every rotated corner lands on screen.
import math
import sys

W, H = 640, 480
CORNERS = [
    ("red", (-0.5, -0.5), lambda p: p[0] > 120 and p[1] < 80),
    ("green", (0.5, -0.5), lambda p: p[1] > 120 and p[0] < 80),
    ("blue", (0.5, 0.5), lambda p: p[2] > 120 and p[0] < 80),
    ("yellow", (-0.5, 0.5),
     lambda p: p[0] > 120 and p[1] > 120 and p[2] < 80),
]


def read_ppm(path):
    d = open(path, "rb").read()
    hdr, data = d.split(b"255\n", 1)
    toks = hdr.split()
    assert toks[0] == b"P6"
    w, h = int(toks[1]), int(toks[2])
    assert len(data) == w * h * 3
    return w, h, data


def px(d, w, x, y):
    i = (y * w + x) * 3
    return tuple(d[i:i + 3])


def region_has(d, w, h, cx, cy, r, pred):
    for y in range(max(0, cy - r), min(h, cy + r)):
        for x in range(max(0, cx - r), min(w, cx + r)):
            if pred(px(d, w, x, y)):
                return True
    return False


def check_frame(path, frame):
    w, h, d = read_ppm(path)
    theta = math.radians(frame * 30.0)
    c, s = math.cos(theta), math.sin(theta)
    for name, (x0, y0), pred in CORNERS:
        # 2x2 rotation: x' = x*cos - y*sin ; y' = x*sin + y*cos
        xr = x0 * c - y0 * s
        yr = x0 * s + y0 * c
        px_ = int((xr + 1) / 2 * w)
        py_ = int((yr + 1) / 2 * h)
        # corners are rasterized slightly inward of the exact
        # projected vertex — search a 25px box
        assert region_has(d, w, h, px_, py_, 25, pred), \
            f"{path}: {name} corner expected near ({px_},{py_})"
    return True


def main():
    for f in range(4):
        check_frame(f"out{f}.ppm", f)
    print("OK: 4 frames, each rotated a further 30deg with correct "
          "corner colors")


if __name__ == "__main__":
    main()
