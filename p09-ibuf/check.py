#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify post 9's indexed quad.
# Quad corners in NDC (+Y down): TL(-0.6,-0.6) red, TR green,
# BR blue, BL yellow -> pixel corners at (128,96)/(512,384).
import sys

W, H = 640, 480


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


def region_has(d, cx, cy, r, pred):
    for y in range(cy - r, cy + r):
        for x in range(cx - r, cx + r):
            if pred(px(d, W, x, y)):
                return True
    return False


def main():
    w, h, d = read_ppm(sys.argv[1] if len(sys.argv) > 1 else "out.ppm")
    assert (w, h) == (W, H)
    # corner colors near the projected vertex positions
    assert region_has(d, 140, 108, 20,
                      lambda p: p[0] > 150 and p[1] < 90), "no red TL"
    assert region_has(d, 500, 108, 20,
                      lambda p: p[1] > 150 and p[0] < 90), "no green TR"
    assert region_has(d, 500, 372, 20,
                      lambda p: p[2] > 150 and p[0] < 90), "no blue BR"
    assert region_has(d, 140, 372, 20,
                      lambda p: p[0] > 150 and p[1] > 150), "no yellow"
    # outside the quad is background
    for x, y in [(10, 10), (630, 10), (10, 470), (630, 470),
                 (320, 20)]:
        assert px(d, W, x, y) == (13, 13, 20), f"({x},{y}) not bg"
    # quad centre sits on the shared diagonal: red<->blue blend
    p = px(d, W, 320, 240)
    assert p[0] > 80 and p[2] > 80 and p[1] < 100, f"centre {p}"
    print("OK: indexed quad, 4 verts / 6 indices, correct corners")


if __name__ == "__main__":
    main()
