#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify post 4's pushed transform.
# Push constants used by --headless: offset=(0.15,0.10), scale=0.60,
# tint=(0.9,0.7,0.9). Expected vertex positions (NDC +Y down):
#   red   (-0.33, 0.58) -> px (214, 379)
#   green ( 0.63, 0.58) -> px (522, 379)
#   blue  ( 0.15,-0.38) -> px (368, 149)
import sys


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


BG = (13, 13, 20)


def dominant(p, ch):
    others = [(p[(ch + 1) % 3], p[(ch + 2) % 3])]
    return p[ch] > 100 and p[ch] > max(others[0]) + 60


def region_has(d, w, h, cx, cy, r, pred):
    """True if any pixel in a r-box around (cx,cy) matches pred."""
    for y in range(max(0, cy - r), min(h, cy + r)):
        for x in range(max(0, cx - r), min(w, cx + r)):
            if pred(px(d, w, x, y)):
                return True
    return False


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out.ppm"
    w, h, d = read_ppm(path)
    assert (w, h) == (640, 480)

    # untransformed-triangle positions must now be background —
    # proves the push constants actually moved the geometry
    for x, y in [(320, 120), (85, 425), (555, 425)]:
        assert px(d, w, x, y) == BG, f"({x},{y}) old position not bg"

    # expected tinted vertex regions (30px tolerance box)
    assert region_has(d, w, h, 368, 149, 30,
                      lambda p: dominant(p, 2)), "no blue apex"
    assert region_has(d, w, h, 214, 379, 30,
                      lambda p: dominant(p, 0)), "no red corner"
    assert region_has(d, w, h, 522, 379, 30,
                      lambda p: dominant(p, 1)), "no green corner"

    # coverage of the scaled triangle: 0.6^2 * ~32% ~ 11-12%
    n = sum(1 for y in range(0, h, 4) for x in range(0, w, 4)
            if px(d, w, x, y) != BG)
    frac = n / ((w // 4) * (h // 4))
    assert 0.07 < frac < 0.17, f"coverage {frac:.2%} out of range"
    print(f"OK: shifted+scaled triangle, coverage {frac:.2%}")


if __name__ == "__main__":
    main()
