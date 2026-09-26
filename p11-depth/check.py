#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — prove the depth buffer worked.
# NEAR red quad drawn FIRST at x=[128..384], FAR blue quad drawn
# SECOND at x=[256..512]. Overlap x=[256..384] must stay RED —
# without depth testing it would show the blue drawn later.
import sys

W, H = 640, 480
BG = (13, 13, 20)


def read_ppm(path):
    d = open(path, "rb").read()
    hdr, data = d.split(b"255\n", 1)
    toks = hdr.split()
    assert toks[0] == b"P6"
    w, h = int(toks[1]), int(toks[2])
    assert len(data) == w * h * 3
    return w, h, data


def px(d, x, y):
    i = (y * W + x) * 3
    return tuple(d[i:i + 3])


def main():
    w, h, d = read_ppm(sys.argv[1] if len(sys.argv) > 1 else "out.ppm")
    assert (w, h) == (W, H)

    def is_red(p): return p[0] > 150 and p[1] < 100
    def is_blue(p): return p[2] > 150 and p[0] < 100

    # pure zones
    assert is_red(px(d, 160, 240)), "left quad missing (red)"
    assert is_blue(px(d, 460, 240)), "right quad missing (blue)"
    # THE depth assertion: overlap drawn by FAR quad LAST still shows
    # the NEAR quad's color
    for x in (280, 320, 360):
        assert is_red(px(d, x, 240)), \
            f"overlap at x={x} shows {px(d, x, 240)} not near-red"
    # outside everything: background
    for x in (60, 580):
        assert px(d, x, 240) == BG, f"x={x} not background"
    print("OK: near quad wins the overlap although drawn FIRST")


if __name__ == "__main__":
    main()
