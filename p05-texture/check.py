#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify post 5's textured fullscreen render.
# The 256x256 texture (8x8 checkerboard, 32px cells) is sampled with
# NEAREST onto the whole 640x480 target -> each texture cell covers
# an 80x60 px block. Cell (i,j) is color A if (i+j) even else B.
import sys

A = (230, 40, 60)
B = (40, 230, 60)


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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out.ppm"
    w, h, d = read_ppm(path)
    assert (w, h) == (640, 480)

    # sample the centre of every screen cell (cells are 80x60 px)
    for j in range(8):
        for i in range(8):
            x, y = i * 80 + 40, j * 60 + 30
            want = A if (i + j) % 2 == 0 else B
            got = px(d, w, x, y)
            assert got == want, \
                f"cell({i},{j}) px({x},{y})={got} want {want}"
    print("OK: 8x8 checkerboard sampled exactly")


if __name__ == "__main__":
    main()
