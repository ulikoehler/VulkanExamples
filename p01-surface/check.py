#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# Semi-automated check for post 1.
#
#   ./app --headless out.ppm && python3 check.py out.ppm
#
# Asserts that the headless render produced a solid (51,102,204) blue
# 640x480 image — sampling a 4x4 grid of pixels plus every pixel of a
# centre row for uniformity.

import sys


def read_ppm(path):
    d = open(path, "rb").read()
    # Split on "255\n": whitespace-tokenizing the header would eat
    # leading pixel bytes that happen to be \r/\n/space.
    hdr, data = d.split(b"255\n", 1)
    toks = hdr.split()
    assert toks[0] == b"P6", "not a binary PPM"
    w, h = int(toks[1]), int(toks[2])
    assert len(data) == w * h * 3, "truncated pixel data"
    return w, h, data


def px(data, w, x, y):
    i = (y * w + x) * 3
    return tuple(data[i:i + 3])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out.ppm"
    w, h, d = read_ppm(path)
    assert (w, h) == (640, 480), f"unexpected size {w}x{h}"

    expected = (51, 102, 204)  # the blue cleared in main.cpp
    for gy in range(4):
        for gx in range(4):
            p = px(d, w, 80 + gx * 160, 60 + gy * 120)
            assert p == expected, f"pixel({gx},{gy})={p}"
    # full centre row must be uniform
    row = {px(d, w, x, h // 2) for x in range(w)}
    assert row == {expected}, f"centre row not uniform: {row}"
    print(f"OK: {w}x{h} uniform {expected}")


if __name__ == "__main__":
    main()
