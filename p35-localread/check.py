#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify in-rendering local read: left half = green +
# 0.5*draw1-red, right half = pure green.
import sys


def read_ppm(path):
    d = open(path, "rb").read()
    hdr, data = d.split(b"255\n", 1)
    toks = hdr.split()
    w, h = int(toks[1]), int(toks[2])
    assert len(data) == w * h * 3
    return w, h, data


def px(d, w, x, y):
    i = (y * w + x) * 3
    return tuple(d[i:i + 3])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out.ppm"
    w, h, d = read_ppm(path)
    assert (w, h) == (128, 128)
    y = h // 2
    left = px(d, w, 32, y)     # inside draw1's red half
    right = px(d, w, 96, y)    # untouched by draw1
    print("left:", left, "right:", right)
    # right: 0.5*clear(0) + green 0.6 -> (0,153,0)
    assert right[0] < 60 and 100 < right[1] < 200 and right[2] < 60, \
        right
    # left: 0.5*(255,51,51) + (0,153,0) -> (~127,~180,~25)
    assert left[0] > 80 and left[0] > right[0] + 40, left
    assert left[1] > 100, left
    print("OK: subpassLoad read draw1's output inside the same "
          "rendering")


if __name__ == "__main__":
    main()
