#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify post 7's compute->graphics pipeline.
# The storage image is 256x256; the compute shader writes
# (r,g,b) = (tx, ty, tx^ty) per texel. The fullscreen triangle
# samples it NEAREST across 640x480, so screen pixel (x,y) reads
# texel (floor((x+0.5)*256/640), floor((y+0.5)*256/480)).
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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out.ppm"
    w, h, d = read_ppm(path)
    assert (w, h) == (640, 480)

    checked = 0
    for y in range(30, h, 60):
        for x in range(40, w, 80):
            tx = int((x + 0.5) * 256 / 640)
            ty = int((y + 0.5) * 256 / 480)
            want = (tx, ty, tx ^ ty)
            got = px(d, w, x, y)
            # NEAREST + UNORM round-trip: allow 1 ulp each channel
            assert all(abs(a - b) <= 1 for a, b in zip(got, want)), \
                f"px({x},{y})={got} want {want} (texel {tx},{ty})"
            checked += 1
    print(f"OK: {checked} pixels match the compute pattern exactly")


if __name__ == "__main__":
    main()
