#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify sync2 chain worked end to end: compute wrote the
# plasma (r=sinusoidal, g=u, b=v), the barrier2 made it visible to
# the fragment shader, and the fullscreen pass sampled it NEAREST.
import sys
import math


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
    assert (w, h) == (384, 384)

    checked = 0
    # screen px (x,y) samples texel floor((x+.5)*256/384)
    for y in range(20, h, 60):
        for x in range(20, w, 60):
            tx = int((x + 0.5) * 256 / 384)
            ty = int((y + 0.5) * 256 / 384)
            u, v = tx / 256.0, ty / 256.0
            r = int((0.5 + 0.5 * math.sin(u * 24.0)
                     * math.cos(v * 24.0)) * 255 + 0.5)
            want = (r, tx, ty)
            got = px(d, w, x, y)
            assert all(abs(a - b) <= 2 for a, b in zip(got, want)), \
                f"px({x},{y})={got} want {want}"
            checked += 1
    print(f"OK: {checked} pixels — compute->barrier2->sample chain "
          "exact")


if __name__ == "__main__":
    main()
