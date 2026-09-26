#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify the compute blur ran: a 5x5 box blur of an 8x8
# black/white checkerboard softens edges — boundary pixels become
# mid-gray instead of pure 0.05/0.9 levels.
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
    return d[i]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out.ppm"
    w, h, d = read_ppm(path)
    assert (w, h) == (128, 128)

    vals = [px(d, w, x, y) for y in range(h) for x in range(w)]
    lo, hi = min(vals), max(vals)
    mids = sum(1 for v in vals if 60 < v < 180)
    # unblurred checker = only ~13 and ~230; blur adds mid-tones
    print(f"range: {lo}..{hi}, mid-tone pixels: {mids}")
    assert lo < 30 and hi > 200, "checkerboard levels missing"
    assert mids > 500, f"too few blurred mid-tones: {mids}"
    print("OK: compute blur produced softened checkerboard edges")


if __name__ == "__main__":
    main()
