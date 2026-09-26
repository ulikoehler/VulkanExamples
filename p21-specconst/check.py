#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify specialization constants took effect.
#
# Same SPIR-V, two pipelines: left quad CELLS=4, right quad CELLS=8.
# The 512x256 image has quads centered at x=133 and x=379, each
# ~0.45 NDC half-width -> ~115px wide, ~57px tall around y=128.
#
# Left quad: 4 cells across ~230px -> ~57px cell pitch.
# Right quad: 8 cells -> ~28px pitch.
# We walk a horizontal line through each quad's center and count
# color transitions: CELLS-1 sign flips expected (checker parity).
import sys


def load_ppm(path):
    d = open(path, "rb").read()
    hdr, data = d.split(b"255\n", 1)
    parts = hdr.split()
    w, h = int(parts[1]), int(parts[2])
    assert len(data) == w * h * 3
    return w, h, data


def px(data, w, x, y):
    i = (y * w + x) * 3
    return data[i], data[i + 1], data[i + 2]


def transitions(data, w, y, x0, x1):
    """Count red<->blue flips along the scanline."""
    t = 0
    prev = None
    for x in range(x0, x1):
        p = px(data, w, x, y)
        # classify: red-dominant vs blue-dominant (ignore bg)
        if p[0] > p[2] and p[0] > 100:
            cls = "R"
        elif p[2] > p[0] and p[2] > 100:
            cls = "B"
        else:
            continue
        if prev and cls != prev:
            t += 1
        prev = cls
    return t


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out.ppm"
    w, h, data = load_ppm(path)
    assert (w, h) == (512, 256)

    y = h // 2
    left = transitions(data, w, y, 20, 248)
    right = transitions(data, w, y, 266, 494)
    print(f"transitions: left={left} (expect 4 cells -> 3-5), "
          f"right={right} (expect 8 cells -> 7-9)")
    assert 3 <= left <= 5, f"left {left}"
    assert 7 <= right <= 9, f"right {right}"
    print("OK: same SPIR-V, two spec-constant values, "
          "different checker densities")


if __name__ == "__main__":
    main()
