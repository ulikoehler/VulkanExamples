#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify MSAA actually anti-aliased the triangle edges.
#
# A 512x512 image with one big RGB triangle on a dark background.
# With 4x MSAA resolved through eAverage, pixels along the triangle
# edges are weighted mixtures of triangle color and background —
# values that CANNOT appear without multisample resolve (no
# blending is enabled, so aliased rendering only produces pure
# vertex-interpolated colors or pure background).
import sys


def load_ppm(path):
    d = open(path, "rb").read()
    hdr, data = d.split(b"255\n", 1)
    parts = hdr.split()
    assert parts[0] == b"P6"
    w, h = int(parts[1]), int(parts[2])
    assert len(data) == w * h * 3
    return w, h, data


def px(data, w, x, y):
    i = (y * w + x) * 3
    return data[i], data[i + 1], data[i + 2]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out.ppm"
    w, h, data = load_ppm(path)
    assert (w, h) == (512, 512)

    BG = (13, 13, 20)

    def is_bg(p):
        return all(abs(a - b) <= 2 for a, b in zip(p, BG))

    # Scan a horizontal line through the slanted left edge of the
    # triangle (edge from (-0.8,0.7) to (0,-0.8) NDC). At y=192 the
    # edge crosses x~140. Collect runs of bg/edge/tri pixels.
    y = 192
    edge_found = 0
    for x in range(1, w - 1):
        p = px(data, w, x, y)
        if is_bg(p):
            continue
        # an AA edge pixel is neither bg nor a "pure" triangle color:
        # pure triangle colors have the same hue signature as vertex
        # interpolation (r+g+b ~ 255), edge pixels mixed with dark bg
        # have a much lower sum.
        s = sum(p)
        if s < 200:  # dark blend -> bg + color mix
            edge_found += 1
    assert edge_found > 0, "no AA edge pixels found — MSAA resolve " \
                           "did not happen (aliased edge would be " \
                           "a hard bg->color jump)"

    # Interior of the triangle (center) must be a bright mix ~sum 255
    cx, cy = 256, 200
    cp = px(data, w, cx, cy)
    assert sum(cp) > 180 and not is_bg(cp), f"center {cp}"

    # Far corner must be background
    assert is_bg(px(data, w, 5, 5)), px(data, w, 5, 5)

    print(f"OK: {edge_found} AA edge pixels on scanline y={y} — "
          "4x MSAA resolved correctly")


if __name__ == "__main__":
    main()
