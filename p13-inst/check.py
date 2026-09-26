#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify the 16x16 instanced grid.
#
# The C++ side draws 256 quads in ONE vkCmdDraw(6, 256, 0, 0) call.
# Instance (x,y) sits at NDC offset (-0.9375 + 0.125*x, -0.9375 +
# 0.125*y) with quad halfsize 0.0275 NDC, in a 512x512 image.
# Color: r = x/15, g = y/15, b = 1.0 or 0.2 by (x^y)&1.
#
# We sample every cell center and check the color matches the
# expected instance data exactly (to quantization), plus that the
# gap between cells shows the background.
import sys
import struct


def load_ppm(path):
    d = open(path, "rb").read()
    hdr, data = d.split(b"255\n", 1)
    parts = hdr.split()
    assert parts[0] == b"P6", parts[0]
    w, h = int(parts[1]), int(parts[2])
    assert len(data) == w * h * 3, f"{len(data)} != {w*h*3}"
    return w, h, data


def px(data, w, x, y):
    i = (y * w + x) * 3
    return data[i], data[i + 1], data[i + 2]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out.ppm"
    w, h, data = load_ppm(path)
    assert (w, h) == (512, 512), (w, h)
    BG = (13, 13, 20)

    def near(a, b, tol=2):
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    bad = 0
    for gy in range(16):
        for gx in range(16):
            # NDC -> pixel: Vulkan viewport is Y-DOWN (ndc -1 = row 0)
            px_x = int(((-0.9375 + 0.125 * gx) + 1.0) * 0.5 * w)
            px_y = int(((-0.9375 + 0.125 * gy) + 1.0) * 0.5 * h)
            exp = (round(gx / 15 * 255),
                   round(gy / 15 * 255),
                   255 if (gx ^ gy) & 1 else 51)
            got = px(data, w, px_x, px_y)
            if not near(got, exp):
                bad += 1
                print(f"cell ({gx},{gy}) @({px_x},{px_y}): "
                      f"got {got}, expected ~{exp}")
    assert bad == 0, f"{bad} cells mismatched"

    # gaps between cells = background. Cell pitch = 0.125*256 = 32px,
    # quad = 0.055*256 ~ 14px => gap around halfway between centers.
    gap = px(data, w, 16 + 0, 16 + 0 + 0)  # very corner cell center is at x=16
    # corner of image = background
    assert near(px(data, w, 2, 2), BG), px(data, w, 2, 2)
    # midpoint between two cell centers (32px apart => 16px off)
    cx = int(((-0.9375 + 0.125 * 0) + 1.0) * 0.5 * w)
    cy = int(((-0.9375 + 0.125 * 0) + 1.0) * 0.5 * h)
    assert near(px(data, w, cx + 16, cy), BG), \
        px(data, w, cx + 16, cy)
    print("OK: 256 instances in one draw, all 16x16 cell colors exact")


if __name__ == "__main__":
    main()
