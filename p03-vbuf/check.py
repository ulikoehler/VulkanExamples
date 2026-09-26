#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify the headless render of post 3's triangle.
# Expected: dark background outside, RGB-graded triangle inside.
# Vertices: red=bottom-left, green=bottom-right, blue=top-center
# (Vulkan NDC has +Y down).
import sys


def read_ppm(path):
    d = open(path, "rb").read()
    # Our writer emits exactly "P6\n<W> <H>\n255\n" — split on \n only.
    # (Whitespace-tokenizing the header would eat leading pixel bytes
    # that happen to be \r/\n/space.)
    hdr, data = d.split(b"255\n", 1)
    toks = hdr.split()
    assert toks[0] == b"P6", "not a binary PPM"
    w, h = int(toks[1]), int(toks[2])
    assert len(data) == w * h * 3
    return w, h, data


def px(d, w, x, y):
    i = (y * w + x) * 3
    return tuple(d[i:i + 3])


BG = (13, 13, 20)  # clear color 0.05,0.05,0.08 in 8-bit


def near(c, target, tol=40):
    return all(abs(a - b) <= tol for a, b in zip(c, target))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out.ppm"
    w, h, d = read_ppm(path)
    assert (w, h) == (640, 480), f"unexpected size {w}x{h}"

    # corners must be the clear color
    for x, y in [(5, 5), (635, 5), (5, 475), (635, 475), (320, 30)]:
        assert px(d, w, x, y) == BG, f"({x},{y}) not background"

    # blue apex region (top-center)
    assert near(px(d, w, 320, 120), (0, 0, 255), 60), "apex not blue"
    # green bottom-right corner region
    assert near(px(d, w, 555, 425), (0, 255, 0), 60), "not green"
    # red-ish bottom-left corner region
    assert near(px(d, w, 85, 425), (255, 0, 0), 60), "not red"
    # bottom-centre is a red-green mix (no blue contribution yet)
    p = px(d, w, 320, 430)
    assert p[0] > 80 and p[1] > 80 and p[2] < 60, f"centre {p}"

    # triangle coverage sanity: count non-background pixels
    n = sum(
        1 for y in range(0, h, 4) for x in range(0, w, 4)
        if px(d, w, x, y) != BG)
    frac = n / ((w // 4) * (h // 4))
    assert 0.25 < frac < 0.45, f"triangle area {frac:.2f} out of range"
    print(f"OK: 640x480 RGB triangle, coverage {frac:.2%}")


if __name__ == "__main__":
    main()
