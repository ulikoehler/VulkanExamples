#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify the two-pass post-process.
#
# Pass 1 scene: green field (0.2,0.6,0.25) with a 4x4 red/blue
# checkerboard quad inside uv [0.15,0.85].
# Pass 2: fullscreen triangle samples it and writes 1 - color.
#
# So the OUTPUT should contain: magenta-ish field ((0.8,0.4,0.75)),
# inverted-red cells ((0.1,0.85,0.85)) and inverted-blue cells
# ((0.85,0.75,0.1)) in a checker arrangement.
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

    def near(a, b, tol=15):
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    # the scene quad covers NDC +-0.7 -> px [51,461]; inside it the
    # checker occupies uv[0.15..0.85] -> px [112,400]. The ring
    # between is green field: 1-(0.2,0.6,0.25) = (0.8,0.4,0.75)*255
    corner = px(data, w, 80, 128)
    assert near(corner, (204, 102, 191)), f"field {corner}"

    # 4x4 checker over px [112,400] -> each cell ~72px.
    # cell (0,0) center ~ px(148,148): scene blue -> inverted
    # (0.85,0.75,0.1)*255 = (217,191,26)
    c00 = px(data, w, 148, 148)
    assert near(c00, (217, 191, 26)), f"cell00 {c00}"
    # cell (1,0) center ~ px(220,148): scene red -> inverted
    # (0.1,0.85,0.85)*255 = (26,217,217)
    c10 = px(data, w, 220, 148)
    assert near(c10, (26, 217, 217)), f"cell10 {c10}"

    print("OK: two-pass pipeline — inverted checkerboard verified")


if __name__ == "__main__":
    main()
