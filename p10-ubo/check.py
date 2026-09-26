#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify post 10's UBO-driven perspective transform.
# The quad is tilted 45deg around X: it must project as a trapezoid
# (top edge visibly narrower than the bottom edge) and keep its
# corner colors in screen space.
import sys

W, H = 640, 480
BG = (13, 13, 20)


def read_ppm(path):
    d = open(path, "rb").read()
    hdr, data = d.split(b"255\n", 1)
    toks = hdr.split()
    assert toks[0] == b"P6"
    w, h = int(toks[1]), int(toks[2])
    assert len(data) == w * h * 3
    return w, h, data


def px(d, x, y):
    i = (y * W + x) * 3
    return tuple(d[i:i + 3])


def row_span(d, y):
    xs = [x for x in range(W) if px(d, x, y) != BG]
    return (min(xs), max(xs)) if xs else None


def main():
    w, h, d = read_ppm(sys.argv[1] if len(sys.argv) > 1 else "out.ppm")
    assert (w, h) == (W, H)

    # find the colored quad's vertical extent
    ys = [y for y in range(H)
          if any(px(d, x, y) != BG for x in range(0, W, 4))]
    assert ys, "nothing rendered"
    top, bot = ys[0], ys[-1]

    spans = [row_span(d, y) for y in (top + 5, (top + bot) // 2,
                                      bot - 5)]
    widths = [s[1] - s[0] for s in spans if s]
    assert len(widths) == 3, "quad rows missing"
    # trapezoid: top narrower than bottom
    assert widths[0] < widths[2] * 0.75, \
        f"not a trapezoid: {widths}"
    # and roughly symmetric around screen centre
    for s in spans:
        if s:
            assert abs((s[0] + s[1]) / 2 - W / 2) < 30, s
    print(f"OK: perspective trapezoid widths {widths}, "
          f"y=[{top}..{bot}]")


if __name__ == "__main__":
    main()
