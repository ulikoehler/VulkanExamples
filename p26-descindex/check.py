#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify one arrayed sampler binding served 4 different
# textures to 4 draws: 2x2 grid R/G/B/white on dark background.
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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out.ppm"
    w, h, data = load_ppm(path)
    assert (w, h) == (384, 384)
    BG = (13, 13, 20)

    def near(a, b, tol=40):
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    lo, hi = int(0.275 * w), int(0.725 * w)
    tlo, thi = int(0.275 * h), int(0.725 * h)
    q_tl = px(data, w, lo, tlo)   # tex[0] red
    q_tr = px(data, w, hi, tlo)   # tex[1] green
    q_bl = px(data, w, lo, thi)   # tex[2] blue
    q_br = px(data, w, hi, thi)   # tex[3] white
    print("quads:", q_tl, q_tr, q_bl, q_br)
    assert q_tl[0] > 200 and q_tl[1] < 120, f"tl {q_tl}"
    assert q_tr[1] > 200 and q_tr[0] < 120, f"tr {q_tr}"
    assert q_bl[2] > 200 and q_bl[0] < 120, f"bl {q_bl}"
    assert all(c > 200 for c in q_br), f"br {q_br}"
    assert near(px(data, w, w // 2, h // 2), BG), \
        px(data, w, w // 2, h // 2)
    print("OK: tex[i] indexed per draw from a single array binding")


if __name__ == "__main__":
    main()
