#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify the mip chain holds correct downsampled data.
#
# Level 0 of the texture is four solid quadrants:
#   TL=red  TR=blue  BL=green  BR=yellow
# (check uv->pixel mapping in main.cpp: left->r, top->no g,
#  bottom-left->g, bottom-right->r+b=yellow? verify below)
#
# Two quads sample the same texture:
#   LEFT  quad -> textureLod(tex, uv, 4): 16x16 mip — quadrants must
#          still be distinguishable
#   RIGHT quad -> textureLod(tex, uv, 8): 1x1 mip — ONE uniform
#          average color
#
# If the blit chain were broken, lod 8 would show level 0 data or
# garbage — not the average.
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
    assert (w, h) == (512, 256)

    def near(a, b, tol=30):
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    # ---- left quad (lod 4): quadrant colors distinguishable ------
    # left quad center (133,128); NDC halfsize 0.45 -> x +/-115px,
    # y +/-57px (the viewport is 2:1, quads are wide)
    tl = px(data, w, 70, 90)
    bl = px(data, w, 70, 165)
    tr = px(data, w, 200, 90)
    br = px(data, w, 200, 165)
    print("lod4 quadrants:", tl, tr, bl, br)
    # all four must be DIFFERENT from each other (roughly)
    def dist(a, b):
        return max(abs(x - y) for x, y in zip(a, b))
    assert dist(tl, bl) > 80, "lod4 top/bottom-left identical"
    assert dist(tr, br) > 80, "lod4 top/bottom-right identical"
    assert dist(tl, tr) > 80, "lod4 left/right-top identical"

    # ---- right quad (lod 8): single uniform color -----------------
    samples = [px(data, w, x, y)
               for x in (330, 384, 440)
               for y in (90, 128, 165)]
    print("lod8 samples:", samples)
    for s in samples[1:]:
        assert near(s, samples[0], 10), \
            f"lod8 not uniform: {samples[0]} vs {s}"
    # and it must be the average of the 4 quadrants:
    # avg of (255,0,0),(0,0,255),(0,255,0),(255,255,0) = (128,128,64)
    avg = samples[0]
    assert near(avg, (128, 128, 64), 25), \
        f"lod8 color {avg} != quadrant average ~(128,128,64)"

    print("OK: mip chain verified — lod4 quadrants, lod8 = average")


if __name__ == "__main__":
    main()
