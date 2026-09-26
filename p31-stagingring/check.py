#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify the staging ring: 5 uploads through 2 slots
# (so 3 wraparound waits must happen), and the final sampled image
# is upload #4's pattern.
import re
import subprocess
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
    return tuple(d[i:i + 3])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out.ppm"
    out = subprocess.run(["./app", "--headless", path],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    print(out.stdout, end="")

    submits = re.findall(r"upload (\d+) -> slot (\d+) submitted",
                         out.stdout)
    waits = re.findall(r"slot (\d+) in flight, waiting", out.stdout)
    assert len(submits) == 5, submits
    # slot pattern: 0,1,0,1,0 -> 3 wraparound waits
    assert [int(s) for _, s in submits] == [0, 1, 0, 1, 0]
    assert len(waits) == 3, waits

    w, h, d = read_ppm(path)
    # final image = upload 4. RGBA bytes written as
    # (40+4*30, 40+4*30, 220) on even cells -> RGB (160,160,220);
    # odd cells (20,20,160). NEAREST fullscreen -> cell (0,0) covers
    # screen ~[0..48]px.
    e = px(d, w, 72, 21)            # even cell
    o = px(d, w, 21, 21)            # odd cell
    print("even/odd cell:", e, o)
    assert abs(e[0] - 160) < 40 and abs(e[2] - 220) < 40, e
    assert abs(o[0] - 20) < 40 and abs(o[2] - 160) < 40, o
    print("OK: 5 uploads streamed through 2-slot ring, final "
          "image = upload 4")


if __name__ == "__main__":
    main()
