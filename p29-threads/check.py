#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify 4 thread-recorded secondaries rendered their
# strips (R/G/B/W left-to-right) AND the log proves the recording
# ran on worker threads.
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
    # all 4 workers recorded
    for i in range(4):
        assert f"[thread {i}] recorded secondary" in out.stdout

    w, h, d = read_ppm(path)
    assert (w, h) == (384, 384)
    y = h // 2
    # strips centered at NDC x = -0.75,-0.25,+0.25,+0.75
    # -> px x = (1+x)*192: 24, 144, 240, 336
    c0 = px(d, w, 48, y)
    c1 = px(d, w, 144, y)
    c2 = px(d, w, 240, y)
    c3 = px(d, w, 336, y)
    print("strips:", c0, c1, c2, c3)
    assert c0[0] > 150 and c0[1] < 120, c0
    assert c1[1] > 150 and c1[0] < 120, c1
    assert c2[2] > 150 and c2[0] < 120, c2
    assert all(c > 200 for c in c3), c3
    print("OK: 4 secondaries recorded on worker threads, replayed "
          "in one rendering")


if __name__ == "__main__":
    main()
