#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify timestamp queries produced a sane, positive
# GPU-time measurement AND the frame rendered correctly.
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

    m = re.search(r"gpu time: ([0-9.]+) us \((\d+) ticks "
                  r"@ ([0-9.]+) ns\)", out.stdout)
    assert m, "no timing line"
    us, ticks = float(m.group(1)), int(m.group(2))
    assert ticks > 0, "timestamps identical — query never resolved?"
    assert 0.01 < us < 10_000, f"implausible GPU time: {us}us"
    assert "timestampPeriod:" in out.stdout

    # the draw must have run: center pixel != clear black
    w, h, d = read_ppm(path)
    p = px(d, w, w // 2, h // 2)
    assert p[1] > 50 and p[2] > 100, f"draw missing: {p}"
    print(f"OK: GPU timestamp diff = {us:.1f}us, draw rendered")


if __name__ == "__main__":
    main()
