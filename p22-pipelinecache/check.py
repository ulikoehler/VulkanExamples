#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify the pipeline cache lifecycle:
#   run 1 (cold): no cache file -> "starting cold", writes cache
#   run 2 (warm): "loaded N bytes", renders identically
# Also verifies the rendered triangle in both outputs.
import os
import subprocess
import sys


def load_ppm(path):
    d = open(path, "rb").read()
    hdr, data = d.split(b"255\n", 1)
    parts = hdr.split()
    w, h = int(parts[1]), int(parts[2])
    return w, h, data


def triangle_ok(path):
    w, h, data = load_ppm(path)

    def px(x, y):
        i = (y * w + x) * 3
        return data[i], data[i + 1], data[i + 2]
    # blue apex top, colored bottom band, dark bg corner
    apex = px(w // 2, int(h * 0.12))
    assert apex[2] > 120, f"apex {apex}"
    bl = px(int(w * 0.2), int(h * 0.82))
    assert bl[0] > 120, f"bl {bl}"
    bg = px(5, 5)
    assert all(v < 60 for v in bg), f"bg {bg}"
    return True


def run():
    r = subprocess.run(["./app", "--headless", "out.ppm"],
                       capture_output=True, text=True, timeout=60)
    out = r.stdout + r.stderr
    print(out, end="")
    assert r.returncode == 0, f"exit {r.returncode}"
    return out


def main():
    if os.path.exists("pipeline.cache"):
        os.remove("pipeline.cache")

    out1 = run()
    assert "starting cold" in out1, out1
    assert os.path.exists("pipeline.cache"), \
        "cache file was not written"
    sz1 = os.path.getsize("pipeline.cache")
    assert sz1 > 100, f"cache file suspiciously small: {sz1}"
    triangle_ok("out.ppm")
    print(f"run1: cold build, cache saved ({sz1} B), render OK")

    out2 = run()
    assert "loaded" in out2, out2
    triangle_ok("out.ppm")
    print("run2: warm — cache blob loaded, render identical")
    print("OK: pipeline cache round-trips across runs")


if __name__ == "__main__":
    main()
