#!/usr/bin/env python3
"""Bit-exact verification of GPU pixel-format conversion vs PIL.

All I/O goes through the app's `--raw WxHxC` mode (headerless bytes) —
no PNM/PAM header quirks in the loop. Random noise + gradient patterns
cover all 256 YCbCr LUT entries and every HSV sector.

    ./check.py            full corpus
    ./check.py --bench    + throughput comparison vs PIL
"""
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".check")

def ref(im, conv):
    """reference output for `conv` applied to PIL image `im`."""
    if conv == "rgb2rgba":
        return np.asarray(im.convert("RGBA"))
    if conv == "rgba2rgb":
        return np.asarray(im.convert("RGB"))
    if conv == "rgb2bgr":
        return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
    if conv == "bgr2rgb":
        return np.asarray(im)
    if conv in ("rgb2gray", "rgba2gray"):
        return np.asarray(im.convert("L"))
    if conv == "gray2rgb":
        return np.asarray(im.convert("RGB"))
    if conv == "gray2rgba":
        return np.asarray(im.convert("RGBA"))
    if conv == "rgb2ycbcr":
        return np.asarray(im.convert("YCbCr"))
    if conv == "ycbcr2rgb":
        return np.asarray(im.convert("RGB"))
    if conv == "rgb2hsv":
        return np.asarray(im.convert("HSV"))
    if conv == "hsv2rgb":
        return np.asarray(im.convert("RGB"))
    if conv == "gray2ycbcr":
        return np.asarray(im.convert("YCbCr"))
    if conv == "ycbcr2gray":
        return np.asarray(im.convert("L"))
    raise ValueError(conv)

# (conv, source mode)
CASES = [
    ("rgb2rgba",   "RGB"), ("rgba2rgb",  "RGBA"),
    ("rgb2bgr",    "RGB"), ("bgr2rgb",   "BGR"),
    ("rgb2gray",   "RGB"), ("gray2rgb",  "L"),
    ("rgba2gray",  "RGBA"), ("gray2rgba", "L"),
    ("rgb2ycbcr",  "RGB"), ("ycbcr2rgb", "YCbCr"),
    ("rgb2hsv",    "RGB"), ("hsv2rgb",   "HSV"),
    ("gray2ycbcr", "L"),   ("ycbcr2gray", "YCbCr"),
]

SIZES = [(640, 360), (333, 257), (1, 1), (16, 16), (2049, 3)]


def gen(W, H, mode, seed):
    """returns (im, src_bytes, ref_override)

    src_bytes: what to feed the app (defaults to im's pixels)
    ref_override: expected output when it isn't simply ref(im, conv)
    """
    rng = np.random.RandomState(seed)
    if mode == "L":
        g = (rng.rand(H, W) * 255).astype(np.uint8)
        yy, xx = np.mgrid[0:H, 0:W]
        g = np.where((xx + yy) % 7 == 0, (xx * 7 + yy * 3) % 256, g)
        return Image.fromarray(g.astype(np.uint8), "L"), None, None
    a = (rng.rand(H, W, 3) * 255).astype(np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    a[:, :, 0] = np.where((xx + yy) % 5 == 0, xx % 256, a[:, :, 0])
    a[:, :, 1] = np.where((xx * 3 + yy) % 11 == 0, yy % 256, a[:, :, 1])
    if mode == "RGBA":
        al = (rng.rand(H, W, 1) * 255).astype(np.uint8)
        return Image.fromarray(np.concatenate([a, al], 2), "RGBA"), None, None
    im = Image.fromarray(a, "RGB")
    if mode == "BGR":
        # PIL has no BGR mode — feed reversed channels, expect RGB back
        return im, np.ascontiguousarray(a[:, :, ::-1]), np.asarray(im)
    if mode != "RGB":
        im = im.convert(mode)
    return im, None, None


def main():
    bench = "--bench" in sys.argv
    os.makedirs(WORK, exist_ok=True)
    src = os.path.join(WORK, "src.raw")
    got = os.path.join(WORK, "got.raw")

    npass = nfail = 0
    for conv, mode in CASES:
        for i, (W, H) in enumerate(SIZES):
            im, src_override, ref_override = gen(W, H, mode,
                                                 seed=i * 31 + len(conv))
            data = (np.ascontiguousarray(np.asarray(im))
                    if src_override is None else src_override)
            data.tofile(src)
            r = subprocess.run(
                [APP, "--raw", f"{W}x{H}x{data.shape[2] if data.ndim == 3 else 1}", src, got, conv],
                capture_output=True)
            if r.returncode != 0:
                print(f"FAIL {conv} {W}x{H}: rc={r.returncode} "
                      f"{r.stderr.decode()[:100]}")
                nfail += 1
                continue
            b = np.asarray(ref_override if ref_override is not None
                           else ref(im, conv)).astype(int)
            a = np.fromfile(got, np.uint8).astype(int)
            if a.size != b.size:
                print(f"FAIL {conv} {W}x{H}: size {a.size} vs {b.size}")
                nfail += 1
                continue
            a = a.reshape(b.shape)
            d = np.abs(a - b)
            if d.max() == 0:
                npass += 1
            else:
                nfail += 1
                loc = np.unravel_index(d.argmax(), d.shape)
                print(f"FAIL {conv} {W}x{H}: max diff {d.max()} at {loc} "
                      f"({(d > 0).sum()} px)")
    print(f"{npass} pass, {nfail} fail "
          f"({'ALL BIT-EXACT' if nfail == 0 else 'FAILURES'})")

    if bench:
        W, H = 6000, 4000
        im, _, _ = gen(W, H, "RGB", 1)
        np.ascontiguousarray(np.asarray(im)).tofile(src)
        print(f"\nbench {W}x{H} (24 MP):")
        for conv, pilcall in [
            ("rgb2gray", lambda: im.convert("L")),
            ("rgb2ycbcr", lambda: im.convert("YCbCr")),
            ("rgb2hsv", lambda: im.convert("HSV")),
            ("rgb2rgba", lambda: im.convert("RGBA")),
        ]:
            r = subprocess.run(
                [APP, "--bench", "10", "--raw", f"{W}x{H}x3", src, got, conv],
                capture_output=True, text=True)
            t0 = time.time()
            for _ in range(10):
                pilcall()
            tp = (time.time() - t0) / 10 * 1000
            gpu = [l for l in r.stdout.splitlines() if "GB" in l]
            print(f"  {conv:11s} {gpu[0] if gpu else r.stderr.strip()[:60]}"
                  f"  | pil {tp:6.1f} ms/iter")

    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
