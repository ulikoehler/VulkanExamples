#!/usr/bin/env python3
"""Bit-exact verification of the Vulkan resizer vs PIL Image.resize().

Generates a corpus of (image, target size) pairs, runs ./app for every
filter x case, and diffs against PIL.  Any nonzero pixel diff fails.

    ./check.py            full corpus
    ./check.py --bench    + perf comparison vs PIL on a large image
"""
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".check")

FILTERS = {
    "nearest": Image.NEAREST,
    "box": Image.BOX,
    "bilinear": Image.BILINEAR,
    "hamming": Image.HAMMING,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}

# (name, W, H, out_w, out_h, mode)
CASES = [
    # downscale + upscale, both axes and single-axis
    ("d_320x240_100x77",   320, 240, 100,  77,  "RGB"),
    ("u_320x240_641x481",  320, 240, 641,  481, "RGB"),
    ("u_1x_100x77_321x77", 100, 77,  321,  77,  "RGB"),
    ("d_1x_321x77_100x77", 321, 77,  100,  77,  "RGB"),
    ("same_200x150",       200, 150, 200,  150, "RGB"),
    ("extreme_d_2048_16",  2048, 1536, 16,   12, "RGB"),
    ("extreme_u_16_2048",  16,   12,   2048, 1536, "RGB"),
    ("wide_2048x64_97x31", 2048, 64,  97,   31,  "RGB"),
    ("tall_64x2048_31x97", 64,  2048, 31,   97,  "RGB"),
    ("tiny_1x1_1x1",       1,    1,    1,    1,  "RGB"),
    ("tiny_1x1_7x5",       1,    1,    7,    5,  "RGB"),
    ("tiny_7x5_1x1",       7,    5,    1,    1,  "RGB"),
    ("odd_99x53_33x17",    99,   53,   33,   17, "RGB"),
    ("odd_101x73_300x201", 101,  73,   300,  201, "RGB"),
    ("gray_d_512_128",     512,  384,  128,  96, "L"),
    ("gray_u_128_512",     128,  96,   512,  384, "L"),
    ("gray_odd_97_55",     97,   55,   30,   42, "L"),
    ("rat_2x_640_320",     640,  480,  320,  320, "RGB"),  # aspect change
    ("rat_0.5_320_640",    320,  640,  640,  320, "RGB"),
    ("bigd_4000x3000_512", 4000, 3000, 512,  384, "RGB"),
]


def gen_image(W, H, mode, seed=7):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    chans = [
        np.clip(np.sin(xx / 13.0 + yy / 40.0) * 140 + 128 + rng.randn(H, W) * 20, 0, 255),
        np.clip(np.cos(yy / 17.0 - xx / 50.0) * 120 + 128 + rng.randn(H, W) * 20, 0, 255),
        ((xx * yy) // 31 + rng.randint(0, 40, (H, W))) % 256,
    ]
    if mode == "L":
        return Image.fromarray(chans[0].astype(np.uint8), "L")
    return Image.fromarray(np.stack(chans, 2).astype(np.uint8), "RGB")


def ppm_of(im):
    """bytes of PIL image as P6/P5 (same as our app's output)."""
    import io
    b = io.BytesIO()
    im.save(b, "PPM")
    return b.getvalue()


def main():
    bench = "--bench" in sys.argv
    os.makedirs(WORK, exist_ok=True)
    src = os.path.join(WORK, "src.ppm")
    got = os.path.join(WORK, "got.ppm")

    npass = nfail = 0
    for name, W, H, ow, oh, mode in CASES:
        im = gen_image(W, H, mode)
        im.save(src)
        for fname, resample in FILTERS.items():
            r = subprocess.run(
                [APP, src, got, f"{ow}x{oh}", fname], capture_output=True)
            if r.returncode != 0:
                print(f"FAIL {name} {fname}: app rc={r.returncode} "
                      f"{r.stderr.decode()[:120]}")
                nfail += 1
                continue
            ref = im.resize((ow, oh), resample)
            a = np.asarray(Image.open(got)).astype(int)
            b = np.asarray(ref).astype(int)
            d = np.abs(a - b)
            if d.max() == 0:
                npass += 1
            else:
                nfail += 1
                print(f"FAIL {name} {fname}: max diff {d.max()}"
                      f" at {np.unravel_index(d.argmax(), d.shape)}")
    print(f"{npass} pass, {nfail} fail "
          f"({'ALL BIT-EXACT vs PIL' if nfail == 0 else 'FAILURES'})")

    if bench:
        W, H, ow, oh = 6000, 4000, 1500, 1000
        im = gen_image(W, H, "RGB")
        im.save(src)
        print(f"\nbench {W}x{H} -> {ow}x{oh}:")
        for fname, resample in FILTERS.items():
            r = subprocess.run(
                [APP, "--bench", "20", src, got, f"{ow}x{oh}", fname],
                capture_output=True, text=True)
            gpu_ms = 0.0
            for ln in r.stderr.splitlines():
                if "per-iter" in ln:
                    gpu_ms = sum(float(x) for x in
                                 ln.replace("ms", "").split()
                                 if x.replace(".", "", 1).isdigit()
                                 and not x.startswith("per-iter"))
            t0 = time.time()
            for _ in range(20):
                im.resize((ow, oh), resample)
            tp = (time.time() - t0) / 20 * 1000
            print(f"  {fname:9s} gpu total {gpu_ms:6.1f} ms"
                  f" ({r.stderr.split('per-iter:')[-1].strip() if 'per-iter' in r.stderr else ''})"
                  f"  pil {tp:6.1f} ms")

    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
