#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify post 6's compute-shader output.
# values.bin = 1024 floats of f(i)=i*i+7i-3, then 4 group sums.
import struct
import sys

N, GROUPS = 1024, 4


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "values.bin"
    d = open(path, "rb").read()
    vals = struct.unpack(f"<{N}f", d[: 4 * N])
    sums = struct.unpack(f"<{GROUPS}f", d[4 * N:])

    # verify every element — the whole point of compute readback
    for i, v in enumerate(vals):
        want = float(i * i + 7 * i - 3)
        assert v == want, f"out[{i}]={v} want {want}"

    # group sums: workgroup g summed its 256 lanes in shared memory.
    # The GPU accumulates in f32, so emulate f32 rounding per add —
    # comparing against a float64 exact sum fails on rounding alone.
    def f32(x):
        return struct.unpack("<f", struct.pack("<f", x))[0]

    for g in range(GROUPS):
        want = 0.0
        for i in range(g * 256, g * 256 + 256):
            want = f32(want + f32(i * i + 7 * i - 3))
        assert sums[g] == want, f"sums[{g}]={sums[g]} want {want}"

    print(f"OK: {N} elements + {GROUPS} group sums verified exactly")


if __name__ == "__main__":
    main()
