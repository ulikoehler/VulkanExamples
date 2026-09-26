#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — verify the pipelined readback: all 6 frames verified
# from the 3-slot ring, and no MISMATCH lines appear.
import re
import subprocess
import sys


def main():
    out = subprocess.run(["./app"], capture_output=True, text=True)
    print(out.stdout, end="")
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert "MISMATCH" not in o
    reads = re.findall(r"read slot (\d+) <- frame (\d+)", o)
    assert len(reads) == 6, reads
    # frames are read in submission order, always 3 behind
    frames = [int(f) for _, f in reads]
    assert frames == [0, 1, 2, 3, 4, 5], frames
    m = re.search(r"done: 6/6 frames verified, (\d+) fence waits", o)
    assert m, o
    print("OK: 6 frames read back through 3-slot ring, all correct")


if __name__ == "__main__":
    main()
