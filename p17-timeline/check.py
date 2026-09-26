#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — run the timeline-semaphore demo and verify:
#   1. exit 0
#   2. all 4 slots written in order (the chained waits worked)
#   3. the semaphore counter reached 4
import re
import subprocess
import sys


def main():
    exe = sys.argv[1] if len(sys.argv) > 1 else "./app"
    r = subprocess.run([exe], capture_output=True, text=True,
                       timeout=60)
    out = r.stdout + r.stderr
    print(out, end="")
    assert r.returncode == 0, f"exit {r.returncode}"

    m = re.search(r"readback: ([0-9a-f]+) ([0-9a-f]+) ([0-9a-f]+) "
                  r"([0-9a-f]+)", out)
    assert m, "no readback line"
    vals = [int(g, 16) for g in m.groups()]
    assert vals == [0x11111111, 0x22222222, 0x33333333, 0x44444444], \
        f"slots {vals}"

    m2 = re.search(r"counter: (\d+)", out)
    assert m2 and int(m2.group(1)) == 4, "counter != 4"
    print("OK: 4 chained submits ordered + completed via "
          "timeline values")


if __name__ == "__main__":
    main()
