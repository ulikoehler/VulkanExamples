#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — run the skeleton under Xvfb: must render ~90 frames,
# handle the self-triggered resize (swapchain recreation logged),
# and shut down cleanly.
import re
import subprocess
import sys


def main():
    out = subprocess.run(["xvfb-run", "-a", "./app"],
                         capture_output=True, text=True, timeout=60)
    print(out.stdout, end="")
    print(out.stderr, end="", file=sys.stderr)
    assert out.returncode == 0, f"exit {out.returncode}"
    o = out.stdout
    assert re.search(r"\[resize\] swapchain -> 800x500", o), \
        "resize did not trigger recreation"
    m = re.search(r"frames rendered: (\d+)", o)
    assert m and int(m.group(1)) >= 60, o
    assert "shutdown clean" in o
    print("OK: full skeleton loop — init, frames, resize, teardown")


if __name__ == "__main__":
    main()
