#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — drive the resize demo under Xvfb and verify:
#   1. the windowed run exits 0
#   2. the swapchain was actually recreated at the new extent
#   3. the log reports the expected new size (800x600 framebuffer,
#      or a HiDPI multiple of it)
import re
import subprocess
import sys


def main():
    exe = sys.argv[1] if len(sys.argv) > 1 else "./app"
    r = subprocess.run(["xvfb-run", "-a", exe],
                       capture_output=True, text=True, timeout=180)
    out = r.stdout + r.stderr
    print(out, end="")
    assert r.returncode == 0, f"exit {r.returncode}"

    assert "[demo] resized window to 800x600" in out, \
        "resize was never triggered"
    m = re.findall(r"\[swapchain\] recreated: (\d+)x(\d+)", out)
    assert m, "no swapchain recreation happened"
    for w, h in m:
        # X11 framebuffer == window size here (no HiDPI under Xvfb)
        assert (int(w), int(h)) == (800, 600), f"bad extent {w}x{h}"
    print(f"OK: clean exit after {len(m)} recreation(s) to 800x600")


if __name__ == "__main__":
    main()
