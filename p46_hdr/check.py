# SPDX-License-Identifier: CC0-1.0
import re, subprocess

o = subprocess.run(["xvfb-run", "-a", "./app", "--windowed"],
                   check=True, capture_output=True, text=True).stdout
assert "picked:" in o and "presented on" in o, o
# HDR was either picked and presented, or cleanly fell back to SDR
m = re.search(r"presented on (\S+) swapchain", o)
assert m.group(1) in ("HDR10", "scRGB", "SDR"), o
# the enumeration must list at least SDR
assert "SRGB_NONLINEAR" in o, o
print(f"OK: surface pairs enumerated, {m.group(1)} swapchain live")
