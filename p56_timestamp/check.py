# SPDX-License-Identifier: CC0-1.0
import re, subprocess, sys

o = subprocess.run(["./app", "--headless", "out.ppm"], check=True,
                   capture_output=True, text=True).stdout
if "SKIP" in o:
    print(o.strip()); sys.exit(0)
assert "timestampValidBits=64" in o or re.search(
    r"timestampValidBits=[1-9]", o), o
m = re.search(r"draw: ([\d.]+) us GPU", o)
assert m, o
us = float(m.group(1))
assert 1 < us < 500000, f"implausible GPU time {us}us"
print(f"OK: timestamp query measured {us}us GPU time")
