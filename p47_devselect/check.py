# SPDX-License-Identifier: CC0-1.0
import re, subprocess

o = subprocess.run(["./app"], check=True, capture_output=True,
                   text=True).stdout
assert "DISCRETE_GPU" in o and "INTEGRATED_GPU" in o, o
m = re.search(r"selected: (.*) — discrete GPU", o)
assert m, o  # dGPU must win on this machine

c = subprocess.run(["./app", "--cpu"], check=True, capture_output=True,
                   text=True).stdout
assert "software fallback" in c.split("selected:")[-1], c
print(f"OK: {m.group(1).strip()} picked; --cpu override picks llvmpipe")
