# SPDX-License-Identifier: CC0-1.0
import re, subprocess

o = subprocess.run(["./app", "--headless", "out.ppm"], check=True,
                   capture_output=True, text=True).stdout
m0 = re.search(r"query\[0\] occluded quad: (\d+) samples", o)
m1 = re.search(r"query\[1\] visible quad : (\d+) samples", o)
assert m0 and m1, o
assert int(m0.group(1)) == 0, o          # fully occluded -> zero
assert int(m1.group(1)) > 1000, o        # visible quad ~1521
print(f"OK: occlusion counts 0 / {m1.group(1)}")
