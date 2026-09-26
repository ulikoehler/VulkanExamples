# SPDX-License-Identifier: CC0-1.0
import re, subprocess

o = subprocess.run(["./app"], check=True, capture_output=True,
                   text=True).stdout

def parse(text):
    return {i: (int(u), int(b)) for i, u, b in re.findall(
        r"heap(\d+) size=\d+MB usage=(\d+)MB budget=(\d+)MB "
        r"flags=0x1", text)}

before, _, after = o.partition("--- after 3x256MB device-local ---")
heaps_b = parse(before)
heaps_a = parse(after)
assert heaps_b and heaps_a, o
# device-local heap(s): usage grew by ~768MB
grew = [i for i in heaps_a
        if heaps_a[i][0] - heaps_b.get(i, (0, 0))[0] >= 700]
assert grew, f"no heap grew by ~768MB: {o}"
print("OK: heap budget/usage reported, device-local usage +768MB")
