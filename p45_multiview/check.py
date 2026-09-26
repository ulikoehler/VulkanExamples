# SPDX-License-Identifier: CC0-1.0
import subprocess, sys

subprocess.run(["./app", "--headless", "out.ppm"], check=True)

def px(path, x, y):
    d = open(path, "rb").read().split(b"255\n", 1)[1]
    i = (y * 384 + x) * 3
    return d[i], d[i + 1], d[i + 2]

l0 = px("out.ppm", 192, 192)
l1 = px("out.ppm_l1.ppm", 192, 192)
assert l0 == (255, 51, 51), l0   # gl_ViewIndex 0 -> red
assert l1 == (51, 51, 255), l1   # gl_ViewIndex 1 -> blue
print("OK: one draw -> layer0 red, layer1 blue (gl_ViewIndex)")
