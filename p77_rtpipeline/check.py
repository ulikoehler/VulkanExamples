#!/usr/bin/env python3
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
r = subprocess.run("g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc".split(),
                   capture_output=True, text=True)
if r.returncode:
    print(r.stderr); sys.exit("BUILD FAILED")
o = subprocess.run(["./app", "--headless", "out.ppm"], check=True,
                   capture_output=True, text=True).stdout
assert "orange hit pixels" in o and "traceRayEXT" in o, o
import re
n = int(re.search(r"(\d+) orange hit", o).group(1))
assert n > 1000, o
print(f"OK: RT pipeline traced, {n} hit pixels")
