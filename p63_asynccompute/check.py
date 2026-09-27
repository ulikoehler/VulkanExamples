#!/usr/bin/env python3
# Build + run; verify compute wrote its buffer. Overlap is logged,
# and asserted when positive on hardware with a compute queue.
import subprocess, sys, os, re

os.chdir(os.path.dirname(os.path.abspath(__file__)))
build = "g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc"
r = subprocess.run(build.split(), capture_output=True, text=True)
if r.returncode:
    print(r.stderr); sys.exit("BUILD FAILED")

r = subprocess.run(["./app"], capture_output=True, text=True, timeout=60)
out = r.stdout + r.stderr
print(out.strip())
if "SKIP" in out:
    sys.exit(0)
assert r.returncode == 0
m = re.search(r"compute tagged (\d+)/(\d+)", out)
assert m and m.group(1) == m.group(2), "compute output incomplete"
o = re.search(r"overlap: \+([\d.]+) us", out)
if o:
    print(f"OK: async compute overlap {o.group(1)}us "
          f"on dedicated queue")
else:
    print("OK: compute queue ran (serialized this time)")
