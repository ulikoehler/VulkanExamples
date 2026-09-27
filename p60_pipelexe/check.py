#!/usr/bin/env python3
# Build + run; verify executables, stats (VGPRs/SGPRs) and an
# assembly internal representation came back.
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
build = "g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc"
r = subprocess.run(build.split(), capture_output=True, text=True)
if r.returncode:
    print(r.stderr); sys.exit("BUILD FAILED")

r = subprocess.run(["./app"], capture_output=True, text=True,
                   timeout=60)
out = r.stdout + r.stderr
print(out[:3000])
if r.returncode:
    sys.exit(1)
need = ["executables", "Vertex", "Fragment", "VGPRs", "SGPRs",
        "Assembly", "OK"]
missing = [n for n in need if n not in out]
assert not missing, f"missing markers: {missing}"
print("OK: pipeline executables queried — stats + ISA present")
