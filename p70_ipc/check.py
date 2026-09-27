#!/usr/bin/env python3
# Build + run the two-process external-memory demo.
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
build = "g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc"
r = subprocess.run(build.split(), capture_output=True, text=True)
if r.returncode:
    print(r.stderr); sys.exit("BUILD FAILED")

o = subprocess.run(["./app"], check=True, capture_output=True,
                   text=True).stdout
if "SKIP" in o:
    print(o.strip()); sys.exit(0)
assert "1024/1024 words == 0xdeadbeef" in o, o
assert "shared across processes" in o, o
print("OK: GPU memory exported as fd, imported + verified by child")
