#!/usr/bin/env python3
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
build = "g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc"
r = subprocess.run(build.split(), capture_output=True, text=True)
if r.returncode:
    print(r.stderr); sys.exit("BUILD FAILED")

r = subprocess.run(["./app"], capture_output=True, text=True, timeout=60)
print(r.stdout.strip())
assert r.returncode == 0 and "OK" in r.stdout, "subgroup mismatch"
print("OK: subgroup reduce/ballot/broadcast/shuffle verified")
