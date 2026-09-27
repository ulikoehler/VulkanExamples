#!/usr/bin/env python3
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
r = subprocess.run("g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc".split(),
                   capture_output=True, text=True)
if r.returncode:
    print(r.stderr); sys.exit("BUILD FAILED")
o = subprocess.run(["./app"], check=True, capture_output=True,
                   text=True).stdout
assert "signedAccelerated=1" in o and "bad=0" in o, o
print("OK: packed int8 dot products verified bit-exactly")
