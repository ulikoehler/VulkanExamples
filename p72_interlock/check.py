#!/usr/bin/env python3
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
r = subprocess.run("g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc".split(),
                   capture_output=True, text=True)
if r.returncode:
    print(r.stderr); sys.exit("BUILD FAILED")
o = subprocess.run(["./app", "--headless", "out.ppm"], check=True,
                   capture_output=True, text=True).stdout
if "SKIP" in o:
    print(o.strip()); sys.exit(0)
assert "bad=0" in o and "inside interlock" in o, o
print("OK: interlocked non-atomic RMW produced exact products")
