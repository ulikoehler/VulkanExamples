#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — run the validation-layer demo and verify:
#   1. the app completes (exit 0)
#   2. the validation layer actually CAUGHT the deliberate
#      fillBuffer-without-TRANSFER_DST error
#   3. the error message mentions our named object "demo-buffer"
import os
import re
import subprocess
import sys


def main():
    exe = sys.argv[1] if len(sys.argv) > 1 else "./app"
    env = dict(os.environ)
    # if the validation layer was unpacked locally (vl-local/) use it;
    # otherwise rely on a system install (vulkan-validationlayers)
    here = os.path.dirname(os.path.abspath(__file__))
    layerdir = os.path.join(here,
        "vl-local/usr/share/vulkan/explicit_layer.d")
    libdir = os.path.join(here, "vl-local/usr/lib/x86_64-linux-gnu")
    if os.path.isdir(layerdir):
        env["VK_ADD_LAYER_PATH"] = layerdir
        env["LD_LIBRARY_PATH"] = \
            libdir + ":" + env.get("LD_LIBRARY_PATH", "")
    r = subprocess.run([exe], capture_output=True, text=True,
                       timeout=60, env=env)
    out = r.stdout + r.stderr
    print(out, end="")
    assert r.returncode == 0, f"exit {r.returncode}"

    # the layer must have produced an Error-severity message
    assert "[validation:Error]" in out, \
        "no validation error reported — is the layer installed?"
    # VUID for fillBuffer usage: CmdFillBuffer-apiVersion-07830
    # (wording varies; check for usage-flag complaint + VUID)
    assert "VUID" in out, "no VUID in validation output"
    assert "TRANSFER" in out.upper(), "expected usage-flag complaint"
    # object naming must have wired through
    assert "demo-buffer" in out, "named object not in message"
    print("OK: validation layer caught the deliberate error, "
          "with VUID + object name")


if __name__ == "__main__":
    main()
