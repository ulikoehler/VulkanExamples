#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — run ./app and verify the capability report covers all
# queried fields with sane values for this device.
import re
import subprocess
import sys


def main():
    out = subprocess.run(["./app"], capture_output=True, text=True)
    print(out.stdout, end="")
    assert out.returncode == 0, "device does not meet requirements"
    o = out.stdout

    def num(name):
        m = re.search(rf"{name}: (\d+)", o)
        assert m, f"missing {name}"
        return int(m.group(1))

    assert re.search(r"apiVersion: 1\.\d+\.\d+", o)
    assert num("maxPushConstantsSize") >= 128, "spec minimum is 128"
    assert num("maxDrawIndirectCount") > 0
    assert num("maxPerStageDescriptorSampledImages") >= 16
    assert num("maxImageDimension2D") >= 4096
    assert re.search(r"pipelineCacheUUID: *[0-9a-f]{32}\n", o), \
        "UUID must be 16 bytes hex"
    assert re.search(r"synchronization2: 1", o)
    assert re.search(r"bufferDeviceAddress: 1", o)
    assert re.search(r"timestampPeriod: [0-9.]+ ns/tick", o)
    # at least one DEVICE_LOCAL and one HOST_VISIBLE memory type
    assert re.search(r"type +\d+: heap=\d+ flags=.*DeviceLocal", o)
    assert re.search(r"type +\d+: heap=\d+ flags=.*HostVisible", o)
    assert "series requirements: MET" in o
    print("OK: capability report complete, all assertions sane")


if __name__ == "__main__":
    main()
