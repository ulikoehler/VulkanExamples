#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
# check.py — run the app under Xvfb (or verify an existing shot.png),
# decode the PNG in pure python (zlib + unfilter) and assert the
# captured frame contains the triangle pixels.
import struct
import subprocess
import sys
import zlib


def load_png(path):
    """Minimal PNG decoder: 8-bit RGB/RGBA, non-interlaced."""
    d = open(path, "rb").read()
    assert d[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos = 8
    idat = b""
    w = h = ct = bd = None
    while pos < len(d):
        ln, typ = struct.unpack(">I4s", d[pos:pos + 8])
        chunk = d[pos + 8:pos + 8 + ln]
        pos += 12 + ln
        if typ == b"IHDR":
            w, h, bd, ct, _cm, _fm, il = struct.unpack(">IIBBBBB",
                                                       chunk)
            assert bd == 8 and il == 0, (bd, il)
        elif typ == b"IDAT":
            idat += chunk
        elif typ == b"IEND":
            break
    raw = zlib.decompress(idat)
    ch = 3 if ct == 2 else 4 if ct == 6 else None
    assert ch, f"color type {ct}"
    stride = w * ch
    out = bytearray(h * stride)
    prev = bytearray(stride)
    i = 0
    for y in range(h):
        f = raw[i]
        i += 1
        row = bytearray(raw[i:i + stride])
        i += stride
        for x in range(stride):
            a = row[x - ch] if x >= ch else 0
            b = prev[x]
            c = prev[x - ch] if x >= ch else 0
            if f == 1:
                row[x] = (row[x] + a) & 0xFF
            elif f == 2:
                row[x] = (row[x] + b) & 0xFF
            elif f == 3:
                row[x] = (row[x] + (a + b) // 2) & 0xFF
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if pa <= pb and pa <= pc else \
                    b if pb <= pc else c
                row[x] = (row[x] + pr) & 0xFF
        out[y * stride:(y + 1) * stride] = row
        prev = row
    # strip alpha if present
    if ch == 4:
        rgb = bytearray(w * h * 3)
        for p in range(w * h):
            rgb[p * 3:p * 3 + 3] = out[p * 4:p * 4 + 3]
        return w, h, bytes(rgb)
    return w, h, bytes(out)


def px(data, w, x, y):
    i = (y * w + x) * 3
    return data[i], data[i + 1], data[i + 2]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "shot.png"
    if len(sys.argv) < 2:
        subprocess.run(["xvfb-run", "-a", "./app"], check=True,
                       timeout=180)
    w, h, data = load_png(path)
    print(f"decoded PNG {w}x{h}")
    BG = (13, 13, 20)

    def near(a, b, tol=40):
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    assert near(px(data, w, 10, 10), BG), px(data, w, 10, 10)
    apex = px(data, w, w // 2, int(h * 0.12))
    assert apex[2] > 150 and apex[0] < 120, f"apex {apex}"
    bl = px(data, w, int(w * 0.20), int(h * 0.82))
    assert bl[0] > 150, f"bottom-left {bl}"
    br = px(data, w, int(w * 0.80), int(h * 0.82))
    assert br[1] > 150, f"bottom-right {br}"
    print("OK: PNG screenshot decoded + contains the rendered frame")


if __name__ == "__main__":
    main()
