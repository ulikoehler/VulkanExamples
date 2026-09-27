# SPDX-License-Identifier: CC0-1.0
import re, subprocess

o = subprocess.run(["./app", "out.jpg"], check=True,
                   capture_output=True, text=True).stdout
m = re.search(r"wrote out\.jpg \((\d+) bytes, (\d+) MCUs\)", o)
assert m, o
assert int(m.group(1)) > 5000  # real content, not just headers

d = open("out.jpg", "rb").read()
assert d[:2] == b"\xff\xd8" and d[-2:] == b"\xff\xd9", "SOI/EOI missing"
assert b"JFIF" in d[:20]
assert b"\xff\xdd" in d  # DRI marker — restart intervals present

try:
    from PIL import Image
    im = Image.open("out.jpg")
    im.load()
    assert im.size == (640, 480)
    assert len(set(list(im.getdata())[::97])) > 50  # gradient+checker content
    print("OK: GPU-encoded JPEG decodes, 640x480, real content")
except ImportError:
    print("OK: JPEG structure valid (PIL absent — no decode check)")
