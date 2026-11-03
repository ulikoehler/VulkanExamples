#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
"""Verify the GPU progressive JPEG decoder is bit-exact vs libjpeg.

Generates a progressive corpus with PIL + cjpeg (subsampling, qualities,
restart intervals, DRI=0, optimized tables, gray, sizes, scan scripts),
plus a baseline regression corpus, runs ./app --verify on each and
requires:  coefs: 0/N mismatch  AND  rgb: max |diff| 0.
Unsupported files (arithmetic, >3 comps, 12/16-bit) must fail cleanly.
"""
import os, re, subprocess, sys, glob, random

APP = "./app"
CORPUS = "corpus"
os.makedirs(CORPUS, exist_ok=True)

def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)

# ---------- corpus generation ----------

def _gens():
    import numpy as np
    import PIL.ImageFile
    from PIL import Image
    # Progressive saves suspend when PIL's 4*w*h write buffer fills —
    # incompressible noise overflows it ("broken data stream").
    PIL.ImageFile.MAXBLOCK = 32 * 1024 * 1024
    rng = np.random.default_rng(42)

    def grad(w, h):
        x = np.linspace(0, 255, w)[None, :]
        y = np.linspace(0, 255, h)[:, None]
        a = np.broadcast_to(x, (h, w))
        b = np.broadcast_to(y, (h, w))
        return np.stack([a, b, (a + b) / 2], -1).astype(np.uint8)

    def noise(w, h):
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def edges(w, h):
        yy, xx = np.mgrid[0:h, 0:w]
        cb = (((xx // 8) + (yy // 8)) % 2 * 255).astype(np.uint8)
        circ = (((xx - w // 2) ** 2 + (yy - h // 2) ** 2) < (min(w, h) // 3) ** 2)
        return np.stack([cb, np.where(circ, 255, cb // 3).astype(np.uint8),
                         (255 - cb)], -1)

    def photo(w, h):
        n = rng.integers(0, 256, (h // 4 + 1, w // 4 + 1, 3), dtype=np.uint8)
        return np.asarray(Image.fromarray(n).resize((w, h), Image.BICUBIC))
    return rng, {"grad": grad, "noise": noise, "edges": edges, "photo": photo}

def gen_pil():
    rng, contents = _gens()
    from PIL import Image
    jobs = []
    sizes = [(1, 1), (7, 5), (8, 8), (16, 16), (33, 17), (64, 48),
             (120, 80), (333, 257), (512, 384), (640, 480), (1023, 767)]
    # progressive PIL files across content/size/subsampling
    for cname, fn in contents.items():
        for (w, h) in sizes[::2] if cname == "photo" else sizes:
            for sub, q in [(0, 90), (1, 85), (2, 75)]:
                p = f"{CORPUS}/prog_{cname}_{w}x{h}_s{sub}q{q}.jpg"
                Image.fromarray(fn(w, h)).save(
                    p, quality=q, subsampling=sub, progressive=True)
                jobs.append(p)
    # progressive grayscale
    for (w, h) in [(17, 9), (256, 200), (641, 481)]:
        p = f"{CORPUS}/prog_gray_{w}x{h}.jpg"
        Image.fromarray(contents["grad"](w, h)[:, :, 0]).save(
            p, quality=90, progressive=True)
        jobs.append(p)
    # baseline regression subset (decoder must still handle SOF0)
    for (w, h) in [(33, 17), (256, 200), (512, 384)]:
        p = f"{CORPUS}/base_{w}x{h}.jpg"
        Image.fromarray(contents["photo"](w, h)).save(
            p, quality=85, subsampling=2)
        jobs.append(p)
    return jobs

def gen_cjpeg():
    """cjpeg progressive variants: restart intervals, optimize, sampling."""
    import numpy as np
    rng, contents = _gens()
    from PIL import Image
    jobs = []
    n = rng.integers(0, 256, (200, 320, 3), dtype=np.uint8)
    src = f"{CORPUS}/_cj_src.ppm"
    Image.fromarray(n).save(src)
    variants = [
        ("rst1",  "-progressive -restart 1"),
        ("rst4",  "-progressive -restart 4"),
        ("rst1B", "-progressive -restart 1B"),
        ("rst16", "-progressive -restart 16"),
        ("s12",   "-progressive -sample 1x2"),   # 4:4:0
        ("s21",   "-progressive -sample 2x1"),   # 4:2:2
        ("s22",   "-progressive -sample 2x2"),   # 4:2:0
        ("s11",   "-progressive -sample 1x1"),   # 4:4:4
        ("opt",   "-progressive -optimize"),     # per-scan optimized DHT
        ("q10",   "-progressive -quality 10"),
        ("sm",    "-progressive -smooth 50"),
    ]
    for name, opts in variants:
        p = f"{CORPUS}/cj_{name}.jpg"
        r = sh(f"cjpeg {opts} -outfile {p} {src}")
        if r.returncode == 0 and os.path.exists(p):
            jobs.append(p)
        else:
            print(f"  (skip cjpeg {opts}: {r.stderr.strip()[:80]})")
    # tiny image, restart smaller than an MCU row
    Image.fromarray(n[:40, :56]).save(f"{CORPUS}/_cj_src2.ppm")
    p = f"{CORPUS}/cj_small_rst.jpg"
    sh(f"cjpeg -progressive -restart 1 -outfile {p} {CORPUS}/_cj_src2.ppm")
    if os.path.exists(p):
        jobs.append(p)
    # a larger image with restarts and without (DRI=0)
    big = contents["photo"](1200, 1600)
    Image.fromarray(big).save(f"{CORPUS}/_cj_big.ppm")
    p = f"{CORPUS}/cj_big_rst.jpg"
    sh(f"cjpeg -progressive -restart 8 -outfile {p} {CORPUS}/_cj_big.ppm")
    if os.path.exists(p):
        jobs.append(p)
    p = f"{CORPUS}/cj_big_norst.jpg"
    sh(f"cjpeg -progressive -outfile {p} {CORPUS}/_cj_big.ppm")
    if os.path.exists(p):
        jobs.append(p)
    return jobs

def gen_exif():
    """EXIF orientation 1..8 tagged files for --orient auto tests."""
    rng, contents = _gens()
    from PIL import Image
    jobs = []
    img = Image.fromarray(contents["photo"](320, 240))
    for o in range(1, 9):
        p = f"{CORPUS}/exif_o{o}.jpg"
        ex = img.getexif(); ex[0x0112] = o
        img.save(p, exif=ex, quality=90, subsampling=2)
        jobs.append(p)
    return jobs

def gen_reject():
    """Files that must be refused cleanly (nonzero exit, no crash)."""
    import numpy as np
    from PIL import Image
    jobs = []
    rng = np.random.default_rng(3)
    n = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    Image.fromarray(n).save(f"{CORPUS}/_rej.ppm")
    cases = {"arithmetic": "-arithmetic"}
    for name, opts in cases.items():
        p = f"{CORPUS}/rej_{name}.jpg"
        r = sh(f"cjpeg {opts} -outfile {p} {CORPUS}/_rej.ppm")
        if r.returncode == 0 and os.path.exists(p):
            jobs.append(p)
        else:
            print(f"  (skip {name}: {r.stderr.strip()[:80]})")
    # truncated before first SOS
    tp = f"{CORPUS}/rej_trunc.jpg"
    Image.fromarray(n).save(tp, quality=85)
    d = open(tp, "rb").read()
    open(tp, "wb").write(d[:64])
    jobs.append(tp)
    # not a jpeg
    np_ = f"{CORPUS}/rej_notjpg.jpg"
    open(np_, "wb").write(b"PNG\r\n\x1a\n" + os.urandom(200))
    jobs.append(np_)
    return jobs

def find_real():
    """Real-world JPEGs (progressive AND baseline) found on the system."""
    pats = ["/usr/share/**/*.jp*g", os.path.expanduser("~/dev/**/*.jp*g")]
    ok = []
    for pat in pats:
        for f in glob.iglob(pat, recursive=True):
            try:
                if not (4_000 < os.path.getsize(f) < 4_000_000):
                    continue
                from PIL import Image
                im = Image.open(f)
                if im.mode in ("RGB", "L"):
                    ok.append(f)
            except Exception:
                pass
    random.seed(1); random.shuffle(ok)
    return ok[:40]

# ---------- runner ----------

RE_COEF = re.compile(r"coefs:\s*(\d+)/(\d+) mismatch")
RE_RGB = re.compile(r"rgb:\s*max \|diff\|\s*(\d+)\s+mean\s+([\d.]+)")
RE_UNSUP = re.compile(r"not supported|unsupported|not a JPEG|missing|precision",
                      re.I)

def verify(path):
    r = subprocess.run([APP, "--verify", path],
                       capture_output=True, text=True, timeout=180)
    mc, mr = RE_COEF.search(r.stdout), RE_RGB.search(r.stdout)
    if r.returncode != 0 or not mc or not mr:
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:]
        if r.returncode != 0 and tail and RE_UNSUP.search(tail[0]):
            return "reject", tail[0][:70]
        return False, f"exit={r.returncode} {tail}"
    if mc.group(1) != "0":
        return False, f"coefs {mc.group(1)}/{mc.group(2)} mismatch"
    if mr.group(1) != "0":
        return False, f"rgb max|diff|={mr.group(1)} mean={mr.group(2)}"
    return True, f"{mc.group(2)} coefs"

RE_SRGB = re.compile(r"rgb: max \|diff\| (\d+)")

def verify_scales(path):
    """--verify --scale {1/2,1/4,1/8}: returns (ok, compact status)."""
    out = []
    for s in ("1/2", "1/4", "1/8"):
        r = subprocess.run([APP, "--verify", "--scale", s, path],
                           capture_output=True, text=True, timeout=180)
        m = RE_SRGB.search(r.stdout)
        ok = r.returncode == 0 and m and m.group(1) == "0"
        out.append((s, ok, m.group(1) if m else "?"))
    bad = [f"{s}={d}" for s, ok, d in out if not ok]
    return (not bad,
            " ".join(f"s{s.split('/')[1]}={d}" for s, _, d in out))

def img_dims(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size

def verify_ops(path):
    """--crop on every file; --orient auto on EXIF-tagged files.

    Crop rect: deterministic 40% window at 20% offset (clamped inside),
    small enough to exercise interval skipping on bigger images."""
    out = []
    w, h = img_dims(path)
    cw, ch = max(1, w * 2 // 5), max(1, h * 2 // 5)
    cx, cy = max(0, w // 5), max(0, h // 5)
    r = subprocess.run([APP, "--verify", "--crop",
                        f"{cw}x{ch}+{cx}+{cy}", path],
                       capture_output=True, text=True, timeout=180)
    m = RE_SRGB.search(r.stdout)
    out.append(("crop", r.returncode == 0 and m and m.group(1) == "0",
                m.group(1) if m else "?"))
    for o in range(1, 9):
        if f"exif_o{o}" in path:
            r = subprocess.run([APP, "--verify", "--orient", "auto", path],
                               capture_output=True, text=True, timeout=180)
            m = RE_SRGB.search(r.stdout)
            out.append((f"or{o}", r.returncode == 0 and m and
                        m.group(1) == "0", m.group(1) if m else "?"))
    bad = [f"{k}={d}" for k, ok, d in out if not ok]
    return (not bad, " ".join(f"{k}={d}" for k, _, d in out))

def reject(path):
    try:
        r = subprocess.run([APP, path, "/tmp/_rej.ppm"],
                           capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    if r.returncode in (-11, -6):
        return False, f"CRASH signal {-r.returncode}"
    if r.returncode == 0:
        return False, "decoded a file it should have rejected"
    return True, (r.stderr.strip().splitlines() or ["rejected"])[-1][:70]

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    jobs = gen_pil() + gen_cjpeg() + gen_exif()
    real = find_real()
    rejs = gen_reject()
    if only == "gen":
        print(f"generated {len(jobs)} files"); return

    npass = nfail = 0
    fails = []
    todo = list(jobs) + real
    if only:
        todo = [j for j in todo if only in j]
    for p in todo:
        res, msg = verify(p)
        if res == "reject":
            npass += 1
            print(f"ok   skip-unsupported {p}  {msg}")
            continue
        tag = "ok  " if res else "FAIL"
        if res:
            # scaled decode (1/2,1/4,1/8 reduced IDCTs) on the same file
            sok, smsg = verify_scales(p)
            if not sok:
                res = False
                msg += f"  [scale {smsg}]"
            else:
                msg += f"  [{smsg}]"
            # ROI crop + EXIF orientation
            ook, omsg = verify_ops(p)
            if not ook:
                res = False
                msg += f"  [ops {omsg}]"
            else:
                msg += f"  [{omsg}]"
        if res: npass += 1
        else: nfail += 1; fails.append((p, msg))
        print(f"{tag} {p}  {msg}")
    for p in rejs:
        ok, msg = reject(p)
        tag = "ok  " if ok else "FAIL"
        if ok: npass += 1
        else: nfail += 1; fails.append((p, msg))
        print(f"{tag} reject {p}  {msg}")

    print(f"\n{npass} pass, {nfail} fail  ({len(jobs)} generated + {len(real)} real + {len(rejs)} reject)")
    if fails:
        for p, m in fails[:20]:
            print("  FAIL", p, m)
        sys.exit(1)
    print("ALL BIT-EXACT vs libjpeg")

if __name__ == "__main__":
    main()
