# SPDX-License-Identifier: CC0-1.0
"""Verify the GPU JPEG decoder is bit-exact vs libjpeg on a broad corpus.

Generates a corpus with PIL/cjpeg (sizes, subsampling, qualities, restart
intervals, gray, optimized Huffman, edge cases), adds real-world JPEGs found
on the system, runs ./app --verify on each and requires:
  coefs: 0/N mismatch   AND   rgb: max |diff| 0
Unsupported files (progressive, arithmetic) must fail cleanly, not crash.
"""
import os, re, subprocess, sys, glob, random

APP = "./app"
CORPUS = "corpus"
os.makedirs(CORPUS, exist_ok=True)

def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)

# ---------- corpus generation ----------

def gen_pil():
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(42)

    def grad(w, h):  # smooth diagonal gradient
        x = np.linspace(0, 255, w)[None, :]
        y = np.linspace(0, 255, h)[:, None]
        a = np.broadcast_to(x, (h, w))
        b = np.broadcast_to(y, (h, w))
        return np.stack([a, b, (a + b) / 2], -1).astype(np.uint8)

    def noise(w, h):
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def edges(w, h):  # checkerboard + circles: hard edges, high AC energy
        yy, xx = np.mgrid[0:h, 0:w]
        cb = (((xx // 8) + (yy // 8)) % 2 * 255).astype(np.uint8)
        circ = (((xx - w // 2) ** 2 + (yy - h // 2) ** 2) < (min(w, h) // 3) ** 2)
        g = np.stack([cb, np.where(circ, 255, cb // 3).astype(np.uint8),
                      (255 - cb)], -1)
        return g.astype(np.uint8)

    def photo(w, h):  # smoothed noise ~ natural image statistics
        n = rng.integers(0, 256, (h // 4 + 1, w // 4 + 1, 3), dtype=np.uint8)
        im = Image.fromarray(n).resize((w, h), Image.BICUBIC)
        return np.asarray(im)

    jobs = []
    contents = {"grad": grad, "noise": noise, "edges": edges, "photo": photo}
    sizes = [(1, 1), (7, 5), (8, 8), (16, 16), (33, 17), (64, 48),
             (120, 80), (333, 257), (512, 384), (640, 480), (1023, 767)]
    # (name, subsampling) — PIL: 0=4:4:4 1=4:2:2 2=4:2:0
    for cname, fn in contents.items():
        for (w, h) in sizes[::2] if cname == "photo" else sizes:
            for sub, q in [(0, 90), (1, 85), (2, 75)]:
                p = f"{CORPUS}/{cname}_{w}x{h}_s{sub}q{q}.jpg"
                Image.fromarray(fn(w, h)).save(p, quality=q, subsampling=sub)
                jobs.append(p)
    # quality sweep on one photo
    for q in (5, 25, 50, 75, 95, 100):
        p = f"{CORPUS}/photo_q{q}.jpg"
        Image.fromarray(photo(320, 240)).save(p, quality=q, subsampling=2)
        jobs.append(p)
    # grayscale
    for (w, h) in [(17, 9), (256, 200), (641, 481)]:
        p = f"{CORPUS}/gray_{w}x{h}.jpg"
        Image.fromarray(grad(w, h)[:, :, 0]).save(p, quality=90)
        jobs.append(p)
    return jobs

def gen_cjpeg():
    """cjpeg paths PIL can't do: restart intervals, 4:4:0, optimize, RGB."""
    import numpy as np
    from PIL import Image
    jobs = []
    rng = np.random.default_rng(7)
    n = rng.integers(0, 256, (200, 320, 3), dtype=np.uint8)
    src = f"{CORPUS}/_cj_src.ppm"
    Image.fromarray(n).save(src)
    variants = [
        ("rst1",  "-restart 1"),          # DRI=1: every MCU
        ("rst4",  "-restart 4"),
        ("rst1B", "-restart 1B"),         # restart every row (byte units)
        ("rst16", "-restart 16"),
        ("s12",   "-sample 1x2"),         # 4:4:0 vertical subsampling
        ("s21",   "-sample 2x1"),         # 4:2:2
        ("s22",   "-sample 2x2"),         # 4:2:0
        ("opt",   "-optimize"),           # optimized Huffman tables
        ("q10",   "-quality 10"),
        ("sm",    "-smooth 50"),          # smoothed dct input
    ]
    for name, opts in variants:
        p = f"{CORPUS}/cj_{name}.jpg"
        r = sh(f"cjpeg {opts} -outfile {p} {src}")
        if r.returncode == 0 and os.path.exists(p):
            jobs.append(p)
        else:
            print(f"  (skip cjpeg {opts}: {r.stderr.strip()[:80]})")
    # tiny restarts on a small image (intervals smaller than an MCU row)
    Image.fromarray(n[:40, :56]).save(f"{CORPUS}/_cj_src2.ppm")
    p = f"{CORPUS}/cj_small_rst.jpg"
    sh(f"cjpeg -restart 1 -outfile {p} {CORPUS}/_cj_src2.ppm")
    if os.path.exists(p):
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
    cases = {
        "progressive":  ("-progressive", None),
        "arithmetic":   ("-arithmetic", None),
        "progressive2": (None, {"progressive": True}),
    }
    for name, (opts, pilkw) in cases.items():
        p = f"{CORPUS}/rej_{name}.jpg"
        if opts:
            r = sh(f"cjpeg {opts} -outfile {p} {CORPUS}/_rej.ppm")
            ok = r.returncode == 0
        else:
            try:
                Image.fromarray(n).save(p, **pilkw)
                ok = True
            except Exception as e:
                print(f"  (skip {name}: {e})")
                ok = False
        if ok and os.path.exists(p):
            jobs.append(p)
    # truncated before SOS -> unparseable
    tp = f"{CORPUS}/rej_trunc.jpg"
    Image.fromarray(n).save(tp, quality=85)
    d = open(tp, "rb").read()
    open(tp, "wb").write(d[:64])
    jobs.append(tp)
    # not a jpeg at all
    np_ = f"{CORPUS}/rej_notjpg.jpg"
    open(np_, "wb").write(b"PNG\r\n\x1a\n" + os.urandom(200))
    jobs.append(np_)
    return jobs

def find_real():
    """Real-world JPEGs -> (decodable, must-reject) split via PIL."""
    pats = ["/usr/share/**/*.jp*g", os.path.expanduser("~/dev/**/*.jp*g")]
    ok, bad = [], []
    for pat in pats:
        for f in glob.iglob(pat, recursive=True):
            try:
                if not (4_000 < os.path.getsize(f) < 4_000_000):
                    continue
                from PIL import Image
                im = Image.open(f)
                if im.mode not in ("RGB", "L") or im.info.get("progressive") \
                        or im.info.get("progression"):
                    bad.append(f)
                else:
                    ok.append(f)
            except Exception:
                bad.append(f)
    random.seed(1); random.shuffle(ok)
    return ok[:40], bad[:10]

# ---------- runner ----------

RE_COEF = re.compile(r"coefs:\s*(\d+)/(\d+) mismatch")
RE_RGB = re.compile(r"rgb:\s*max \|diff\|\s*(\d+)\s+mean\s+([\d.]+)")

RE_UNSUP = re.compile(r"not supported|unsupported|not a JPEG|missing|precision",
                      re.I)

def verify(path):
    """-> (ok, msg) | ('reject', msg) for legitimately unsupported files."""
    r = subprocess.run([APP, "--verify", path],
                       capture_output=True, text=True, timeout=120)
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

def reject(path):
    try:
        r = subprocess.run([APP, path, "/tmp/_rej.ppm"],
                           capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    if r.returncode == -11 or r.returncode == -6:
        return False, f"CRASH signal {-r.returncode}"
    if r.returncode == 0:
        return False, "decoded a file it should have rejected"
    return True, (r.stderr.strip().splitlines() or ["rejected"])[-1][:70]

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    jobs = gen_pil() + gen_cjpeg()
    real, real_rej = find_real()
    rejs = gen_reject() + real_rej
    if only == "gen":
        print(f"generated {len(jobs)} files"); return

    npass = nfail = 0
    fails = []
    todo = [j for j in jobs] + real
    if only:
        todo = [j for j in todo if only in j]
    for p in todo:
        res, msg = verify(p)
        if res == "reject":
            npass += 1
            print(f"ok   skip-unsupported {p}  {msg}")
            continue
        tag = "ok  " if res else "FAIL"
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
