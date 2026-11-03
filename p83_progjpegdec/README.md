# p83_progjpegdec — Progressive JPEG decoding in a Vulkan compute shader

Progressive (SOF2) **and** baseline (SOF0/1) JPEG decoding on the GPU,
extending p80_gpujpegdec to multi-scan streams: spectral selection,
successive approximation, EOBRUN zero-runs and AC refinement correction
bits — all mirrored instruction-for-instruction from libjpeg's
`jdphuff.c`. **Bit-exact vs libjpeg-turbo** across the whole corpus
(progressive + baseline + restart + no-restart + 4:2:0/4:4:4/gray +
Pillow- and cjpeg-encoded).

**Tutorial post:** [How to decode progressive JPEG entirely on the GPU](https://techoverflow.net/2026/11/03/how-to-decode-progressive-jpeg-entirely-on-the-gpu/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc -ljpeg
./app input.jpg out.ppm          # decode (baseline or progressive)
./app --verify input.jpg         # compare against libjpeg (coefs + pixels)
./app --bench 10 input.jpg       # throughput benchmark
./check.py                       # builds + verifies the full corpus
./app --cpu input.jpg            # force CPU (libjpeg) path for A/B timing
./app --scale 1/8 in.jpg t.ppm   # reduced-IDCT scale (1/2, 1/4, 1/8)
./app --crop 400x300+80+60 in.jpg c.ppm    # MCU-window ROI decode
./app --orient auto in.jpg o.ppm # apply EXIF orientation (1..8)
STAGE=1 ./app in.jpg out.ppm     # per-stage timing
SCANLIMIT=N ./app in.jpg         # stop after scan N (debug)
SCANMIN=N ./app in.jpg           # start at scan N (debug)
SUBMITMCUS=N ./app in.jpg        # MCU budget per submit (default 32768)
```

`-ljpeg` (libjpeg-turbo) is only the *reference*: `jpeg_read_coefficients`
verifies the entropy stage coef-for-coef, `jpeg_start_decompress` verifies
final pixels. All decode stages run on-GPU.

## How it works

A progressive JPEG is N sequential *scans*; each scan only covers a
subset of the coefficient space (`Ss..Se` spectral band, `Ah/Al`
approximation bits). One decoder handles both worlds — a baseline file
is just a single sequential scan:

```
JPEG file ──host marker walk──► per-scan descriptors:
        │   SOS params (Ss/Se/Ah/Al + component table ids)
        │   restart-interval grid (byteStart, mcuStart, nMcu)
        │   *snapshotted* DHT/DQT tables — cjpeg re-emits optimized
        │   tables before every scan; each scan needs its own copy!
        ▼
   entropy shader: one invocation per restart interval
        │   16-bit fused Huffman LUT (len|run|size|value in 1 load)
        │   5 scan modes: sequential, DC-first, DC-refine,
        │   AC-first, AC-refine — coefficients persist GPU-side
        │   across scan dispatches; refine scans read-modify-write.
        │   Long intervals split into 2048-MCU generations handing
        │   bit-reader state + DC predictors + EOBRUN through an SSBO.
        │   DRI=0 scans: CPU "survey" symbol-walk synthesizes virtual
        │   restart intervals (flag=2 state slots) — see below.
        │   --crop: intervals fully outside the window emit nMcu=0.
        ▼  quantized coefs (zigzag i16 pairs in SSBO)
   recon shader: one invocation per 8x8 block
        │   dequant + libjpeg islow integer IDCT
        │   or jidctred 4x4/2x2/1x1 reduced IDCT (--scale 1/2|1/4|1/8)
        │   --crop: out-of-window blocks early-return
        ▼  component planes
   upsample shader: one invocation per output pixel
        │   libjpeg "fancy" upsampling + fixed-point YCbCr→RGB
        │   --crop/--orient: output px -> un-oriented fetch via omap()
        ▼  PPM
```

## Three bugs worth documenting

1. **Per-scan Huffman tables.** cjpeg optimizes tables per scan and
   re-emits DHT before every SOS — decoding every scan with the final
   table state works for DC (never redefined) but garbles all AC scans.
   Each `Scan` snapshots tables at SOS-parse time; unchanged tables
   share one 256K-entry fast-LUT slot.

2. **Marker prefetch vs restart intervals.** The bit reader greedily
   buffers bits — near an interval end it prefetches the RST marker
   itself and sets `end`, skipping later MCUs that still hold real
   buffered bits. Fix: a `pad` counter tracks how many buffered bits
   came *after* the marker; an MCU is skipped only when
   `end && cnt <= pad` (nothing real remains).

3. **Watchdog-split submits.** All scan/generation dispatches go into
   one command buffer (tables/LUTs/chunk grids are concatenated shared
   buffers with per-scan push-constant offsets) — but a single
   multi-second submit can get killed by driver job timeouts on some
   GPUs, producing *non-deterministic* partial output. Dispatches are
   grouped into submits capped at 32768 MCU-work each; typical files
   stay single-submit.

## DRI=0: virtual restart intervals via CPU survey

Most real-world JPEGs have no restart markers (`DRI=0`) — the entropy
stream is one serial bit stream per scan, which used to mean ~1 MP/s.
Instead of speculative decode, a cheap **survey pass** walks the scan's
Huffman symbols once on the CPU (canonical mincode/maxcode lookup —
bit position + DC predictors + EOBRUN only, ~2–4% of full decode cost)
and snapshots decoder state every `kSurveyIv=256` MCUs. Those 12-word
states are uploaded into the GPU state buffer; chunks referencing them
run with `flag=2` — each virtual interval starts mid-stream with full
bit-reader context, so the unchanged parallel path decodes it.

Two subtleties:

- **Deferred surveys.** AC-refine scans can't be surveyed up front —
  whether a correction bit exists depends on coef values written by
  *previous* scans. Their survey runs between submits: entropy for all
  earlier scans completes, the host reads `coefBuf` (host-coherent),
  surveys, uploads states, then dispatches that scan's chunks. If a
  deferred survey lands before coefBuf's first fill, the host zeroes it.
- **Interval length back-fill.** The survey's `snap()` records
  `(byteStart, mcuStart)` per boundary; each interval's `nMcu` is the
  delta to the *next* boundary — forgetting that back-fill emits a
  zero-length first interval the shader's `nMcu==0` early-out skips.

With `--crop`, skipped intervals would leave zero coefs — any component
with a deferred scan therefore keeps *all* scans touching it fully
decoded (survey would desync on missing coefs).

## ROI crop, scaled IDCT, EXIF orientation

- **`--crop WxH+X+Y`** (in *oriented* output space): mapped through the
  orientation to a source-pixel window (+1 MCU margin for fancy
  upsample's ±1 sample reads). Intervals fully outside emit `nMcu=0`
  chunks — bit-exact for restart-marked and surveyed scans; recon skips
  out-of-window blocks; upsample samples the crop only.
- **`--scale M/N`**: libjpeg reduced IDCTs (`jidctred.c` 4×4/2×2/1×1)
  bit-exact, with per-component `DCT_scaled_size` doubling (4:2:0 chroma
  becomes fullsize at 1/2) and word-per-sample planes for ss∈{1,2} vs
  byte-packed for {4,8}. Entropy cost is unchanged — savings come from
  recon+upsample.
- **`--orient auto|1..8`**: APP1/TIFF tag `0x0112` parse; the upsample
  maps each output pixel back through `omap()` — verified vs PIL
  `exif_transpose` on all 8 orientations.

## Performance (Radeon Renoir iGPU, 1.9 MPix files)

| File | full | --crop 400x300 | --scale 1/8 | CPU |
|---|---|---|---|---|
| prog DRI=0 | 18.0 MP/s | 19.9 | 18.2 | ~43 |
| prog restart=8 | 8.7 MP/s | 13.2 | 8.7 | ~43 |
| baseline DRI=0 | 29.4 MP/s | 42.6 | 29.6 | ~48 |
| baseline restart=4 | 20.3 MP/s | 28.5 | 20.5 | ~48 |

Crop skips whole intervals: ~1.5× on files whose kept-window fraction
is small. Scaled IDCT saves mostly recon/upsample work — entropy decode
still walks every bit, so gains are modest on entropy-bound files.
DRI=0 progressive went 1.2 → ~18 MP/s vs the pre-survey serial path.

Supported: baseline SOF0/SOF1 + progressive SOF2, 8-bit, 1–3 components,
all sampling factors, restart markers (any DRI, incl. none), optimized/
custom Huffman tables, arbitrary scan scripts, EXIF orientation.
Rejected cleanly: arithmetic coding, non-8-bit, >3 components, lossless.
