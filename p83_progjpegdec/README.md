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
        ▼  quantized coefs (zigzag i16 pairs in SSBO)
   recon shader: one invocation per 8x8 block
        │   dequant + libjpeg islow integer IDCT
        ▼  component planes
   upsample shader: one invocation per output pixel
           libjpeg "fancy" upsampling + fixed-point YCbCr→RGB ──► PPM
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

## Performance (Radeon Renoir iGPU vs libjpeg-turbo djpeg)

| File | GPU | CPU | Note |
|---|---|---|---|
| prog 3MP, restart=2 | 29 MP/s | ~43 | intervals parallelize per scan |
| prog gray 0.5MP | 39 MP/s | ~37 | faster than CPU |
| prog 4:2:0 0.5MP | 4.5 MP/s | ~24 | mostly-serial AC scans |
| prog 3MP, DRI=0 | 1.2 MP/s | ~43 | one serial stream per scan |
| baseline 4:2:0 0.5MP | 12 MP/s | ~48 | regression-free path |

Entropy parallelism is per restart interval — `DRI=0` progressive files
have a single serial bit stream per scan, which is latency-bound on any
decoder (speculative resync would be the next step).

Supported: baseline SOF0/SOF1 + progressive SOF2, 8-bit, 1–3 components,
all sampling factors, restart markers (any DRI), optimized/custom
Huffman tables, arbitrary scan scripts. Rejected cleanly: arithmetic
coding, non-8-bit, >3 components, lossless modes.
