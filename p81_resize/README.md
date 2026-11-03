# p81_resize — GPU image resizing, bit-exact vs Pillow

Two compute passes resize an 8-bit image with Pillow's exact resampling
semantics — `check.py` verifies **0 pixel diff** against
`PIL.Image.resize()` for every filter (NEAREST, BOX, BILINEAR, HAMMING,
BICUBIC, LANCZOS) across up/down/single-axis scaling, extreme ratios,
odd sizes and grayscale.

## Build / run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc

./app in.ppm out.ppm 640x480 lanczos   # P6 (RGB) or P5 (gray) PNM
./app --bench 20 in.ppm 1500x1000 bicubic   # in-process benchmark
./check.py          # bit-exact corpus vs PIL (needs Pillow)
./check.py --bench  # + per-filter GPU vs PIL timing
python3 compare.py  # writes compare.png
```

## How it works

```
in.ppm ──memcpy──> SSBO (packed u8)
                     │ pass 1: horizontal resample, 1 px/invocation
                     ▼
              tmp SSBO (RGBA u32, inH x outW)
                     │ pass 2: vertical resample, 1 px/invocation
                     ▼
              out SSBO (RGBA u32, outH x outW)
                     │ pass 3: pack RGBA u32 -> u8 (1 word/invocation)
                     ▼
              cached readback -> out.ppm   (pure memcpy, no CPU repack)
```

Same separable structure as Pillow's `libImaging/Resample.c`:

- The host precomputes per-axis `{xmin, nTaps}` bounds and int32
  coefficients, quantized to `PRECISION_BITS=22` with
  round-half-away-from-zero after double-precision filter evaluation and
  normalization — the exact math PIL uses.
- Kernels replicate PIL including its quirks: `filterscale =
  max(in/out, 1)` widens the support on downscale, edges clip the
  window and renormalize, BOX uses the asymmetric `x > -0.5 && x <= 0.5`
  test, NEAREST is `ImagingScaleAffine`'s *iterative* double
  accumulation (`xo += a0` — repeated adds drift off the closed form,
  flipping a few pixel picks), encoded as a degenerate 1-tap resample
  so the same shader covers it.
- Accumulation is int32: `ss = 1<<21; ss += in*k`, then
  `clamp(ss>>22, 0, 255)` — PIL's `clip8`. The horizontal pass outputs
  u8 (clamped), matching PIL's intermediate image exactly.
- All CPU<->GPU format work is on the GPU: the input goes up packed
  (no RGBA expansion), and a tiny pack pass emits packed u8 so the
  readback is a single `memcpy`.

## Performance (RADV/Renoir iGPU, 6000x4000 -> 1500x1000)

| filter   | GPU total* | of which dispatch | PIL resize |
|----------|-----------:|------------------:|-----------:|
| nearest  |    39 ms   |      ~4 ms        |     4 ms   |
| bilinear |    48 ms   |     ~12 ms        |   168 ms   |
| lanczos  |    63 ms   |     ~29 ms        |   397 ms   |

*end-to-end incl. 72 MB upload + readback. With GPU-resident input
(e.g. chained after the p80 GPU JPEG decoder) only the dispatch cost
remains: **~10-30 ms for a 24 MP -> 1.5 MP resize, ~6-14x PIL**.
The 24 MB memcpy dominates the standalone path (~24 ms).

## Files

- `main.cpp` — everything: PNM I/O, PIL-exact coefficient builder,
  GLSL strings, ~150 lines of Vulkan plumbing via
  [`vkmini.hpp`](../vkmini.hpp)
- `check.py` — corpus verification + bench
- `compare.py` — writes `compare.png`
