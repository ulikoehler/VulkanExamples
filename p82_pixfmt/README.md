# p82_pixfmt — pixel-format conversion in compute, bit-exact vs Pillow

One dispatch converts between pixel formats. Every conversion is
verified **0 pixel diff** against `PIL.Image.convert()` (or numpy for
BGR, which PIL has no mode for) on noise + gradient images — noise
exercises all 256 YCbCr LUT entries and every HSV sector.

```
in.pnm ──memcpy──> SSBO (packed u8) ──> conv shader ──> packed u8
                                                      -> out.pnm
```

One invocation produces one 4-byte output word, gathering the 1-2
output pixels whose bytes it needs — packed layout on both sides, no
atomics, no CPU repack.

## Conversions

| conv | in→out | PIL reference |
|---|---|---|
| `rgb2rgba` / `rgba2rgb` | 3→4 / 4→3 | `convert("RGBA"/"RGB")` |
| `rgb2bgr` / `bgr2rgb` | channel swap | numpy `[..., ::-1]` |
| `rgb2gray` / `rgba2gray` | 3/4→1 | `convert("L")` — PIL's `L24` integer luma |
| `gray2rgb` / `gray2rgba` | 1→3/4 | `convert("RGB"/"RGBA")` |
| `rgb2ycbcr` / `ycbcr2rgb` | 3→3 | `convert("YCbCr"/"RGB")` — JFIF integer LUTs |
| `rgb2hsv` / `hsv2rgb` | 3→3 | `convert("HSV"/"RGB")` — colorsys float path |
| `gray2ycbcr` / `ycbcr2gray` | 1→3 / 3→1 | `convert("YCbCr"/"L")` |

## Build / run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc

./app in.ppm out.ppm rgb2gray          # P5/P6 PNM in, P5/P6 out
./app --raw 640x480x4 in.raw out.raw rgba2rgb   # headerless bytes
./app --bench 10 --raw 6000x4000x3 big.raw o.raw rgb2ycbcr
./check.py          # bit-exact corpus vs PIL
./check.py --bench  # + throughput comparison
```

## The interesting part: staying bit-exact on a GPU

PIL's C implementations are *not* all float32-clean, and GPUs don't
guarantee what C does. Three traps found while chasing 0-diff:

1. **fp32 division is 2.5-ULP in Vulkan.** RADV lowers it to
   reciprocal-multiply: `246.0f/246.0f` = 0.99999994, not 1.0 — C's
   SSE division is correctly rounded. The shader computes the HSV
   ratios with **64-bit integer long division** (`idivRN`) instead.
2. **The optimizer removes intermediate float roundings.** C stores
   `h = fmod(h/6.0 + 1.0, 1.0)` to a `float` (rounding), then multiplies
   in double. The shader compiler folds `float(...)` back into the
   producing double expression, so `h*255` came out 169.9999… → 169
   instead of 170. A bit-cast round-trip (`fr()`) forces the rounding
   to materialize.
3. **Fast-math folds fma residuals away.** `fma(x, 255, -x*255)`
   (the exact guard bits of `trunc(x*255)`) is algebraically 0 — the
   compiler may assume that, losing the last bit. `mul255trunc` and
   `div6rn` therefore do their arithmetic on **integer mantissas**
   (shift-and-round), which no FP optimization can rewrite.

YCbCr uses Pillow's exact `INT16` tables (`SCALE=6`, `>>6` arithmetic
shift, `+128` chroma bias) regenerated on the host as
`(int)(coef*64*i + 0.5)` — C's round-half-up cast, which is asymmetric
for negatives.

## Performance (RADV/Renoir iGPU, 24 MP, end-to-end incl. 72 MB upload + readback)

| conv | GPU | PIL | |
|---|---:|---:|---|
| rgb2rgba | 104 ms | 135 ms | 1.3x |
| rgb2gray |  51 ms |  38 ms | PIL's L convert is hand-SIMD; transfer-bound here |
| rgb2ycbcr |  99 ms | 225 ms | 2.3x |
| rgb2hsv | 424 ms | 701 ms | 1.7x (integer-exact path is compute-heavy) |

Subtract ~35 ms of fixed upload/readback: the dispatch itself is
16-390 ms depending on conversion. Chained after a GPU decoder (p80)
or ahead of a GPU resizer (p81) there is no transfer at all.
