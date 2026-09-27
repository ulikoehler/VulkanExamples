# p68_pngdec — How to decode PNG entirely on the GPU — DEFLATE inflate in a compute shader

PNG decode in compute: full DEFLATE inflate (stored/fixed/dynamic + LZ77) + unfiltering

**Tutorial post:** [How to decode PNG entirely on the GPU — DEFLATE inflate in a compute shader](https://techoverflow.net/2026/09/27/how-to-decode-png-entirely-on-the-gpu-deflate-inflate-in-a-compute-shader/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
