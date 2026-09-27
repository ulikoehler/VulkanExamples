# p67_pngenc — How to encode PNG in a Vulkan compute shader

PNG encode in compute: adaptive scanline filters + zlib stored-block framing + Adler-32

**Tutorial post:** [How to encode PNG in a Vulkan compute shader](https://techoverflow.net/2026/09/27/how-to-encode-png-in-a-vulkan-compute-shader/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
