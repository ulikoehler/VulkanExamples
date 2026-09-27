# p06-compute — Vulkan compute shaders: GPU-side data generation and readback via storage buffers

Compute shader writing into an SSBO + host readback — GPU-side data generation

**Tutorial post:** [Vulkan compute shaders: GPU-side data generation and readback via storage buffers](https://techoverflow.net/2026/09/26/vulkan-compute-shaders-gpu-side-data-generation-and-readback-via-storage-buffers/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
