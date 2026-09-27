# p10-ubo — How to use uniform buffers in Vulkan for MVP matrix transformations

UBO + MVP matrix — a tilting quad verified by scanline widths

**Tutorial post:** [How to use uniform buffers in Vulkan for MVP matrix transformations](https://techoverflow.net/2026/09/27/how-to-use-uniform-buffers-in-vulkan-for-mvp-matrix-transformations/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
