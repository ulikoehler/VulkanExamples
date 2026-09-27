# p50_wall — How to write a Vulkan video wall: 3×3 grid pipeline, push constants, click-zoom

3×3 video wall: `sampler2DArray`, 9 draws, push-constant tile rect + per-tile filter + click zoom

**Tutorial post:** [How to write a Vulkan video wall: 3×3 grid pipeline, push constants, click-zoom](https://techoverflow.net/2026/09/27/how-to-write-a-vulkan-video-wall-grid-pipeline-push-constants-zoom/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
