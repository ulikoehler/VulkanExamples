# p04-push — Vulkan push constants explained: per-draw shader data

Push constants — per-draw data (offset + color) without descriptors; includes the std430 `vec3` alignment trap

**Tutorial post:** [Vulkan push constants explained: per-draw shader data](https://techoverflow.net/2026/09/26/vulkan-push-constants-explained-per-draw-shader-data/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
