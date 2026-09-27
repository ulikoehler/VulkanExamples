# p02-triangle — How to draw your first triangle in Vulkan with dynamic rendering

First triangle with `VK_KHR_dynamic_rendering` — no render pass object, no vertex buffers, `gl_VertexIndex` fullscreen-less triangle, headless PPM output

**Tutorial post:** [How to draw your first triangle in Vulkan with dynamic rendering](https://techoverflow.net/2026/09/25/how-to-draw-your-first-triangle-in-vulkan-with-dynamic-rendering/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
