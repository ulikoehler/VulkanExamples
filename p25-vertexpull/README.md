# p25-vertexpull — How to do vertex pulling in Vulkan using an SSBO instead of vertex buffers

Vertex pulling — empty vertex-input state, `gl_VertexIndex` indexes an SSBO; no vertex buffer at all

**Tutorial post:** [How to do vertex pulling in Vulkan using an SSBO instead of vertex buffers](https://techoverflow.net/2026/09/27/how-to-do-vertex-pulling-in-vulkan-using-an-ssbo-instead-of-vertex-buffers/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
