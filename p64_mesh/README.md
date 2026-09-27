# p64_mesh — How to replace the vertex pipeline with VK_EXT_mesh_shader

`VK_EXT_mesh_shader` — task+mesh pipeline emits 4 procedural quads, zero vertex input

**Tutorial post:** [How to replace the vertex pipeline with VK_EXT_mesh_shader](https://techoverflow.net/2026/09/27/how-to-replace-the-vertex-pipeline-with-vk-ext-mesh-shader/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
