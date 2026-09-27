# p26-descindex — How to index into a texture array in Vulkan with arrayed sampler bindings

`sampler2D tex[4]` array + dynamic-uniform index — four draws, four textures, one binding

**Tutorial post:** [How to index into a texture array in Vulkan with arrayed sampler bindings](https://techoverflow.net/2026/09/27/how-to-index-into-a-texture-array-in-vulkan-with-arrayed-sampler-bindings/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
