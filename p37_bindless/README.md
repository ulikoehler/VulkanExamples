# p37_bindless — How to use bindless texture arrays in Vulkan

`VK_EXT_descriptor_indexing` — one runtime descriptor array, `nonuniformEXT` per-fragment indexing

**Tutorial post:** [How to use bindless texture arrays in Vulkan](https://techoverflow.net/2026/09/27/how-to-use-bindless-texture-arrays-in-vulkan-with-vk-ext-descriptor-indexing/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
