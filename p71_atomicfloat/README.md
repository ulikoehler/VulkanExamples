# p71_atomicfloat — How to use VK_EXT_shader_atomic_float in Vulkan

`VK_EXT_shader_atomic_float` — `atomicAdd(float)` on an SSBO, 1M adds vs CPU reference

**Tutorial post:** [How to use VK_EXT_shader_atomic_float in Vulkan](https://techoverflow.net/2026/09/27/how-to-use-vk-ext-shader-atomic-float-in-vulkan/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
