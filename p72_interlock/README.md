# p72_interlock — How to use VK_EXT_fragment_shader_interlock in Vulkan

`VK_EXT_fragment_shader_interlock` — critical section for per-pixel non-atomic RMW

**Tutorial post:** [How to use VK_EXT_fragment_shader_interlock in Vulkan](https://techoverflow.net/2026/09/27/how-to-use-vk-ext-fragment-shader-interlock-in-vulkan/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
