# p73_intdot — How to use VK_KHR_shader_integer_dot_product in Vulkan

`VK_KHR_shader_integer_dot_product` — `dotPacked4x8EXT` int8 dot products

**Tutorial post:** [How to use VK_KHR_shader_integer_dot_product in Vulkan](https://techoverflow.net/2026/09/27/how-to-use-vk-khr-shader-integer-dot-product-in-vulkan/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
