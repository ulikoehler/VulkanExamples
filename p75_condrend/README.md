# p75_condrend — How to use VK_EXT_conditional_rendering in Vulkan

`VK_EXT_conditional_rendering` — GPU-side predicate skips a recorded draw

**Tutorial post:** [How to use VK_EXT_conditional_rendering in Vulkan](https://techoverflow.net/2026/09/27/how-to-use-vk-ext-conditional-rendering-in-vulkan/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
