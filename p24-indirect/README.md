# p24-indirect — How to use vkCmdDrawIndirect in Vulkan for GPU-driven drawing

`vkCmdDrawIndirect` — draw parameters live in a device buffer, one call renders a 2×2 grid

**Tutorial post:** [How to use vkCmdDrawIndirect in Vulkan for GPU-driven drawing](https://techoverflow.net/2026/09/27/how-to-use-vkcmddrawindirect-in-vulkan-for-gpu-driven-drawing/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
