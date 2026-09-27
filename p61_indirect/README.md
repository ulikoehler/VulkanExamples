# p61_indirect — How to do GPU-driven rendering in Vulkan with drawIndirectCount

Compute culling → `vkCmdDrawIndirectCount` — 97 of 256 draws decided on-GPU

**Tutorial post:** [How to do GPU-driven rendering in Vulkan with drawIndirectCount](https://techoverflow.net/2026/09/27/how-to-do-gpu-driven-rendering-in-vulkan-with-drawindirectcount/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
