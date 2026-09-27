# p78_coopmat — How to use VK_KHR_cooperative_matrix in Vulkan

`VK_KHR_cooperative_matrix` — 16×16×16 int8 matmul distributed over the subgroup

**Tutorial post:** [How to use VK_KHR_cooperative_matrix in Vulkan](https://techoverflow.net/2026/09/27/how-to-use-vk-khr-cooperative-matrix-in-vulkan/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
