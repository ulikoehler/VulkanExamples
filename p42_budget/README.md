# p42_budget — How to query GPU memory budget and live heap usage in Vulkan

`VK_EXT_memory_budget` — per-heap budget/usage, tracks 768MB allocation growth

**Tutorial post:** [How to query GPU memory budget and live heap usage in Vulkan](https://techoverflow.net/2026/09/27/how-to-query-gpu-memory-budget-and-live-heap-usage-in-vulkan-with-vk-ext-memory-budget/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
