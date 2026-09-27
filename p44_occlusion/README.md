# p44_occlusion — How to measure fragment visibility with Vulkan occlusion queries

`vkCmdBeginQuery(eOcclusion)` — one occluded quad reports exactly 0 passing samples, visible one 1521

**Tutorial post:** [How to measure fragment visibility with Vulkan occlusion queries](https://techoverflow.net/2026/09/27/how-to-measure-fragment-visibility-with-vulkan-occlusion-queries/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
