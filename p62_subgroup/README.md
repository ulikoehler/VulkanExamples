# p62_subgroup — How to use subgroup operations in Vulkan compute shaders

`subgroupAdd`/`subgroupBallot`/`subgroupBroadcastFirst`/`subgroupShuffle` verified via SSBO

**Tutorial post:** [How to use subgroup operations in Vulkan compute shaders](https://techoverflow.net/2026/09/27/how-to-use-subgroup-operations-in-vulkan-compute-shaders/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
