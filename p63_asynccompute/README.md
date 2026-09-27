# p63_asynccompute — How to use a dedicated async compute queue in Vulkan

Dedicated compute queue + timeline semaphore + timestamps — true graphics/compute overlap

**Tutorial post:** [How to use a dedicated async compute queue in Vulkan](https://techoverflow.net/2026/09/27/how-to-use-a-dedicated-async-compute-queue-in-vulkan/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
