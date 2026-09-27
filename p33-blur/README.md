# p33-blur — How to write a compute post-processing filter in Vulkan

Compute post-filter: 5×5 box blur on a storage image, `eGeneral` layout

**Tutorial post:** [How to write a compute post-processing filter in Vulkan](https://techoverflow.net/2026/09/28/how-to-write-a-compute-post-processing-filter-in-vulkan/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
