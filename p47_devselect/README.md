# p47_devselect — How to select between iGPU and dGPU in Vulkan

Physical-device scoring: gate on requirements, prefer `eDiscreteGpu` > `eIntegratedGpu` > `eCpu`, `--cpu` override

**Tutorial post:** [How to select between iGPU and dGPU in Vulkan](https://techoverflow.net/2026/09/27/how-to-select-between-igpu-and-dgpu-in-vulkan-physical-device-picking/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
