# p60_pipelexe — How to inspect compiled Vulkan shaders with VK_KHR_pipeline_executable_properties

`VK_KHR_pipeline_executable_properties` — SGPR/VGPR/spill stats + NIR/ACO/ISA dumps

**Tutorial post:** [How to inspect compiled Vulkan shaders with VK_KHR_pipeline_executable_properties](https://techoverflow.net/2026/09/27/how-to-inspect-compiled-vulkan-shaders-with-vk-khr-pipeline-executable-properties/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
