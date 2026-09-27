# p76_gpl — How to use VK_EXT_graphics_pipeline_library in Vulkan

`VK_EXT_graphics_pipeline_library` — pipeline split into 4 libs, two variants fast-linked

**Tutorial post:** [How to use VK_EXT_graphics_pipeline_library in Vulkan](https://techoverflow.net/2026/09/27/how-to-use-vk-ext-graphics-pipeline-library-in-vulkan/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
