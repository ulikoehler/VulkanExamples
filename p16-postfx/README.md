# p16-postfx — How to build a multi-pass post-processing pipeline in Vulkan

Two-pass post-processing: scene → offscreen → fullscreen-triangle invert pass, one command buffer

**Tutorial post:** [How to build a multi-pass post-processing pipeline in Vulkan](https://techoverflow.net/2026/09/27/how-to-build-a-multi-pass-post-processing-pipeline-in-vulkan-offscreen-pass-to-screen/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
