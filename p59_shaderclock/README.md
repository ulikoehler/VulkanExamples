# p59_shaderclock — How to measure GPU cycles inside a shader with VK_KHR_shader_clock

`VK_KHR_shader_clock` — in-shader `clock2x32ARB` deltas rendered as a per-pixel heatmap

**Tutorial post:** [How to measure GPU cycles inside a shader with VK_KHR_shader_clock](https://techoverflow.net/2026/09/27/how-to-measure-gpu-cycles-inside-a-shader-with-vk-khr-shader-clock/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
