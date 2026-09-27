# p13-inst — How to use instanced rendering in Vulkan with per-instance vertex attributes

`vkCmdDraw(instanced)` — 256 quads in one draw, per-instance data via instance-rate vertex attributes

**Tutorial post:** [How to use instanced rendering in Vulkan with per-instance vertex attributes](https://techoverflow.net/2026/09/27/how-to-use-instanced-rendering-in-vulkan-with-per-instance-vertex-attributes/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
