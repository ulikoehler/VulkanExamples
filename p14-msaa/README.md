# p14-msaa — How to enable MSAA multisampling in Vulkan with dynamic rendering

`eSampleCount4` + `resolveImageView` — checker counts AA blend pixels at the triangle edge

**Tutorial post:** [How to enable MSAA multisampling in Vulkan with dynamic rendering](https://techoverflow.net/2026/09/27/how-to-enable-msaa-multisampling-in-vulkan-with-dynamic-rendering/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
