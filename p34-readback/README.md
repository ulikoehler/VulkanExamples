# p34-readback — How to do non-stalling GPU readback in Vulkan with a fenced ring

Non-stalling readback: 3-slot fenced ring, 6 frames, zero `waitIdle`

**Tutorial post:** [How to do non-stalling GPU readback in Vulkan with a fenced ring](https://techoverflow.net/2026/09/28/how-to-do-non-stalling-gpu-readback-in-vulkan-with-a-fenced-ring/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
