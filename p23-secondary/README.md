# p23-secondary — How to use secondary command buffers in Vulkan to record drawing in parallel

Secondary command buffers + `eRenderPassContinue` inheritance — three buffers, three colored quads

**Tutorial post:** [How to use secondary command buffers in Vulkan to record drawing in parallel](https://techoverflow.net/2026/09/27/how-to-use-secondary-command-buffers-in-vulkan-to-record-drawing-in-parallel/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
