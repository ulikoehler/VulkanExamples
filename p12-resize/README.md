# p12-resize — How to handle swapchain recreation on window resize in Vulkan

Swapchain recreation on resize: `OUT_OF_DATE`/`SUBOPTIMAL` + framebuffer-size polling under Xvfb

**Tutorial post:** [How to handle swapchain recreation on window resize in Vulkan](https://techoverflow.net/2026/09/27/how-to-handle-swapchain-recreation-on-window-resize-in-vulkan/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
