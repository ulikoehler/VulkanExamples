# p46_hdr — How to use HDR output in Vulkan: wide-color swapchain formats and color spaces

`VK_EXT_swapchain_colorspace` — enumerate (format, colorSpace) pairs, prefer HDR10/scRGB, fall back to SDR

**Tutorial post:** [How to use HDR output in Vulkan: wide-color swapchain formats and color spaces](https://techoverflow.net/2026/09/27/how-to-use-hdr-output-in-vulkan-wide-color-swapchain-formats-and-color-spaces/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
