# p19-screenshot — How to take a screenshot of a Vulkan window programmatically

Swapchain readback: `eTransferSrc` usage gate, layout transitions around `vkCmdCopyImageToBuffer`, 10-bit format unpacking

**Tutorial post:** [How to take a screenshot of a Vulkan window programmatically](https://techoverflow.net/2026/09/27/how-to-take-a-screenshot-of-a-vulkan-window-programmatically-swapchain-readback/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
