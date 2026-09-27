# p18-validation — How to enable Vulkan validation layers and use debug labels

`VK_LAYER_KHRONOS_validation` + debug utils labels — intentionally triggers an error to show the message format

**Tutorial post:** [How to enable Vulkan validation layers and use debug labels](https://techoverflow.net/2026/09/27/how-to-enable-vulkan-validation-layers-and-use-debug-labels-for-troubleshooting/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
