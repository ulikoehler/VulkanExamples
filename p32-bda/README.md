# p32-bda — How to use VK_KHR_buffer_device_address in Vulkan for pointer-based shader access

`VK_KHR_buffer_device_address` — `uint64` pointer in a push constant, zero descriptor sets

**Tutorial post:** [How to use VK_KHR_buffer_device_address in Vulkan for pointer-based shader access](https://techoverflow.net/2026/09/28/how-to-use-vk-khr-buffer-device-address-in-vulkan-for-pointer-based-shader-access/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
