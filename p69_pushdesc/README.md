# p69_pushdesc — How to use VK_KHR_push_descriptor in Vulkan

`VK_KHR_push_descriptor` — descriptors written into the command buffer, no pool/set

**Tutorial post:** [How to use VK_KHR_push_descriptor in Vulkan](https://techoverflow.net/2026/09/27/how-to-use-vk-khr-push-descriptor-in-vulkan-descriptors-without-descriptor-sets/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
