# p41_multiqueue — How to use a dedicated transfer queue in Vulkan for async texture uploads

Dedicated transfer queue: async texture upload + queue-family ownership transfer synced by timeline semaphore

**Tutorial post:** [How to use a dedicated transfer queue in Vulkan for async texture uploads](https://techoverflow.net/2026/09/27/how-to-use-a-dedicated-transfer-queue-in-vulkan-for-async-texture-uploads/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
