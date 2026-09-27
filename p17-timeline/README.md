# p17-timeline — How to use timeline semaphores in Vulkan for value-based GPU synchronization

`VkSemaphoreType::eTimeline` — one semaphore counting 0→4 across submits, `vkSignalSemaphore`/`vkWaitSemaphores`

**Tutorial post:** [How to use timeline semaphores in Vulkan for value-based GPU synchronization](https://techoverflow.net/2026/09/27/how-to-use-timeline-semaphores-in-vulkan-for-value-based-gpu-synchronization/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
