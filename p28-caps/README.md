# p28-caps — How to query Vulkan device capabilities: properties, limits and features

`vkGetPhysicalDeviceProperties2` chain — limits, features, `pipelineCacheUUID`

**Tutorial post:** [How to query Vulkan device capabilities: properties, limits and features](https://techoverflow.net/2026/09/28/how-to-query-vulkan-device-capabilities-properties-limits-and-features/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
