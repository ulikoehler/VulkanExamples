# p09-ibuf — How to use index buffers in Vulkan (vkCmdDrawIndexed)

`vkCmdDrawIndexed` + `VK_INDEX_TYPE_UINT16` — 4 verts / 6 indices quad

**Tutorial post:** [How to use index buffers in Vulkan (vkCmdDrawIndexed)](https://techoverflow.net/2026/09/27/how-to-use-index-buffers-in-vulkan-to-draw-indexed-geometry-with-vkcmddrawindexed/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
