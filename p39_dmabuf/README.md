# p39_dmabuf — How to import a Linux DMA-Buf as a VkImage for zero-copy sharing

`VK_EXT_external_memory_dma_buf` + `VK_EXT_image_drm_format_modifier` — export a VkImage's memory as dma-buf fd, re-import it zero-copy

**Tutorial post:** [How to import a Linux DMA-Buf as a VkImage for zero-copy sharing](https://techoverflow.net/2026/09/27/how-to-import-a-linux-dma-buf-as-a-vkimage-for-zero-copy-sharing-in-vulkan/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
