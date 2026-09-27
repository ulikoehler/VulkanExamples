# p43_hostcopy — How to copy pixels to a Vulkan image without a command buffer

`VK_EXT_host_image_copy` — `vkCopyMemoryToImageEXT`/`vkCopyImageToMemoryEXT` without any command buffer

**Tutorial post:** [How to copy pixels to a Vulkan image without a command buffer](https://techoverflow.net/2026/09/27/how-to-copy-pixels-to-a-vulkan-image-without-a-command-buffer-vk-ext-host-image-copy/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
