# p15-mips — How to generate mipmaps in Vulkan using vkCmdBlitImage

Mipmap chain via `vkCmdBlitImage` — `textureLod(8)` proves the full downsample chain

**Tutorial post:** [How to generate mipmaps in Vulkan using vkCmdBlitImage](https://techoverflow.net/2026/09/27/how-to-generate-mipmaps-in-vulkan-using-vkcmdblitimage/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
