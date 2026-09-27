# p05-texture — How to upload and sample a texture in Vulkan

Texture upload + sampling: staging → `VkImage`, layout transitions, `sampler2D` + descriptor set

**Tutorial post:** [How to upload and sample a texture in Vulkan](https://techoverflow.net/2026/09/26/how-to-upload-and-sample-a-texture-in-vulkan/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
