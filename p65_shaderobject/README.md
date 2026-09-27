# p65_shaderobject — How to use VK_EXT_shader_object — Vulkan without VkPipeline

`VK_EXT_shader_object` — `VkShaderEXT` bound directly, all state via `vkCmdSet*EXT`, no `VkPipeline`

**Tutorial post:** [How to use VK_EXT_shader_object — Vulkan without VkPipeline](https://techoverflow.net/2026/09/27/how-to-use-vk-ext-shader-object-vulkan-without-pipelines/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
