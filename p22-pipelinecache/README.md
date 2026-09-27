# p22-pipelinecache — How to persist compiled Vulkan pipelines with VkPipelineCache

`vkCreatePipelineCache` + `vkGetPipelineCacheData` — cold run writes a blob, warm run reloads it

**Tutorial post:** [How to persist compiled Vulkan pipelines with VkPipelineCache](https://techoverflow.net/2026/09/27/how-to-persist-compiled-vulkan-pipelines-with-vkpipelinecache/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
