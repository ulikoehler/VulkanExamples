# p77_rtpipeline — How to write a Vulkan ray tracing pipeline with vkCmdTraceRaysKHR

`VK_KHR_ray_tracing_pipeline` — `vkCmdTraceRaysKHR` raygen→miss/hit via SBT

**Tutorial post:** [How to write a Vulkan ray tracing pipeline with vkCmdTraceRaysKHR](https://techoverflow.net/2026/09/27/how-to-write-a-vulkan-ray-tracing-pipeline-with-vkcmdtracerayskhr/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
