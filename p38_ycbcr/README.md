# p38_ycbcr — How to sample YUV video frames directly as RGB in Vulkan

`VK_KHR_sampler_ycbcr_conversion` — NV12 planes → RGB inside an immutable sampler

**Tutorial post:** [How to sample YUV video frames directly as RGB in Vulkan](https://techoverflow.net/2026/09/27/how-to-sample-yuv-video-frames-directly-as-rgb-in-vulkan-with-vk-khr-sampler-ycbcr-conversion/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
