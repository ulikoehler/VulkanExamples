# p40_video — How to decode H.264 on the GPU with Vulkan Video

`VK_KHR_video_decode_queue` + `VK_KHR_video_decode_h264` — SPS/PPS parsing, video session, decode queue; clean SKIP without a decode queue

**Tutorial post:** [How to decode H.264 on the GPU with Vulkan Video](https://techoverflow.net/2026/09/27/how-to-decode-h-264-on-the-gpu-with-vulkan-video-vk-khr-video-decode-queue/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
