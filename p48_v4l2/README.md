# p48_v4l2 — How to capture a USB webcam zero-copy into Vulkan with V4L2 DMA-Buf export

V4L2 `VIDIOC_EXPBUF` — USB webcam frame as dma-buf fd → `VkImage`, YUYV unpacked in the shader

**Tutorial post:** [How to capture a USB webcam zero-copy into Vulkan with V4L2 DMA-Buf export](https://techoverflow.net/2026/09/27/how-to-capture-a-usb-webcam-zero-copy-into-vulkan-with-v4l2-dma-buf-export/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
