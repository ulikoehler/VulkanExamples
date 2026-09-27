# p49_vaapi — How to decode H.264 with FFmpeg VA-API and import frames into Vulkan zero-copy

FFmpeg `AV_PIX_FMT_VAAPI` → `av_hwframe_map` → `AV_PIX_FMT_DRM_PRIME` → per-plane `VkImage` import

**Tutorial post:** [How to decode H.264 with FFmpeg VA-API and import frames into Vulkan zero-copy](https://techoverflow.net/2026/09/27/how-to-decode-h-264-with-ffmpeg-va-api-and-import-frames-into-vulkan-zero-copy/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
