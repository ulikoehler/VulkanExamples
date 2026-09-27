# p57_hevc — How to decode HEVC Main10 with FFmpeg VA-API and import into Vulkan

HEVC Main10: FFmpeg `hevc_vaapi` → DRM-PRIME P010 → R16/R16G16 `VkImage` import

**Tutorial post:** [How to decode HEVC Main10 with FFmpeg VA-API and import into Vulkan](https://techoverflow.net/2026/09/27/how-to-decode-hevc-main10-with-ffmpeg-va-api-and-import-into-vulkan/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc -lavcodec -lavformat -lavutil
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
