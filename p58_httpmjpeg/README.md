# p58_httpmjpeg — How to ingest HTTP MJPEG camera streams into Vulkan

HTTP `multipart/x-mixed-replace` ingest: socket parser → FFmpeg JPEG decode → upload → render

**Tutorial post:** [How to ingest HTTP MJPEG camera streams into Vulkan](https://techoverflow.net/2026/09/27/how-to-ingest-http-mjpeg-camera-streams-into-vulkan/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc -lavcodec -lavutil -lswscale
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
