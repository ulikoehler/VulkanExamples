# p36-skeleton — How to structure a minimal Vulkan renderer: init, frame, resize, shutdown

Full app skeleton: init → frame loop → resize → clean shutdown, 90 frames under Xvfb

**Tutorial post:** [How to structure a minimal Vulkan renderer: init, frame, resize, shutdown](https://techoverflow.net/2026/09/28/how-to-structure-a-minimal-vulkan-renderer-init-frame-resize-shutdown/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
