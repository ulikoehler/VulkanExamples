# p20-pngshot — How to encode a Vulkan framebuffer readback as PNG with libpng

Same readback → libpng PNG encode; check.py contains a pure-Python PNG decoder as end-to-end verifier

**Tutorial post:** [How to encode a Vulkan framebuffer readback as PNG with libpng](https://techoverflow.net/2026/09/27/how-to-encode-a-vulkan-framebuffer-readback-as-png-with-libpng/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
