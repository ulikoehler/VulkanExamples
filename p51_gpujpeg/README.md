# p51_gpujpeg — How to encode JPEG in a Vulkan compute shader — parallel Huffman via restart markers

Baseline JPEG encoder in a compute shader: per-MCU invocations, restart markers, SSBO segments

**Tutorial post:** [How to encode JPEG in a Vulkan compute shader — parallel Huffman via restart markers](https://techoverflow.net/2026/09/27/how-to-encode-jpeg-in-a-vulkan-compute-shader-per-mcu-parallel-huffman/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
