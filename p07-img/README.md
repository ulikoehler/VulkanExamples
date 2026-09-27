# p07-img — How to write a storage image in a Vulkan compute shader and sample it

Compute shader writing a `storageImage` (rgba8), then sampling it in a graphics pass — compute→sample chaining

**Tutorial post:** [How to write a storage image in a Vulkan compute shader and sample it](https://techoverflow.net/2026/09/26/how-to-write-a-storage-image-in-a-vulkan-compute-shader-and-sample-it/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
