# p53_letterbox — How to do aspect-correct letterboxing and cover cropping in Vulkan tile layouts

Aspect-correct fit (letterbox) / fill (stretch) / cover (crop) from source + dest rects

**Tutorial post:** [How to do aspect-correct letterboxing and cover cropping in Vulkan tile layouts](https://techoverflow.net/2026/09/27/how-to-do-aspect-correct-letterboxing-and-cover-cropping-in-vulkan-tile-layouts/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
