# p11-depth — How to add a depth attachment in Vulkan for correct 3D occlusion

Depth attachment (D32), `depthWriteEnable`, near quad wins over far quad regardless of draw order

**Tutorial post:** [How to add a depth attachment in Vulkan for correct 3D occlusion](https://techoverflow.net/2026/09/27/how-to-add-a-depth-attachment-in-vulkan-for-correct-3d-occlusion/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
