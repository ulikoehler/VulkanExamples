# p21-specconst — How to use specialization constants in Vulkan to bake shader variants

`VkSpecializationInfo` — one SPIR-V → two pipeline variants (checkerboard density differs per constant)

**Tutorial post:** [How to use specialization constants in Vulkan to bake shader variants](https://techoverflow.net/2026/09/27/how-to-use-specialization-constants-in-vulkan-to-bake-shader-variants-into-pipelines/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
