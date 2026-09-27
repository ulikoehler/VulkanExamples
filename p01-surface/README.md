# p01-surface — How to create a Vulkan surface using GLFW

Vulkan instance + GLFW window + `VkSurfaceKHR` + clearing to a solid color — the minimal "is Vulkan working" program

**Tutorial post:** [How to create a Vulkan surface using GLFW](https://techoverflow.net/2026/09/25/how-to-create-a-vulkan-surface-using-glfw/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
