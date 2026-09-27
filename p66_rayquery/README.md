# p66_rayquery — How to do ray tracing in a fragment shader with VK_KHR_ray_query

`VK_KHR_ray_query` — BLAS+TLAS build, hardware shadow ray per fragment in a normal pipeline

**Tutorial post:** [How to do ray tracing in a fragment shader with VK_KHR_ray_query](https://techoverflow.net/2026/09/27/how-to-do-ray-tracing-in-a-fragment-shader-with-vk-khr-ray-query/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
