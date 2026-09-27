# p45_multiview — How to render multiple views in one Vulkan pass with multiview

`VK_KHR_multiview` — `viewMask` + layered attachment + `gl_ViewIndex`; one draw writes both layers

**Tutorial post:** [How to render multiple views in one Vulkan pass with multiview](https://techoverflow.net/2026/09/27/how-to-render-multiple-views-in-one-vulkan-pass-with-multiview-vk-khr-multiview/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
