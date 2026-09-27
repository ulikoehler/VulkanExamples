# p35-localread — How to read the framebuffer inside a render pass with dynamic_rendering_local_read

`VK_KHR_dynamic_rendering_local_read` + `subpassLoad` — read the framebuffer mid-rendering (TBDR-style feedback)

**Tutorial post:** [How to read the framebuffer inside a render pass with dynamic_rendering_local_read](https://techoverflow.net/2026/09/28/how-to-read-the-framebuffer-inside-a-render-pass-in-vulkan-with-dynamic-rendering-local-read/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
