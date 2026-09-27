# p55_localread — How to read color attachments in the same render pass with VK_KHR_dynamic_rendering_local_read

`VK_KHR_dynamic_rendering_local_read` + `subpassLoad` — in-pass color feedback without a render pass

**Tutorial post:** [How to read color attachments in the same render pass with VK_KHR_dynamic_rendering_local_read](https://techoverflow.net/2026/09/27/how-to-read-color-attachments-in-the-same-render-pass-with-vk-khr-dynamic-rendering-local-read/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
