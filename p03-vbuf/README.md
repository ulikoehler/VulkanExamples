# p03-vbuf — How to use vertex buffers in Vulkan (staging upload + vertex input)

Vertex buffer + staging upload: host-visible staging buffer → `vkCmdCopyBuffer` → device-local vertex buffer → `vkCmdBindVertexBuffers`

**Tutorial post:** [How to use vertex buffers in Vulkan (staging upload + vertex input)](https://techoverflow.net/2026/09/26/how-to-use-vertex-buffers-in-vulkan-staging-upload-vertex-input/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
