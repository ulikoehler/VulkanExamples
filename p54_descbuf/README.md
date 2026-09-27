# p54_descbuf — How to use VK_EXT_descriptor_buffer — descriptors as plain buffer memory

`VK_EXT_descriptor_buffer` — opaque descriptor bytes in a mapped buffer, no pools/sets/updates

**Tutorial post:** [How to use VK_EXT_descriptor_buffer — descriptors as plain buffer memory](https://techoverflow.net/2026/09/27/how-to-use-vk-ext-descriptor-buffer-descriptors-as-plain-buffer-memory/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
