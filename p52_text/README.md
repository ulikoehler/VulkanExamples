# p52_text — How to render text in Vulkan with an SDF glyph atlas (FreeType + HarfBuzz)

HarfBuzz shaping + FreeType + CPU SDF → `R8_UNORM` atlas, `smoothstep`/`fwidth` AA

**Tutorial post:** [How to render text in Vulkan with an SDF glyph atlas (FreeType + HarfBuzz)](https://techoverflow.net/2026/09/27/how-to-render-text-in-vulkan-with-an-sdf-glyph-atlas-freetype-harfbuzz/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
