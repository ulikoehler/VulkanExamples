# p27-sync2 — How to use Vulkan synchronization2 barriers (vkCmdPipelineBarrier2)

`vkCmdPipelineBarrier2` with precise stage/access masks — compute writes → graphics samples, one cmd buffer

**Tutorial post:** [How to use Vulkan synchronization2 barriers (vkCmdPipelineBarrier2)](https://techoverflow.net/2026/09/27/how-to-use-vulkan-synchronization2-barriers-vkcmdpipelinebarrier2/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
