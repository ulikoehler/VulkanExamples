# p30-timestamp — How to measure GPU frame time in Vulkan with timestamp queries

`vkCmdWriteTimestamp` + `timestampPeriod` — real GPU-side µs, not CPU chrono

**Tutorial post:** [How to measure GPU frame time in Vulkan with timestamp queries](https://techoverflow.net/2026/09/28/how-to-measure-gpu-frame-time-in-vulkan-with-timestamp-queries/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
