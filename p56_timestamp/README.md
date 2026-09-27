# p56_timestamp — How to profile Vulkan GPU time with timestamp queries

`vkCmdWriteTimestamp` + `timestampPeriod` — measured GPU µs vs CPU submit-to-done wall time

**Tutorial post:** [How to profile Vulkan GPU time with timestamp queries](https://techoverflow.net/2026/09/27/how-to-profile-vulkan-gpu-time-with-timestamp-queries/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
