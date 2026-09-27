# p29-threads — How to record Vulkan command buffers from multiple threads

Multi-threaded recording: 4 std::threads × private command pools → 4 secondaries → R/G/B/W stripes

**Tutorial post:** [How to record Vulkan command buffers from multiple threads](https://techoverflow.net/2026/09/28/how-to-record-vulkan-command-buffers-from-multiple-threads-with-per-thread-pools/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
