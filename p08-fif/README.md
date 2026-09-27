# p08-fif — Vulkan frames in flight: proper fence and semaphore synchronization

Frames in flight: N command buffers, per-frame fence+semaphore pairs, acquire/present overlap

**Tutorial post:** [Vulkan frames in flight: proper fence and semaphore synchronization](https://techoverflow.net/2026/09/26/vulkan-frames-in-flight-proper-fence-and-semaphore-synchronization/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
