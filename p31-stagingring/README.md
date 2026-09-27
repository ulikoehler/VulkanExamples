# p31-stagingring — How to stream uploads through a persistent staging ring in Vulkan

Persistent staging ring: 2 fenced slots, 5 uploads, wraparound waits logged

**Tutorial post:** [How to stream uploads through a persistent staging ring in Vulkan](https://techoverflow.net/2026/09/28/how-to-stream-uploads-through-a-persistent-staging-ring-in-vulkan/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # verify the rendered output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
