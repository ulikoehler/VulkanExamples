# p70_ipc — How to share GPU memory between processes with VK_KHR_external_memory_fd

`VK_KHR_external_memory_fd` — GPU memory exported as fd, imported+verified by a second process

**Tutorial post:** [How to share GPU memory between processes with VK_KHR_external_memory_fd](https://techoverflow.net/2026/09/27/how-to-share-gpu-memory-between-processes-with-vk-khr-external-memory-fd/)
on [TechOverflow](https://techoverflow.net).

## Build & run

```bash
g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
./check.py   # builds and verifies the output
```

See the [repository README](../README.md) for dependencies and the
shared `vkmini.hpp` helper.
