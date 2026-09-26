// SPDX-License-Identifier: CC0-1.0
//
// VK_EXT_memory_budget: vkGetPhysicalDeviceMemoryProperties2 +
// VkPhysicalDeviceMemoryBudgetPropertiesEXT reports per-heap
// budget + live usage — INCLUDING other processes' allocations
// on shared heaps. This is how "can I afford another 4K texture
// ring?" is answered before vkAllocateMemory fails.
//
// Demo: print heap budget/usage, allocate a few device-local
// buffers, print again -> usage grew, budget unchanged.
//
//   ./app        prints heap report; check.py parses it
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc

#include "vkmini.hpp"

static void report(vk::PhysicalDevice phys) {
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    vk::PhysicalDeviceMemoryProperties2 mp{};
    mp.pNext = &budget;
    phys.getMemoryProperties2(&mp);
    for (uint32_t i = 0; i < mp.memoryProperties.memoryHeapCount;
         ++i) {
        auto& h = mp.memoryProperties.memoryHeaps[i];
        printf("heap%u size=%lluMB usage=%lluMB budget=%lluMB "
               "flags=0x%x\n",
               i,
               (unsigned long long)(h.size >> 20),
               (unsigned long long)(budget.heapUsage[i] >> 20),
               (unsigned long long)(budget.heapBudget[i] >> 20),
               (uint32_t)h.flags);
    }
}

int main() {
    setbuf(stdout, nullptr);
    vkmini::Vk vk;
    vk.createInstance({VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME});
    vk.pickPhysicalDevice();

    // budget lives on the physical device — no device extension
    // needed for the QUERY; enabling VK_EXT_memory_budget on the
    // device only matters for per-allocation stats accuracy.
    printf("--- before allocations ---\n");
    report(vk.phys);

    vk.createDevice({VK_EXT_MEMORY_BUDGET_EXTENSION_NAME});

    // allocate 3 x 256MB device-local buffers
    std::vector<vkmini::Vk::Buffer> keep;
    for (int i = 0; i < 3; ++i)
        keep.push_back(vk.createBuffer(
            256u << 20, vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal));

    printf("--- after 3x256MB device-local ---\n");
    report(vk.phys);
    return 0;
}
