// SPDX-License-Identifier: CC0-1.0
//
// Physical device selection: vkEnumeratePhysicalDevices returns
// EVERY Vulkan device in the system — on a laptop that's typically
// the iGPU (INTEGRATED_GPU), the dGPU (DISCRETE_GPU) and a software
// rasterizer (CPU / llvmpipe). Vulkan never picks for you; the app
// must score candidates against what it actually needs.
//
// Demo: list every device with type/vendor/limits, score by
//   - required feature gates (graphics queue, swapchain ext)
//   - preference weight: discrete > integrated > cpu
//   - VRAM as tie-breaker
// then create a device on the winner.
//
//   ./app            # auto-pick (discrete wins here)
//   ./app --cpu      # force software rasterizer
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* typeName(vk::PhysicalDeviceType t) {
    switch (t) {
    case vk::PhysicalDeviceType::eDiscreteGpu:   return "DISCRETE_GPU (dGPU)";
    case vk::PhysicalDeviceType::eIntegratedGpu: return "INTEGRATED_GPU (iGPU)";
    case vk::PhysicalDeviceType::eVirtualGpu:    return "VIRTUAL_GPU";
    case vk::PhysicalDeviceType::eCpu:           return "CPU (software rasterizer)";
    default: return "other";
    }
}

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    bool forceCpu = argc > 1 && std::string(argv[1]) == "--cpu";

    vkmini::Vk vk;
    vk.createInstance({});
    auto devs = vk.instance->enumeratePhysicalDevices();
    printf("%zu Vulkan device(s) found:\n", devs.size());

    struct Cand {
        vk::PhysicalDevice phys;
        int score;
        size_t vram;
        const char* why;
    };
    Cand best{};
    int bestScore = -1;

    for (auto phys : devs) {
        auto props = phys.getProperties();
        auto mem = phys.getMemoryProperties();
        size_t vram = 0;
        for (uint32_t i = 0; i < mem.memoryHeapCount; ++i)
            if (mem.memoryHeaps[i].flags &
                vk::MemoryHeapFlagBits::eDeviceLocal)
                vram += mem.memoryHeaps[i].size;

        // --- gates: what does the app actually need? -------------
        auto qf = phys.getQueueFamilyProperties();
        bool gfx = false;
        for (auto& q : qf)
            gfx |= !!(q.queueFlags & vk::QueueFlagBits::eGraphics);
        bool swap = false;
        for (auto& e : phys.enumerateDeviceExtensionProperties())
            if (!strcmp(e.extensionName, "VK_KHR_swapchain"))
                swap = true;

        int score = 0;
        const char* why = "ok";
        if (!gfx)       { score = -1; why = "no graphics queue"; }
        else if (!swap) { score = -1; why = "no VK_KHR_swapchain"; }
        else switch (props.deviceType) {
            case vk::PhysicalDeviceType::eDiscreteGpu:
                score = 300; why = "discrete GPU"; break;
            case vk::PhysicalDeviceType::eIntegratedGpu:
                score = 200; why = "integrated GPU"; break;
            case vk::PhysicalDeviceType::eCpu:
                score = 100; why = "software fallback"; break;
            default:
                score = 50;  why = "unclassified"; break;
            }
        if (forceCpu && props.deviceType !=
                            vk::PhysicalDeviceType::eCpu) {
            score = -1; why = "skipped (--cpu)";
        }
        printf("  %-40s %-28s vendor=%04x device=%04x "
               "vram=%zuMB -> %s\n",
               props.deviceName.data(), typeName(props.deviceType),
               props.vendorID, props.deviceID,
               vram / (1 << 20), why);
        if (score >= 0) {
            // tie-breaker inside the same class: more VRAM wins
            int total = score * 100000
                        + (int)std::min<size_t>(vram >> 20, 99999);
            if (total > bestScore)
                best = {phys, total, vram, why}, bestScore = total;
        }
    }

    if (bestScore < 0) {
        printf("SKIP: no usable device\n");
        return 0;
    }
    vk.phys = best.phys;
    vk.createDevice({});
    auto p = best.phys.getProperties();
    printf("selected: %s — %s, %zuMB device-local, api %u.%u\n",
           p.deviceName.data(), best.why, best.vram >> 20,
           VK_API_VERSION_MAJOR(p.apiVersion),
           VK_API_VERSION_MINOR(p.apiVersion));
    return 0;
}
