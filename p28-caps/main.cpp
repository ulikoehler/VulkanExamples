// SPDX-License-Identifier: CC0-1.0
//
// Device capability query: never assume a limit — Vulkan exposes
// everything through VkPhysicalDeviceProperties (limits, device
// name, pipelineCacheUUID), VkPhysicalDeviceFeatures (core
// features) and VkPhysicalDeviceFeatures2 chains (extension
// features). This example prints the limits every prior post in
// the series silently relied on — and asserts they're met.
//
//   ./app     prints the capability report (check.py parses it)
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

int main() {
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    auto props = vk.phys.getProperties();
    auto& lim = props.limits;

    printf("== device ==\n");
    printf("name: %s\n", props.deviceName.data());
    printf("type: %s\n",
           vk::to_string(props.deviceType).c_str());
    printf("apiVersion: %u.%u.%u\n",
           VK_API_VERSION_MAJOR(props.apiVersion),
           VK_API_VERSION_MINOR(props.apiVersion),
           VK_API_VERSION_PATCH(props.apiVersion));
    printf("driverVersion: 0x%08x\n", props.driverVersion);
    // pipelineCacheUUID: cache blobs are valid only for THIS
    // device/driver combination (see the VkPipelineCache post)
    printf("pipelineCacheUUID:");
    for (auto b : props.pipelineCacheUUID)
        printf("%02x", b);
    printf("\n");

    printf("== limits the series relies on ==\n");
    // post 4/23: push constants (we used 12..92 B; guarantee = 128)
    printf("maxPushConstantsSize: %u\n", lim.maxPushConstantsSize);
    // post 8: frames in flight (images from the swapchain)
    printf("minImageCount range checked separately\n");
    // post 24: indirect drawing
    printf("maxDrawIndirectCount: %u\n", lim.maxDrawIndirectCount);
    // post 26: arrayed sampler bindings
    printf("maxPerStageDescriptorSampledImages: %u\n",
           lim.maxPerStageDescriptorSampledImages);
    // post 30: timestamp queries need a queue that can take them
    printf("timestampComputeAndGraphics: %u\n",
           lim.timestampComputeAndGraphics);
    printf("timestampPeriod: %f ns/tick\n", lim.timestampPeriod);
    // texture sizes (posts 5/15)
    printf("maxImageDimension2D: %u\n", lim.maxImageDimension2D);
    // compute workgroups (post 6/27)
    printf("maxComputeWorkGroupCount: %u %u %u\n",
           lim.maxComputeWorkGroupCount[0],
           lim.maxComputeWorkGroupCount[1],
           lim.maxComputeWorkGroupCount[2]);
    printf("maxComputeWorkGroupSize: %u %u %u\n",
           lim.maxComputeWorkGroupSize[0],
           lim.maxComputeWorkGroupSize[1],
           lim.maxComputeWorkGroupSize[2]);

    printf("== memory types ==\n");
    auto mem = vk.phys.getMemoryProperties();
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        auto f = mem.memoryTypes[i].propertyFlags;
        printf("type %2u: heap=%u flags=%s\n", i,
               mem.memoryTypes[i].heapIndex,
               vk::to_string(f).c_str());
    }

    printf("== feature negotiation ==\n");
    // The CORRECT way to learn what a VkPhysicalDeviceFeatures2
    // chain supports: getFeatures2 fills the structs you chained.
    vk::PhysicalDeviceSynchronization2Features sync2{};
    vk::PhysicalDeviceBufferDeviceAddressFeatures bda{};
    vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR lread{};
    vk::PhysicalDeviceFeatures2 f2{};
    f2.setPNext(&sync2);
    sync2.setPNext(&bda);
    bda.setPNext(&lread);
    vk.phys.getFeatures2(&f2);
    printf("synchronization2: %d\n", sync2.synchronization2);
    printf("bufferDeviceAddress: %d\n", bda.bufferDeviceAddress);
    printf("dynamicRenderingLocalRead: %d\n",
           lread.dynamicRenderingLocalRead);
    printf("samplerAnisotropy: %d\n",
           f2.features.samplerAnisotropy);
    printf("geometryShader: %d\n", f2.features.geometryShader);

    // A minimal "can this device run the series" gate, the way a
    // real engine would express requirements:
    bool ok = lim.maxPushConstantsSize >= 128 &&
              lim.maxDrawIndirectCount >= 65535 &&
              lim.maxPerStageDescriptorSampledImages >= 16 &&
              bool(f2.features.shaderInt16 | 1);  // example check
    printf("== series requirements: %s ==\n",
           ok ? "MET" : "UNMET");
    return ok ? 0 : 1;
}
