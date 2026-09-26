// SPDX-License-Identifier: CC0-1.0
//
// Post 18: validation layers + VK_EXT_debug_utils. The validation
// layer catches API misuse at runtime; the debug-utils extension
// routes those messages into YOUR callback and lets you attach
// names/labels to objects and command-buffer regions. The demo
// deliberately triggers ONE validation error (fillBuffer on a
// buffer that lacks TRANSFER_DST usage) to prove the whole chain.
//
//   ./app     prints layer messages + a debug label; check.py asserts
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

/// The debug callback signature is fixed by the spec. We print a
/// compact line; check.py greps for it.
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
              vk::DebugUtilsMessageTypeFlagsEXT type,
              const vk::DebugUtilsMessengerCallbackDataEXT* data,
              void*) {
    printf("[validation:%s] %s\n",
           vk::to_string(severity).c_str(), data->pMessage);
    return VK_FALSE; // don't abort the offending call
}

int main() {
    vkmini::Vk vk;

    // ---- instance WITH the validation layer + debug_utils --------
    // The layer is a *layer* (intercepted calls); the messenger is
    // an *extension* (the reporting channel). You need both.
    vk::ApplicationInfo ai{};
    ai.setPApplicationName("vkmini-p18")
        .setApiVersion(VK_API_VERSION_1_3);
    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    const char* exts[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    vk::InstanceCreateInfo ici{};
    ici.setPApplicationInfo(&ai)
        .setPEnabledLayerNames(layers)
        .setPEnabledExtensionNames(exts);
    vk.instance = vk::createInstanceUnique(ici);

    // ---- dynamic dispatch -----------------------------------------
    // EXTENSION functions are not exported by libvulkan — they must
    // be resolved via vkGetInstanceProcAddr. Vulkan-Hpp wraps that
    // in DispatchLoaderDynamic: every EXT call takes it as a
    // trailing argument.
    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr);

    // ---- messenger: created AND destroyed around the instance ----
    vk::DebugUtilsMessengerCreateInfoEXT mi{};
    mi.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::
                              eWarning |
                          vk::DebugUtilsMessageSeverityFlagBitsEXT::
                              eError)
        .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::
                            eGeneral |
                        vk::DebugUtilsMessageTypeFlagBitsEXT::
                            eValidation |
                        vk::DebugUtilsMessageTypeFlagBitsEXT::
                            ePerformance)
        .setPfnUserCallback(debugCallback);
    auto messenger = vk.instance
                         ->createDebugUtilsMessengerEXTUnique(
                             mi, nullptr, dldi);

    vk.pickPhysicalDevice();
    vk.createDevice({});

    // ---- object naming: shows up in every future message ---------
    auto buf = vk.createBuffer(
        16,
        vk::BufferUsageFlagBits::eVertexBuffer, // NOTE: no TRANSFER_DST
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::DebugUtilsObjectNameInfoEXT nameInfo{};
    nameInfo.setObjectType(vk::ObjectType::eBuffer)
        .setObjectHandle(
            reinterpret_cast<uint64_t>(VkBuffer(buf.buf.get())))
        .setPObjectName("demo-buffer");
    vk.device->setDebugUtilsObjectNameEXT(nameInfo, dldi);
    printf("[demo] buffer named 'demo-buffer'\n");

    // ---- the deliberate mistake -----------------------------------
    // fillBuffer requires TRANSFER_DST usage; the buffer doesn't
    // have it. Without the layer: undefined behaviour. With it:
    // a precise VUID message naming 'demo-buffer'.
    vk.oneTime([&](vk::CommandBuffer c) {
        // debug labels bracket regions in tools (RenderDoc, Nsight)
        // and inside this very callback's call stack dumps.
        vk::DebugUtilsLabelEXT label{};
        label.setPLabelName("intentional-mistake-region");
        label.setColor({0.9f, 0.4f, 0.1f, 1.f});
        c.beginDebugUtilsLabelEXT(label, dldi);

        printf("[demo] calling fillBuffer without TRANSFER_DST "
               "usage...\n");
        c.fillBuffer(buf.buf.get(), 0, 4, 0xdeadbeef);

        c.endDebugUtilsLabelEXT(dldi);
    });

    // ---- cleanup --------------------------------------------------
    // Messenger is destroyed BEFORE the instance (Unique handle
    // order in vk does the instance last).
    printf("[demo] done\n");
    return 0;
}
