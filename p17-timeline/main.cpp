// SPDX-License-Identifier: CC0-1.0
//
// Post 17: timeline semaphores. A binary semaphore is a one-shot
// signal; a TIMELINE semaphore is a monotonically increasing
// counter. Any submit can signal a specific value; any wait (device
// or HOST) can wait for a specific value. That replaces the
// fence-per-frame juggling from post 8 and enables submit chains
// that order themselves on the GPU.
//
// The demo submits 4 chained batches — batch i waits for value i-1
// and signals value i — each doing a vkCmdFillBuffer on its own
// 4-byte slot. The HOST then waits for value 4 (one call, no fence
// at all) and reads back the buffer.
//
//   ./app     prints the readback; check.py asserts it
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

int main() {
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    // ---- device with the timelineSemaphore feature ---------------
    auto props = vk.phys.getQueueFamilyProperties();
    for (uint32_t i = 0; i < props.size(); ++i)
        if (props[i].queueFlags & vk::QueueFlagBits::eGraphics)
            vk.qfam = i;
    float prio = 1.0f;
    vk::DeviceQueueCreateInfo qi{};
    qi.setQueueFamilyIndex(vk.qfam)
        .setQueueCount(1)
        .setPQueuePriorities(&prio);
    vk::PhysicalDeviceTimelineSemaphoreFeatures tlFeat{};
    tlFeat.setTimelineSemaphore(true);
    vk::PhysicalDeviceDynamicRenderingFeatures dyn{};
    dyn.setDynamicRendering(true)
        .setPNext(&tlFeat);
    vk::DeviceCreateInfo di{};
    di.setQueueCreateInfos(qi).setPNext(&dyn);
    vk.device = vk.phys.createDeviceUnique(di);
    vk.queue = vk.device->getQueue(vk.qfam, 0);
    vk.pool = vk.device->createCommandPoolUnique(
        {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
         vk.qfam});

    // ---- the timeline semaphore: a counter, not a flag ------------
    vk::SemaphoreTypeCreateInfo typeInfo{};
    typeInfo.setSemaphoreType(vk::SemaphoreType::eTimeline)
        .setInitialValue(0);
    auto sem = vk.device->createSemaphoreUnique({{}, &typeInfo});

    // ---- output buffer: 4 slots, one per batch --------------------
    auto buf = vk.createBuffer(
        16, vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);

    auto cmds = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::ePrimary, 4});
    for (int i = 0; i < 4; ++i) {
        cmds[i]->begin(
            {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        // write (i+1)*0x11111111 into slot i
        cmds[i]->fillBuffer(buf.buf.get(), vk::DeviceSize(i) * 4, 4,
                            uint32_t(i + 1) * 0x11111111u);
        cmds[i]->end();
    }

    // ---- submit the chain ------------------------------------------
    // Timeline semaphore values ride along in VkTimelineSemaphore-
    // SubmitInfo, chained to VkSubmitInfo via pNext. Batch i waits
    // for the semaphore to reach i (i.e. batch i-1's signal) before
    // it may start — GPU-side ordering, zero CPU round-trips.
    for (uint64_t i = 0; i < 4; ++i) {
        uint64_t waitVal = i;      // wait for previous batch
        uint64_t sigVal = i + 1;   // then bump the counter
        vk::TimelineSemaphoreSubmitInfo tl{};
        if (i > 0) {
            tl.setWaitSemaphoreValues(waitVal);
        }
        tl.setSignalSemaphoreValues(sigVal);
        vk::SubmitInfo si{};
        si.setPNext(&tl);
        vk::PipelineStageFlags ws =
            vk::PipelineStageFlagBits::eTransfer;
        if (i > 0) {
            si.setWaitSemaphores(sem.get())
                .setWaitDstStageMask(ws);
        }
        si.setCommandBuffers(cmds[i].get())
            .setSignalSemaphores(sem.get());
        vk.queue.submit(si); // NOTE: no fence!
    }

    // ---- host-side wait on the VALUE, not on a fence --------------
    // One call waits for "all submits that signal <= 4" — equivalent
    // to waiting on 4 separate fences.
    uint64_t target = 4;
    vk::SemaphoreWaitInfo wi{};
    wi.setSemaphores(sem.get()).setValues(target);
    auto res = vk.device->waitSemaphores(wi, UINT64_MAX);
    if (res != vk::Result::eSuccess)
        throw std::runtime_error("timeline wait failed");

    auto* d = static_cast<const uint32_t*>(buf.mapped);
    printf("readback: %08x %08x %08x %08x\n", d[0], d[1], d[2], d[3]);
    printf("counter: %lu (expected 4)\n",
           vk.device->getSemaphoreCounterValue(sem.get()));
    return 0;
}
