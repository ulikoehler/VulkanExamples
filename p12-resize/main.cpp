// SPDX-License-Identifier: CC0-1.0
//
// Post 12: swapchain recreation on resize. Window-only topic — a
// resized window invalidates the swapchain. acquireNextImageKHR then
// returns VK_ERROR_OUT_OF_DATE_KHR and presentKHR may return
// VK_SUBOPTIMAL_KHR. Handled wrongly: crash or corrupted frames.
// Handled right: recreate the swapchain against the new surface caps.
//
// The demo resizes its own window mid-run (frame 120, 640x480 ->
// 800x600) so the whole cycle is exercised automatically.
//
//   ./app     windowed, resizes itself, runs ~240 frames, exits 0
//   check.py  drives it under Xvfb and asserts the log + exit code
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>
#include <cmath>

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) out vec3 vColor;
void main() {
    vec2 p[3] = vec2[](vec2(-0.6, 0.6), vec2(0.6, 0.6),
                       vec2(0.0, -0.6));
    vec3 c[3] = vec3[](vec3(1, 0, 0), vec3(0, 1, 0),
                       vec3(0, 0, 1));
    gl_Position = vec4(p[gl_VertexIndex], 0.0, 1.0);
    vColor = c[gl_VertexIndex];
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(vColor, 1.0); }
)GLSL";

struct Pipe {
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
};

/// Built per swapchain because the color attachment format can (in
/// principle) change between surface renegotiations.
static Pipe makePipeline(vkmini::Vk& vk, vk::Format fmt) {
    Pipe o;
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "t.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "t.frag");
    vk::PipelineShaderStageCreateInfo stages[2];
    stages[0].setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(vs.get())
        .setPName("main");
    stages[1].setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(fs.get())
        .setPName("main");
    vk::PipelineVertexInputStateCreateInfo vin{};
    vk::PipelineInputAssemblyStateCreateInfo ia{};
    ia.setTopology(vk::PrimitiveTopology::eTriangleList);
    vk::PipelineViewportStateCreateInfo vp{};
    vp.setViewportCount(1).setScissorCount(1);
    std::array dynStates{vk::DynamicState::eViewport,
                         vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dyn{};
    dyn.setDynamicStates(dynStates);
    vk::PipelineRasterizationStateCreateInfo rs{};
    rs.setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eNone)
        .setFrontFace(vk::FrontFace::eCounterClockwise)
        .setLineWidth(1.0f);
    vk::PipelineMultisampleStateCreateInfo ms{};
    ms.setRasterizationSamples(vk::SampleCountFlagBits::e1);
    vk::PipelineColorBlendAttachmentState cb{};
    cb.setColorWriteMask(
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    vk::PipelineColorBlendStateCreateInfo blend{};
    blend.setAttachments(cb);
    o.layout = vk.device->createPipelineLayoutUnique({});
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(fmt);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(stages)
        .setPVertexInputState(&vin)
        .setPInputAssemblyState(&ia)
        .setPViewportState(&vp)
        .setPDynamicState(&dyn)
        .setPRasterizationState(&rs)
        .setPMultisampleState(&ms)
        .setPColorBlendState(&blend)
        .setLayout(o.layout.get())
        .setPNext(&rendering);
    auto res = vk.device->createGraphicsPipelineUnique({}, gi);
    if (res.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    o.pipeline = std::move(res.value);
    return o;
}

int main() {
    vkmini::Vk vk;
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* win =
        glfwCreateWindow(640, 480, "vkmini p12", nullptr, nullptr);
    uint32_t n = 0;
    const char** e = glfwGetRequiredInstanceExtensions(&n);
    vk.createInstance({e, e + n});
    VkSurfaceKHR raw;
    glfwCreateWindowSurface(vk.instance.get(), win, nullptr, &raw);
    auto surface = vk::UniqueSurfaceKHR(raw, vk.instance.get());
    vk.pickPhysicalDevice();
    vk.createDevice({VK_KHR_SWAPCHAIN_EXTENSION_NAME});

    vkmini::Swapchain sc;
    auto createSwapchain = [&] {
        int fw = 0, fh = 0;
        // A minimized window reports 0x0 framebuffer — wait until it
        // has a real size again instead of creating a 0-size chain.
        while (fw == 0 || fh == 0) {
            glfwGetFramebufferSize(win, &fw, &fh);
            glfwWaitEvents();
        }
        sc.create(vk, vk.phys, surface.get(), uint32_t(fw),
                  uint32_t(fh));
    };
    createSwapchain();
    Pipe p = makePipeline(vk, sc.format);

    auto cmd = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::ePrimary, 1});
    auto acqSem = vk.device->createSemaphoreUnique({});
    auto doneSem = vk.device->createSemaphoreUnique({});
    auto fence = vk.device->createFenceUnique({});

    bool needRecreate = false;
    int recreations = 0;
    for (int f = 0; f < 480 && !glfwWindowShouldClose(win); ++f) {
        glfwPollEvents();
        // Self-resize at frame 120 — a real app does this from the
        // OS/window callback; the trigger point doesn't matter.
        if (f == 120) {
            glfwSetWindowSize(win, 800, 600);
            printf("[demo] resized window to 800x600\n");
        }

        // Second trigger, independent of driver signalling: if the
        // actual framebuffer size diverges from the swapchain extent,
        // the chain is stale even if acquire/present never errored —
        // happens on some X11 WSI paths.
        {
            int fw = 0, fh = 0;
            glfwGetFramebufferSize(win, &fw, &fh);
            if (fw && fh &&
                (uint32_t(fw) != sc.extent.width ||
                 uint32_t(fh) != sc.extent.height))
                needRecreate = true;
        }

        if (needRecreate) {
            // Only after the GPU is done with the old swapchain.
            vk.queue.waitIdle();
            sc.views.clear();
            sc.sc.reset();
            createSwapchain();
            // Recreate the pipeline if the negotiated format could
            // differ — cheap enough to do unconditionally here.
            p = makePipeline(vk, sc.format);
            needRecreate = false;
            ++recreations;
            printf("[swapchain] recreated: %ux%u (#%d)\n",
                   sc.extent.width, sc.extent.height, recreations);
        }

        vk.device->resetFences(fence.get());
        // ---- acquire with resize handling -------------------------
        auto acq = vk.device->acquireNextImageKHR(
            sc.sc.get(), UINT64_MAX, acqSem.get(), {});
        if (acq.result == vk::Result::eErrorOutOfDateKHR) {
            needRecreate = true;
            continue; // nothing to draw into this frame
        }
        // SUBOPTIMAL: still usable — draw this frame, recreate next
        if (acq.result == vk::Result::eSuboptimalKHR)
            needRecreate = true;

        vk::Image img = sc.images[acq.value];
        cmd[0]->reset();
        cmd[0]->begin(
            {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::ImageSubresourceRange sr{vk::ImageAspectFlagBits::eColor,
                                     0, 1, 0, 1};
        vk::ImageMemoryBarrier toAtt{};
        toAtt.setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(img)
            .setSubresourceRange(sr)
            .setSrcAccessMask({})
            .setDstAccessMask(
                vk::AccessFlagBits::eColorAttachmentWrite);
        cmd[0]->pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, {},
            {}, toAtt);
        vk::RenderingAttachmentInfo att{};
        att.setImageView(sc.views[acq.value].get())
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setClearValue(vk::ClearValue{vk::ClearColorValue{
                std::array{0.05f, 0.05f, 0.08f, 1.f}}});
        vk::RenderingInfo ri{};
        ri.setRenderArea({{0, 0}, sc.extent})
            .setLayerCount(1)
            .setColorAttachments(att);
        cmd[0]->beginRendering(ri);
        cmd[0]->bindPipeline(vk::PipelineBindPoint::eGraphics,
                             p.pipeline.get());
        vk::Viewport vp{0, 0, float(sc.extent.width),
                        float(sc.extent.height), 0.f, 1.f};
        cmd[0]->setViewport(0, vp);
        vk::Rect2D sci{{0, 0}, sc.extent};
        cmd[0]->setScissor(0, sci);
        cmd[0]->draw(3, 1, 0, 0);
        cmd[0]->endRendering();
        vk::ImageMemoryBarrier toPresent{};
        toPresent.setOldLayout(
                     vk::ImageLayout::eColorAttachmentOptimal)
            .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(img)
            .setSubresourceRange(sr)
            .setSrcAccessMask(
                vk::AccessFlagBits::eColorAttachmentWrite)
            .setDstAccessMask({});
        cmd[0]->pipelineBarrier(
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {},
            toPresent);
        cmd[0]->end();

        vk::PipelineStageFlags waitStage =
            vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::SubmitInfo si{};
        si.setWaitSemaphores(acqSem.get())
            .setWaitDstStageMask(waitStage)
            .setCommandBuffers(cmd[0].get())
            .setSignalSemaphores(doneSem.get());
        vk.queue.submit(si, fence.get());
        vk::PresentInfoKHR pi{};
        pi.setWaitSemaphores(doneSem.get())
            .setSwapchains(sc.sc.get())
            .setImageIndices(acq.value);
        vk::Result pr = vk.queue.presentKHR(pi);
        if (pr == vk::Result::eErrorOutOfDateKHR ||
            pr == vk::Result::eSuboptimalKHR)
            needRecreate = true;
        (void)vk.device->waitForFences(fence.get(), VK_TRUE,
                                     UINT64_MAX);
    }
    printf("[demo] done: %d recreation(s)\n", recreations);
    vk.queue.waitIdle();
    sc.views.clear();
    sc.sc.reset();
    surface.reset();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
