// SPDX-License-Identifier: CC0-1.0
//
// The capstone skeleton: a minimal complete renderer loop that
// assembles everything the series built — instance -> surface ->
// device -> swapchain -> pipeline -> frames-in-flight -> resize ->
// teardown — in the order the objects' lifetimes demand.
//
//   ./app     runs a window (use xvfb-run), renders ~90 frames,
//             resizes itself at frame 30, exits at 90.
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* kVert = R"GLSL(
#version 460
void main() {
    vec2 p = vec2((gl_VertexIndex << 1 & 2), gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0, 1);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(push_constant) uniform Push { float t; } pc;
layout(location = 0) out vec4 outColor;
void main() {
    outColor = vec4(abs(sin(pc.t)), 0.3, 0.6, 1.0);
}
)GLSL";

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* win =
        glfwCreateWindow(640, 480, "skeleton", nullptr, nullptr);

    vkmini::Vk vk;
    // glfw's surface extensions must reach the instance
    uint32_t n = 0;
    auto* gext = glfwGetRequiredInstanceExtensions(&n);
    vk.createInstance({gext, gext + n});
    VkSurfaceKHR sraw;
    if (glfwCreateWindowSurface(vk.instance.get(), win, nullptr,
                                &sraw) != VK_SUCCESS)
        throw std::runtime_error("surface failed");
    vk::UniqueSurfaceKHR surface{
        sraw, {vk.instance.get()}};
    vk.pickPhysicalDevice();
    vk.createDevice({VK_KHR_SWAPCHAIN_EXTENSION_NAME});

    vkmini::Swapchain sc;
    sc.create(vk, vk.phys, surface.get(), 640, 480);

    // pipeline — solid fullscreen triangle, animated via push const
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "s.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "s.frag");
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
    vk::PushConstantRange pcr{};
    pcr.setStageFlags(vk::ShaderStageFlagBits::eFragment)
        .setSize(sizeof(float));
    vk::PipelineLayoutCreateInfo plci{};
    plci.setPushConstantRanges(pcr);
    auto layout = vk.device->createPipelineLayoutUnique(plci);
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(sc.format);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(stages)
        .setPVertexInputState(&vin)
        .setPInputAssemblyState(&ia)
        .setPViewportState(&vp)
        .setPDynamicState(&dyn)
        .setPRasterizationState(&rs)
        .setPMultisampleState(&ms)
        .setPColorBlendState(&blend)
        .setLayout(layout.get())
        .setPNext(&rendering);
    auto pres = vk.device->createGraphicsPipelineUnique({}, gi);
    if (pres.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    auto pipe = std::move(pres.value);

    // ---- frames in flight: 2 slots ---------------------------------
    const int FIF = 2;
    auto cbufs = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::ePrimary, FIF});
    std::array<vk::UniqueFence, FIF> inflight;
    std::array<vk::UniqueSemaphore, FIF> acquired;
    std::vector<vk::UniqueSemaphore> rendered;  // per swapchain img!
    std::vector<vk::Fence> imagesInFlight(sc.images.size(),
                                          VK_NULL_HANDLE);
    for (auto& f : inflight)
        f = vk.device->createFenceUnique(
            {vk::FenceCreateFlagBits::eSignaled});
    for (auto& s : acquired)
        s = vk.device->createSemaphoreUnique({});
    for (size_t i = 0; i < sc.images.size(); ++i)
        rendered.push_back(vk.device->createSemaphoreUnique({}));

    int frame = 0;
    bool resized = false;
    while (frame < 90 && !glfwWindowShouldClose(win)) {
        glfwPollEvents();
        if (frame == 30 && !resized) {
            glfwSetWindowSize(win, 800, 500);   // force a resize
            resized = true;
        }
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        if (w == 0 || h == 0)
            continue;   // minimized — don't render

        // ---- resize check: mismatch -> recreate swapchain ---------
        if (uint32_t(w) != sc.extent.width ||
            uint32_t(h) != sc.extent.height) {
            vk.device->waitIdle();    // can't retire in-flight images
            sc.create(vk, vk.phys, surface.get(), uint32_t(w),
                      uint32_t(h));   // Unique handles free the old
            imagesInFlight.assign(sc.images.size(), VK_NULL_HANDLE);
            rendered.clear();
            for (size_t i = 0; i < sc.images.size(); ++i)
                rendered.push_back(
                    vk.device->createSemaphoreUnique({}));
            printf("[resize] swapchain -> %ux%u\n", sc.extent.width,
                   sc.extent.height);
        }

        int slot = frame % FIF;
        (void)vk.device->waitForFences(inflight[slot].get(), true,
                                       UINT64_MAX);

        uint32_t idx;
        try {
            idx = vk.device
                      ->acquireNextImageKHR(sc.sc.get(), UINT64_MAX,
                                            acquired[slot].get(),
                                            {})
                      .value;
        } catch (vk::OutOfDateKHRError&) {
            continue;   // next iteration sees the size mismatch
        }
        if (imagesInFlight[idx] != VK_NULL_HANDLE)
            (void)vk.device->waitForFences(imagesInFlight[idx], true,
                                           UINT64_MAX);
        imagesInFlight[idx] = inflight[slot].get();
        vk.device->resetFences(inflight[slot].get());

        vk::CommandBuffer c = cbufs[slot].get();
        c.reset();
        c.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::ImageMemoryBarrier toDraw{};
        toDraw.setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(sc.images[idx])
            .setSubresourceRange(
                {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})
            .setSrcAccessMask({})
            .setDstAccessMask(
                vk::AccessFlagBits::eColorAttachmentWrite);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::
                              eColorAttachmentOutput,
                          {}, {}, {}, toDraw);
        vk::RenderingAttachmentInfo att{};
        att.setImageView(sc.views[idx].get())
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setClearValue(vk::ClearValue{vk::ClearColorValue{
                std::array{0.05f, 0.05f, 0.1f, 1.f}}});
        vk::RenderingInfo ri{};
        ri.setRenderArea({{0, 0}, sc.extent})
            .setLayerCount(1)
            .setColorAttachments(att);
        c.beginRendering(ri);
        vk::Viewport v{0, 0, float(sc.extent.width),
                       float(sc.extent.height), 0.f, 1.f};
        c.setViewport(0, v);
        vk::Rect2D scr{{0, 0}, sc.extent};
        c.setScissor(0, scr);
        c.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe.get());
        float t = float(frame) * 0.05f;
        c.pushConstants(layout.get(),
                        vk::ShaderStageFlagBits::eFragment, 0,
                        sizeof(float), &t);
        c.draw(3, 1, 0, 0);
        c.endRendering();
        vk::ImageMemoryBarrier toPresent{};
        toPresent.setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(sc.images[idx])
            .setSubresourceRange(
                {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})
            .setSrcAccessMask(
                vk::AccessFlagBits::eColorAttachmentWrite)
            .setDstAccessMask({});
        c.pipelineBarrier(vk::PipelineStageFlagBits::
                              eColorAttachmentOutput,
                          vk::PipelineStageFlagBits::eBottomOfPipe,
                          {}, {}, {}, toPresent);
        c.end();
        vk::SubmitInfo si{};
        vk::Semaphore waitS = acquired[slot].get();
        vk::PipelineStageFlags waitStage =
            vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::Semaphore sigS = rendered[idx].get();
        si.setWaitSemaphores(waitS)
            .setWaitDstStageMask(waitStage)
            .setCommandBuffers(c)
            .setSignalSemaphores(sigS);
        vk.queue.submit(si, inflight[slot].get());
        try {
            vk::PresentInfoKHR pi{};
            pi.setWaitSemaphores(sigS)
                .setSwapchains(sc.sc.get())
                .setImageIndices(idx);
            (void)vk.queue.presentKHR(pi);
        } catch (vk::OutOfDateKHRError&) {
        }
        ++frame;
    }
    printf("frames rendered: %d\n", frame);

    // ---- teardown: waitIdle FIRST, then Unique dtors unwind in
    // reverse order. Swapchain+surface die before glfwDestroyWindow
    // (post-2's destroy-order bug), device last.
    vk.device->waitIdle();
    sc = {};                  // swapchain + views
    surface.reset();          // surface before the window
    glfwDestroyWindow(win);
    glfwTerminate();
    printf("shutdown clean\n");
    return 0;
}
