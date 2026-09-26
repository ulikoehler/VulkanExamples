// SPDX-License-Identifier: CC0-1.0
//
// Post 2: the first triangle — a graphics pipeline + dynamic
// rendering, no VkRenderPass, no vertex buffers (the vertex shader
// synthesizes the three vertices from gl_VertexIndex).
//
//   ./app                 windowed: static triangle
//   ./app --headless o.ppm     one frame -> PPM for check.py
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* kVert = R"GLSL(
#version 460
// Triangle from the vertex index only — nothing to upload.
// Vulkan NDC has +Y pointing *down*, so this is upside-down
// relative to OpenGL conventions.
layout(location = 0) out vec3 vColor;
void main() {
    vec2 p[3] = vec2[](vec2(-0.8, 0.8), vec2(0.8, 0.8),
                       vec2(0.0, -0.8));
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

/// A pipeline AND its layout: the layout must stay alive for as long
/// as the pipeline is bound, so we hand both back together.
struct Pipe {
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
};

/// The aspect of this post: assembling a vk::Pipeline for dynamic
/// rendering. Everything stateful (viewport, blend, topology) is
/// baked in here — no render-pass objects involved.
static Pipe makePipeline(vkmini::Vk& vk, vk::Format colorFormat) {
    Pipe o;
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "tri.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "tri.frag");
    // NOTE: the shader modules only need to outlive the
    // createGraphicsPipelines call itself — `vs`/`fs` are scoped to
    // this function and destroyed right after.

    vk::PipelineShaderStageCreateInfo stages[2];
    stages[0].setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(vs.get())
        .setPName("main");
    stages[1].setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(fs.get())
        .setPName("main");

    vk::PipelineVertexInputStateCreateInfo vin{}; // empty: shader-gen
    vk::PipelineInputAssemblyStateCreateInfo ia{};
    ia.setTopology(vk::PrimitiveTopology::eTriangleList);
    vk::PipelineViewportStateCreateInfo vp{};
    vp.setViewportCount(1).setScissorCount(1); // set dynamically below
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
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)
        .setBlendEnable(false);
    vk::PipelineColorBlendStateCreateInfo blend{};
    blend.setAttachments(cb);

    vk::PipelineLayoutCreateInfo li{};
    o.layout = vk.device->createPipelineLayoutUnique(li);

    // The only render-pass-ish bit left for dynamic rendering: tell
    // the pipeline which attachment format(s) it will render into.
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(colorFormat);

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
        throw std::runtime_error("pipeline creation failed");
    o.pipeline = std::move(res.value);
    return o;
}

/// Record one dynamic-rendering draw of the triangle into `target`.
static void drawTriangle(vk::CommandBuffer c, vk::ImageView target,
                         vk::Extent2D extent, const Pipe& p) {
    vk::RenderingAttachmentInfo att{};
    att.setImageView(target)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearValue{
            vk::ClearColorValue{std::array{0.05f, 0.05f, 0.08f, 1.f}}});
    vk::RenderingInfo ri{};
    ri.setRenderArea({{0, 0}, extent})
        .setLayerCount(1)
        .setColorAttachments(att);
    c.beginRendering(ri);
    c.bindPipeline(vk::PipelineBindPoint::eGraphics, p.pipeline.get());
    vk::Viewport vp{0, 0, float(extent.width), float(extent.height),
                    0.f, 1.f};
    c.setViewport(0, vp);
    vk::Rect2D sc{{0, 0}, extent};
    c.setScissor(0, sc);
    c.draw(3, 1, 0, 0);
    c.endRendering();
}

int main(int argc, char** argv) {
    bool headless = argc > 1 && std::string(argv[1]) == "--headless";
    const char* ppmOut = argc > 2 ? argv[2] : "out.ppm";

    vkmini::Vk vk;
    GLFWwindow* win = nullptr;
    vk::UniqueSurfaceKHR surface;

    if (!headless) {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        win = glfwCreateWindow(640, 480, "vkmini p02", nullptr, nullptr);
    }
    std::vector<const char*> instExts;
    if (!headless) {
        uint32_t n = 0;
        const char** e = glfwGetRequiredInstanceExtensions(&n);
        instExts.assign(e, e + n);
    }
    vk.createInstance(instExts);
    if (!headless) {
        VkSurfaceKHR raw;
        glfwCreateWindowSurface(vk.instance.get(), win, nullptr, &raw);
        surface = vk::UniqueSurfaceKHR(raw, vk.instance.get());
    }
    vk.pickPhysicalDevice();
    vk.createDevice(headless ? std::vector<const char*>{}
                             : std::vector<const char*>{
                                   VK_KHR_SWAPCHAIN_EXTENSION_NAME});

    if (headless) {
        vkmini::Headless hl;
        hl.init(vk, 640, 480);
        Pipe p = makePipeline(vk, vk::Format::eR8G8B8A8Unorm);
        hl.render(vk::ImageLayout::eColorAttachmentOptimal,
                  [&](vk::CommandBuffer c) {
                      drawTriangle(c, hl.color.view.get(), hl.extent, p);
                  });
        hl.savePpm(ppmOut);
        printf("wrote %s\n", ppmOut);
        return 0;
    }

    // ---- windowed -------------------------------------------------------
    vkmini::Swapchain sc;
    sc.create(vk, vk.phys, surface.get(), 640, 480);
    Pipe p = makePipeline(vk, sc.format);

    auto cmd = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::ePrimary, 1});
    auto acqSem = vk.device->createSemaphoreUnique({});
    auto doneSem = vk.device->createSemaphoreUnique({});
    auto fence = vk.device->createFenceUnique({});

    for (int f = 0; f < 240 && !glfwWindowShouldClose(win); ++f) {
        glfwPollEvents();
        vk.device->resetFences(fence.get());
        auto acq = vk.device->acquireNextImageKHR(
            sc.sc.get(), UINT64_MAX, acqSem.get(), {});
        vk::Image img = sc.images[acq.value];

        cmd[0]->reset();
        cmd[0]->begin(
            {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        // UNDEFINED -> COLOR_ATTACHMENT -> present, dynamic rendering
        // in between.
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
        drawTriangle(cmd[0].get(), sc.views[acq.value].get(),
                     sc.extent, p);
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
        (void)vk.queue.presentKHR(pi);
        (void)vk.device->waitForFences(fence.get(), VK_TRUE,
                                     UINT64_MAX);
    }
    // Destruction order matters: the swapchain and surface must be
    // destroyed *before* the GLFW window — the window-system surface
    // (wl_surface/xcb window) they reference must still exist when
    // the driver's WSI code tears them down.
    vk.queue.waitIdle();
    sc.views.clear();
    sc.sc.reset();
    surface.reset();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
