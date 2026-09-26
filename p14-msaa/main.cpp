// SPDX-License-Identifier: CC0-1.0
//
// Post 14: MSAA with dynamic rendering. No VkRenderPass needed —
// the resolve target is declared right in the color attachment's
// VkRenderingAttachmentInfo (resolveImageView + eAverage). The demo
// draws one big triangle at 4x samples; the check verifies that
// edge pixels are BLENDS (proof that samples were resolved, not
// just aliased).
//
//   ./app                 windowed
//   ./app --headless o.ppm     one frame -> PPM for check.py
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) out vec3 vColor;
void main() {
    vec2 p[3] = vec2[](vec2(-0.8, 0.7), vec2(0.8, 0.7),
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

/// Multisampled color attachment. vkmini::Vk::createImage hardcodes
/// e1 samples, so MSAA needs its own creation — the only difference
/// is .setSamples() on the image and NO sampler/usage extras.
static vkmini::Vk::Image makeMsaaColor(vkmini::Vk& vk, uint32_t w,
                                       uint32_t h, vk::Format fmt,
                                       vk::SampleCountFlagBits smp) {
    vkmini::Vk::Image o;
    vk::ImageCreateInfo ii{};
    ii.setImageType(vk::ImageType::e2D)
        .setFormat(fmt)
        .setExtent({w, h, 1})
        .setMipLevels(1)
        .setArrayLayers(1)
        .setSamples(smp) // <- e4 instead of e1
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage(vk::ImageUsageFlagBits::eColorAttachment |
                  vk::ImageUsageFlagBits::eTransientAttachment)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined);
    o.img = vk.device->createImageUnique(ii);
    auto req = vk.device->getImageMemoryRequirements(o.img.get());
    // transient+lazy memory where available; device-local fallback
    try {
        o.mem = vk.device->allocateMemoryUnique(
            {req.size, vk.memoryType(
                 req.memoryTypeBits,
                 vk::MemoryPropertyFlagBits::eLazilyAllocated)});
    } catch (...) {
        o.mem = vk.device->allocateMemoryUnique(
            {req.size, vk.memoryType(
                 req.memoryTypeBits,
                 vk::MemoryPropertyFlagBits::eDeviceLocal)});
    }
    vk.device->bindImageMemory(o.img.get(), o.mem.get(), 0);
    vk::ImageViewCreateInfo vi{};
    vi.setImage(o.img.get())
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(fmt)
        .setSubresourceRange(
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    o.view = vk.device->createImageViewUnique(vi);
    return o;
}

struct Pipe {
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
};

/// The only pipeline difference vs post 2: rasterizationSamples.
/// With dynamic rendering there is no subpass/sample compatibility
/// to worry about — but the pipeline sample count MUST match the
/// attachment's.
static Pipe makePipeline(vkmini::Vk& vk, vk::Format fmt,
                         vk::SampleCountFlagBits smp) {
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
    ms.setRasterizationSamples(smp) // e4 — matches the attachment
        .setSampleShadingEnable(false);
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

/// Rendering into the MSAA attachment and resolving into `resolveView`
/// in the SAME render: resolveImageView + eAverage on the attachment.
static void record(vk::CommandBuffer c, vk::ImageView msaaView,
                   vk::ImageView resolveView, vk::Extent2D extent,
                   const Pipe& p) {
    vk::RenderingAttachmentInfo att{};
    att.setImageView(msaaView)               // draw here, 4x samples
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setResolveMode(vk::ResolveModeFlagBits::eAverage)
        .setResolveImageView(resolveView)    // resolved into this
        .setResolveImageLayout(
            vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(
            vk::AttachmentStoreOp::eDontCare) // msaa data dies here
        .setClearValue(vk::ClearValue{
            vk::ClearColorValue{std::array{0.05f, 0.05f, 0.08f,
                                           1.f}}});
    vk::RenderingInfo ri{};
    ri.setRenderArea({{0, 0}, extent})
        .setLayerCount(1)
        .setColorAttachments(att);
    c.beginRendering(ri);
    c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                   p.pipeline.get());
    vk::Viewport vp{0, 0, float(extent.width), float(extent.height),
                    0.f, 1.f};
    c.setViewport(0, vp);
    vk::Rect2D sc{{0, 0}, extent};
    c.setScissor(0, sc);
    c.draw(3, 1, 0, 0);
    c.endRendering(); // resolve happens at end of rendering
}

static void imgBarrier(vk::CommandBuffer c, vk::Image img,
                       vk::ImageAspectFlags aspect,
                       vk::ImageLayout ol, vk::ImageLayout nl,
                       vk::AccessFlags src, vk::AccessFlags dst,
                       vk::PipelineStageFlags ss,
                       vk::PipelineStageFlags ds) {
    vk::ImageMemoryBarrier b{};
    b.setOldLayout(ol)
        .setNewLayout(nl)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(img)
        .setSubresourceRange({aspect, 0, 1, 0, 1})
        .setSrcAccessMask(src)
        .setDstAccessMask(dst);
    c.pipelineBarrier(ss, ds, {}, {}, {}, b);
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
        win = glfwCreateWindow(512, 512, "vkmini p14", nullptr, nullptr);
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

    constexpr auto kSamples = vk::SampleCountFlagBits::e4;

    if (headless) {
        vkmini::Headless hl;
        hl.init(vk, 512, 512);
        auto msaa = makeMsaaColor(vk, 512, 512,
                                  vk::Format::eR8G8B8A8Unorm,
                                  kSamples);
        Pipe p = makePipeline(vk, vk::Format::eR8G8B8A8Unorm,
                              kSamples);
        hl.render(vk::ImageLayout::eColorAttachmentOptimal,
                  [&](vk::CommandBuffer c) {
                      // the MSAA image needs the same transition
                      imgBarrier(
                          c, msaa.img.get(),
                          vk::ImageAspectFlagBits::eColor,
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eColorAttachmentOptimal,
                          {},
                          vk::AccessFlagBits::eColorAttachmentWrite,
                          vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::
                              eColorAttachmentOutput);
                      record(c, msaa.view.get(),
                             hl.color.view.get(), hl.extent, p);
                  });
        hl.savePpm(ppmOut);
        printf("wrote %s\n", ppmOut);
        return 0;
    }

    vkmini::Swapchain sc;
    sc.create(vk, vk.phys, surface.get(), 512, 512);
    auto msaa =
        makeMsaaColor(vk, sc.extent.width, sc.extent.height,
                      sc.format, kSamples);
    Pipe p = makePipeline(vk, sc.format, kSamples);
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
        // BOTH images go to ColorAttachmentOptimal: the msaa surface
        // and the swapchain image it resolves into.
        imgBarrier(cmd[0].get(), msaa.img.get(),
                   vk::ImageAspectFlagBits::eColor,
                   vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eColorAttachmentOptimal,
                   vk::AccessFlags{},
                   vk::AccessFlagBits::eColorAttachmentWrite,
                   vk::PipelineStageFlagBits::eTopOfPipe,
                   vk::PipelineStageFlagBits::eColorAttachmentOutput);
        imgBarrier(cmd[0].get(), img, vk::ImageAspectFlagBits::eColor,
                   vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eColorAttachmentOptimal,
                   vk::AccessFlags{},
                   vk::AccessFlagBits::eColorAttachmentWrite,
                   vk::PipelineStageFlagBits::eTopOfPipe,
                   vk::PipelineStageFlagBits::eColorAttachmentOutput);
        record(cmd[0].get(), msaa.view.get(),
               sc.views[acq.value].get(), sc.extent, p);
        imgBarrier(cmd[0].get(), img, vk::ImageAspectFlagBits::eColor,
                   vk::ImageLayout::eColorAttachmentOptimal,
                   vk::ImageLayout::ePresentSrcKHR,
                   vk::AccessFlagBits::eColorAttachmentWrite,
                   vk::AccessFlags{},
                   vk::PipelineStageFlagBits::eColorAttachmentOutput,
                   vk::PipelineStageFlagBits::eBottomOfPipe);
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
    vk.queue.waitIdle();
    sc.views.clear();
    sc.sc.reset();
    surface.reset();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
