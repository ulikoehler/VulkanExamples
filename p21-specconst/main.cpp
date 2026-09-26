// SPDX-License-Identifier: CC0-1.0
//
// Specialization constants: compile-time constants baked into the
// PIPELINE, not the SPIR-V. One shader module serves both halves of
// the frame — left quad built with CELLS=4, right quad with
// CELLS=8. Same module, different pipeline, different behavior.
//
//   ./app --headless o.ppm     one frame -> PPM for check.py
//   ./app                      windowed
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) out vec2 vUV;
layout(push_constant) uniform Push { vec2 ofs; } pc;
void main() {
    vec2 p[6] = vec2[](vec2(-0.5,-0.5), vec2(0.5,-0.5), vec2(0.5,0.5),
                       vec2(-0.5,-0.5), vec2(0.5,0.5),  vec2(-0.5,0.5));
    vec2 q = p[gl_VertexIndex];
    gl_Position = vec4(q * 0.9 + pc.ofs, 0.0, 1.0);
    vUV = q + 0.5;
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
// THE specialization constant: an unnamed constant indexed by id.
// Its value arrives via VkSpecializationInfo at pipeline build —
// the SPIR-V bytecode is identical for both pipelines.
layout(constant_id = 0) const int CELLS = 4;   // default 4

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() {
    bool a = ((int(vUV.x * CELLS) ^ int(vUV.y * CELLS)) & 1) != 0;
    outColor = vec4(a ? vec3(0.9, 0.15, 0.15)
                      : vec3(0.15, 0.25, 0.9), 1.0);
}
)GLSL";

struct Push {
    float ofs[2];
};

struct Pipe {
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
};

/// The interesting bit: same `vs`/`fs` modules every call, but
/// `cells` is injected via VkSpecializationInfo -> different native
/// code inside the pipeline.
static Pipe makePipeline(vkmini::Vk& vk, vk::Format fmt,
                         vk::ShaderModule vs, vk::ShaderModule fs,
                         int cells) {
    Pipe o;
    vk::SpecializationMapEntry me{};
    me.setConstantID(0).setOffset(0).setSize(sizeof(int));
    vk::SpecializationInfo spec{};
    spec.setMapEntries(me).setDataSize(sizeof(int)).setPData(&cells);
    vk::PipelineShaderStageCreateInfo stages[2];
    stages[0].setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(vs)
        .setPName("main");
    stages[1].setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(fs)
        .setPName("main")
        .setPSpecializationInfo(&spec); // <- the bake
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
    pcr.setStageFlags(vk::ShaderStageFlagBits::eVertex)
        .setOffset(0)
        .setSize(sizeof(Push));
    o.layout = vk.device->createPipelineLayoutUnique({{}, {}, pcr});
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

static void record(vk::CommandBuffer c, vk::ImageView view,
                   vk::Extent2D extent, const Pipe& p4,
                   const Pipe& p8) {
    vk::RenderingAttachmentInfo att{};
    att.setImageView(view)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearValue{
            vk::ClearColorValue{std::array{0.05f, 0.05f, 0.08f,
                                           1.f}}});
    vk::RenderingInfo ri{};
    ri.setRenderArea({{0, 0}, extent})
        .setLayerCount(1)
        .setColorAttachments(att);
    c.beginRendering(ri);
    vk::Viewport vp{0, 0, float(extent.width), float(extent.height),
                    0.f, 1.f};
    c.setViewport(0, vp);
    vk::Rect2D sc{{0, 0}, extent};
    c.setScissor(0, sc);
    Push lo{{-0.48f, 0.f}}, hi{{0.48f, 0.f}};
    c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                   p4.pipeline.get());
    c.pushConstants(p4.layout.get(),
                    vk::ShaderStageFlagBits::eVertex, 0,
                    sizeof(Push), &lo);
    c.draw(6, 1, 0, 0);
    c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                   p8.pipeline.get());
    c.pushConstants(p8.layout.get(),
                    vk::ShaderStageFlagBits::eVertex, 0,
                    sizeof(Push), &hi);
    c.draw(6, 1, 0, 0);
    c.endRendering();
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
        win = glfwCreateWindow(512, 256, "vkmini spec", nullptr,
                             nullptr);
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
        glfwCreateWindowSurface(vk.instance.get(), win, nullptr,
                                &raw);
        surface = vk::UniqueSurfaceKHR(raw, vk.instance.get());
    }
    vk.pickPhysicalDevice();
    vk.createDevice(headless ? std::vector<const char*>{}
                             : std::vector<const char*>{
                                   VK_KHR_SWAPCHAIN_EXTENSION_NAME});

    // ONE module pair, TWO pipelines
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "s.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "s.frag");
    vk::Format fmt = headless ? vk::Format::eR8G8B8A8Unorm
                              : vk::Format::eB8G8R8A8Unorm;

    if (headless) {
        vkmini::Headless hl;
        hl.init(vk, 512, 256);
        Pipe p4 = makePipeline(vk, fmt, vs.get(), fs.get(), 4);
        Pipe p8 = makePipeline(vk, fmt, vs.get(), fs.get(), 8);
        hl.render(vk::ImageLayout::eColorAttachmentOptimal,
                  [&](vk::CommandBuffer c) {
                      record(c, hl.color.view.get(), hl.extent, p4,
                             p8);
                  });
        hl.savePpm(ppmOut);
        printf("wrote %s\n", ppmOut);
        return 0;
    }

    vkmini::Swapchain sc;
    sc.create(vk, vk.phys, surface.get(), 512, 256);
    Pipe p4 = makePipeline(vk, sc.format, vs.get(), fs.get(), 4);
    Pipe p8 = makePipeline(vk, sc.format, vs.get(), fs.get(), 8);
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
        imgBarrier(cmd[0].get(), img, vk::ImageAspectFlagBits::eColor,
                   vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eColorAttachmentOptimal,
                   vk::AccessFlags{},
                   vk::AccessFlagBits::eColorAttachmentWrite,
                   vk::PipelineStageFlagBits::eTopOfPipe,
                   vk::PipelineStageFlagBits::eColorAttachmentOutput);
        record(cmd[0].get(), sc.views[acq.value].get(), sc.extent,
               p4, p8);
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
