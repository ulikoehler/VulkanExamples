// SPDX-License-Identifier: CC0-1.0
//
// Post 16: multi-pass post-processing. Pass 1 renders the scene
// (a checkerboard quad) into an OFFSCREEN color image; pass 2 draws
// a fullscreen triangle that samples that image and writes the
// inverted picture to the swapchain/headless target. This is the
// skeleton of every real post-fx stack (bloom, tonemap, FXAA).
//
//   ./app                 windowed
//   ./app --headless o.ppm     one frame -> PPM for check.py
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

// ---------- pass 1: the "scene" -------------------------------------
static const char* kSceneVert = R"GLSL(
#version 460
layout(location = 0) out vec2 vUV;
void main() {
    vec2 p[6] = vec2[](vec2(-0.5,-0.5), vec2(0.5,-0.5), vec2(0.5,0.5),
                       vec2(-0.5,-0.5), vec2(0.5,0.5),  vec2(-0.5,0.5));
    gl_Position = vec4(p[gl_VertexIndex] * 1.4, 0.0, 1.0);
    vUV = p[gl_VertexIndex] + 0.5;
}
)GLSL";

static const char* kSceneFrag = R"GLSL(
#version 460
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() {
    // deterministic scene: 4x4 red/blue checkerboard on green field
    bool a = ((int(vUV.x * 4) ^ int(vUV.y * 4)) & 1) != 0;
    vec3 c = a ? vec3(0.9, 0.15, 0.15) : vec3(0.15, 0.25, 0.9);
    float inQuad = step(0.15, vUV.x) * step(vUV.x, 0.85)
                 * step(0.15, vUV.y) * step(vUV.y, 0.85);
    vec3 field = vec3(0.2, 0.6, 0.25);
    outColor = vec4(mix(field, c, inQuad), 1.0);
}
)GLSL";

// ---------- pass 2: the post-process --------------------------------
static const char* kFxVert = R"GLSL(
#version 460
layout(location = 0) out vec2 vUV;
void main() {
    // fullscreen triangle — covers the viewport with one primitive
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

static const char* kFxFrag = R"GLSL(
#version 460
layout(binding = 0) uniform sampler2D sceneTex;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() {
    vec3 c = texture(sceneTex, vUV).rgb;
    outColor = vec4(vec3(1.0) - c, 1.0);   // the "post effect": invert
}
)GLSL";

struct Pipe {
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
    vk::UniqueDescriptorSetLayout dsLayout; // pass2 only
    vk::UniqueDescriptorPool pool;
    vk::UniqueDescriptorSet ds;
    vk::UniqueSampler sampler;
};

static Pipe makePipe(vkmini::Vk& vk, vk::Format fmt,
                     const char* vsrc, const char* fsrc,
                     bool textured, vk::ImageView sampleView) {
    Pipe o;
    vk::PushConstantRange* pcr = nullptr;
    if (textured) {
        vk::DescriptorSetLayoutBinding bnd{};
        bnd.setBinding(0)
            .setDescriptorType(
                vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);
        o.dsLayout =
            vk.device->createDescriptorSetLayoutUnique({{}, bnd});
        o.layout = vk.device->createPipelineLayoutUnique(
            {{}, o.dsLayout.get()});
    } else {
        o.layout = vk.device->createPipelineLayoutUnique({});
    }
    auto vs = vk.shader(vsrc, shaderc_vertex_shader, "s.vert");
    auto fs = vk.shader(fsrc, shaderc_fragment_shader, "s.frag");
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

    if (textured) {
        vk::SamplerCreateInfo sci{};
        sci.setMagFilter(vk::Filter::eNearest)
            .setMinFilter(vk::Filter::eNearest);
        o.sampler = vk.device->createSamplerUnique(sci);
        vk::DescriptorPoolSize ps{
            vk::DescriptorType::eCombinedImageSampler, 1};
        o.pool = vk.device->createDescriptorPoolUnique(
            {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1,
             ps});
        auto sets = vk.device->allocateDescriptorSetsUnique(
            {o.pool.get(), o.dsLayout.get()});
        o.ds = std::move(sets[0]);
        vk::DescriptorImageInfo di{};
        di.setSampler(o.sampler.get())
            .setImageView(sampleView)
            .setImageLayout(
                vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::WriteDescriptorSet wr{};
        wr.setDstSet(o.ds.get())
            .setDstBinding(0)
            .setDescriptorType(
                vk::DescriptorType::eCombinedImageSampler)
            .setImageInfo(di);
        vk.device->updateDescriptorSets(wr, {});
    }
    return o;
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

/// One command buffer, TWO beginRendering blocks — with a barrier in
/// between that turns the scene image from attachment into texture.
static void record(vk::CommandBuffer c, vk::ImageView sceneView,
                   vk::Image sceneImg, vk::ImageView outView,
                   vk::Extent2D extent, const Pipe& scene,
                   const Pipe& fx) {
    // ---- pass 1: scene -> offscreen -------------------------------
    vk::RenderingAttachmentInfo a1{};
    a1.setImageView(sceneView)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore) // must KEEP it
        .setClearValue(vk::ClearValue{
            vk::ClearColorValue{std::array{0.f, 0.f, 0.f, 1.f}}});
    vk::RenderingInfo r1{};
    r1.setRenderArea({{0, 0}, extent})
        .setLayerCount(1)
        .setColorAttachments(a1);
    c.beginRendering(r1);
    c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                   scene.pipeline.get());
    vk::Viewport vp{0, 0, float(extent.width), float(extent.height),
                    0.f, 1.f};
    c.setViewport(0, vp);
    vk::Rect2D sc{{0, 0}, extent};
    c.setScissor(0, sc);
    c.draw(6, 1, 0, 0);
    c.endRendering();

    // ---- the hinge: attachment -> shader-read ---------------------
    imgBarrier(c, sceneImg, vk::ImageAspectFlagBits::eColor,
               vk::ImageLayout::eColorAttachmentOptimal,
               vk::ImageLayout::eShaderReadOnlyOptimal,
               vk::AccessFlagBits::eColorAttachmentWrite,
               vk::AccessFlagBits::eShaderRead,
               vk::PipelineStageFlagBits::eColorAttachmentOutput,
               vk::PipelineStageFlagBits::eFragmentShader);

    // ---- pass 2: fullscreen triangle -> output --------------------
    vk::RenderingAttachmentInfo a2{};
    a2.setImageView(outView)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearValue{
            vk::ClearColorValue{std::array{1.f, 0.f, 1.f, 1.f}}});
    vk::RenderingInfo r2{};
    r2.setRenderArea({{0, 0}, extent})
        .setLayerCount(1)
        .setColorAttachments(a2);
    c.beginRendering(r2);
    c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                   fx.pipeline.get());
    c.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                         fx.layout.get(), 0, fx.ds.get(), {});
    c.setViewport(0, vp);
    c.setScissor(0, sc);
    c.draw(3, 1, 0, 0);
    c.endRendering();

    // restore scene layout for the next frame
    imgBarrier(c, sceneImg, vk::ImageAspectFlagBits::eColor,
               vk::ImageLayout::eShaderReadOnlyOptimal,
               vk::ImageLayout::eColorAttachmentOptimal,
               vk::AccessFlagBits::eShaderRead,
               vk::AccessFlagBits::eColorAttachmentWrite,
               vk::PipelineStageFlagBits::eFragmentShader,
               vk::PipelineStageFlagBits::eColorAttachmentOutput);
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
        win = glfwCreateWindow(512, 512, "vkmini p16", nullptr, nullptr);
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

    const vk::Format offFmt = vk::Format::eR8G8B8A8Unorm;
    auto sceneImg = vk.createImage(
        512, 512, offFmt,
        vk::ImageUsageFlagBits::eColorAttachment |
            vk::ImageUsageFlagBits::eSampled);
    // initial layout for the first render
    vk.oneTime([&](vk::CommandBuffer c) {
        imgBarrier(c, sceneImg.img.get(),
                   vk::ImageAspectFlagBits::eColor,
                   vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eColorAttachmentOptimal, {},
                   vk::AccessFlagBits::eColorAttachmentWrite,
                   vk::PipelineStageFlagBits::eTopOfPipe,
                   vk::PipelineStageFlagBits::eColorAttachmentOutput);
    });

    Pipe scene = makePipe(vk, offFmt, kSceneVert, kSceneFrag, false,
                          {});
    Pipe fx = makePipe(vk, vk::Format::eR8G8B8A8Unorm, kFxVert,
                       kFxFrag, true, sceneImg.view.get());

    if (headless) {
        vkmini::Headless hl;
        hl.init(vk, 512, 512);
        hl.render(vk::ImageLayout::eColorAttachmentOptimal,
                  [&](vk::CommandBuffer c) {
                      record(c, sceneImg.view.get(),
                             sceneImg.img.get(),
                             hl.color.view.get(), hl.extent, scene,
                             fx);
                  });
        hl.savePpm(ppmOut);
        printf("wrote %s\n", ppmOut);
        return 0;
    }

    vkmini::Swapchain sc;
    sc.create(vk, vk.phys, surface.get(), 512, 512);
    // fx pipeline for the SWAPCHAIN format — rebuild with same ds
    Pipe fx2 = makePipe(vk, sc.format, kFxVert, kFxFrag, true,
                        sceneImg.view.get());
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
        record(cmd[0].get(), sceneImg.view.get(), sceneImg.img.get(),
               sc.views[acq.value].get(), sc.extent, scene, fx2);
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
