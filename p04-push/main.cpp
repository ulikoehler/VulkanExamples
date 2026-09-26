// SPDX-License-Identifier: CC0-1.0
//
// Post 4: push constants — tiny per-draw uniforms sent inline with
// the command buffer, no descriptors needed. The vertex-buffer
// triangle from post 3 gets offset/scale/tint pushed per frame:
// animated in windowed mode, fixed values in headless mode so the
// check can assert exact positions.
//
//   ./app                 windowed: bouncing/pulsing triangle
//   ./app --headless o.ppm     one frame -> PPM for check.py
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>
#include <chrono>
#include <cmath>

struct Vertex {
    float pos[2];
    float col[3];
};

static const Vertex kTriangle[] = {
    {{-0.8f, 0.8f}, {1.f, 0.f, 0.f}},
    {{0.8f, 0.8f}, {0.f, 1.f, 0.f}},
    {{0.0f, -0.8f}, {0.f, 0.f, 1.f}},
};

/// Must match the GLSL push_constant block *exactly*, including
/// alignment: vec3 is 16-byte aligned in uniform-style layouts, so
/// after vec2(8B) + float(4B) one explicit padding float is needed
/// before tint lands at offset 16.
struct Push {
    float offset[2];
    float scale;
    float _pad;   // vec3 needs 16-byte alignment -> tint at offset 16
    float tint[3];
    float _pad2;  // struct size padded to 16 like the GLSL block
};
static_assert(sizeof(Push) == 32);

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 vColor;
// Push constants live inside the command buffer itself — the
// cheapest possible way to feed a shader a few floats per draw.
layout(push_constant) uniform Push {
    vec2 offset;
    float scale;
    vec3 tint;
} pc;
void main() {
    gl_Position = vec4(inPos * pc.scale + pc.offset, 0.0, 1.0);
    vColor = inColor * pc.tint;
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(vColor, 1.0); }
)GLSL";

static vkmini::Vk::Buffer makeVertexBuffer(vkmini::Vk& vk,
                                           const void* data,
                                           vk::DeviceSize size) {
    auto staging = vk.createBuffer(
        size, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(staging.mapped, data, size);
    auto vbuf = vk.createBuffer(
        size,
        vk::BufferUsageFlagBits::eVertexBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::BufferCopy r{0, 0, size};
        c.copyBuffer(staging.buf.get(), vbuf.buf.get(), r);
        vk::BufferMemoryBarrier b{};
        b.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(
                vk::AccessFlagBits::eVertexAttributeRead)
            .setBuffer(vbuf.buf.get())
            .setOffset(0)
            .setSize(size);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                          vk::PipelineStageFlagBits::eVertexInput,
                          {}, {}, b, {});
    });
    return vbuf;
}

struct Pipe {
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
};

static Pipe makePipeline(vkmini::Vk& vk, vk::Format colorFormat) {
    Pipe o;
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "tri.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "tri.frag");
    vk::PipelineShaderStageCreateInfo stages[2];
    stages[0].setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(vs.get())
        .setPName("main");
    stages[1].setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(fs.get())
        .setPName("main");
    vk::VertexInputBindingDescription binding{};
    binding.setBinding(0)
        .setStride(sizeof(Vertex))
        .setInputRate(vk::VertexInputRate::eVertex);
    std::array attrs{
        vk::VertexInputAttributeDescription{}
            .setLocation(0)
            .setBinding(0)
            .setFormat(vk::Format::eR32G32Sfloat)
            .setOffset(offsetof(Vertex, pos)),
        vk::VertexInputAttributeDescription{}
            .setLocation(1)
            .setBinding(0)
            .setFormat(vk::Format::eR32G32B32Sfloat)
            .setOffset(offsetof(Vertex, col)),
    };
    vk::PipelineVertexInputStateCreateInfo vin{};
    vin.setVertexBindingDescriptions(binding)
        .setVertexAttributeDescriptions(attrs);
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

    // ---- the new part: declare the push-constant range in the
    // pipeline layout. `stageFlags` says which stages may see it;
    // 128B is the guaranteed minimum total size.
    vk::PushConstantRange pc{};
    pc.setStageFlags(vk::ShaderStageFlagBits::eVertex)
        .setOffset(0)
        .setSize(sizeof(Push));
    vk::PipelineLayoutCreateInfo li{};
    li.setPushConstantRanges(pc);
    o.layout = vk.device->createPipelineLayoutUnique(li);

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

static void drawTriangle(vk::CommandBuffer c, vk::ImageView target,
                         vk::Extent2D extent, const Pipe& p,
                         vk::Buffer vbuf, const Push& push) {
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
    // The actual push: bytes are copied INTO the command buffer at
    // record time — the shader sees them as `pc.*`.
    c.pushConstants(p.layout.get(), vk::ShaderStageFlagBits::eVertex,
                    0, sizeof(Push), &push);
    vk::DeviceSize off = 0;
    c.bindVertexBuffers(0, vbuf, off);
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
        win = glfwCreateWindow(640, 480, "vkmini p04", nullptr, nullptr);
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
    auto vbuf = makeVertexBuffer(vk, kTriangle, sizeof(kTriangle));

    if (headless) {
        vkmini::Headless hl;
        hl.init(vk, 640, 480);
        Pipe p = makePipeline(vk, vk::Format::eR8G8B8A8Unorm);
        // Deterministic transform for the check: shift right/down,
        // shrink to 60%, tint everything slightly magenta.
        Push push{{0.15f, 0.10f}, 0.60f, 0.f, {0.9f, 0.7f, 0.9f}};
        hl.render(vk::ImageLayout::eColorAttachmentOptimal,
                  [&](vk::CommandBuffer c) {
                      drawTriangle(c, hl.color.view.get(), hl.extent,
                                   p, vbuf.buf.get(), push);
                  });
        hl.savePpm(ppmOut);
        printf("wrote %s\n", ppmOut);
        return 0;
    }

    vkmini::Swapchain sc;
    sc.create(vk, vk.phys, surface.get(), 640, 480);
    Pipe p = makePipeline(vk, sc.format);
    auto cmd = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::ePrimary, 1});
    auto acqSem = vk.device->createSemaphoreUnique({});
    auto doneSem = vk.device->createSemaphoreUnique({});
    auto fence = vk.device->createFenceUnique({});

    const auto t0 = std::chrono::steady_clock::now();
    for (int f = 0; f < 480 && !glfwWindowShouldClose(win); ++f) {
        glfwPollEvents();
        float t = std::chrono::duration<float>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
        // Animate purely on the CPU: circle the triangle, pulse its
        // size and hue. Only 32B of push constants per frame.
        Push push{
            {0.35f * std::cos(2.f * t), 0.30f * std::sin(2.f * t)},
            0.45f + 0.25f * std::sin(3.f * t),
            0.f,
            {0.6f + 0.4f * std::sin(t), 0.6f + 0.4f * std::sin(t + 2),
             0.6f + 0.4f * std::sin(t + 4)}};

        vk.device->resetFences(fence.get());
        auto acq = vk.device->acquireNextImageKHR(
            sc.sc.get(), UINT64_MAX, acqSem.get(), {});
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
        drawTriangle(cmd[0].get(), sc.views[acq.value].get(),
                     sc.extent, p, vbuf.buf.get(), push);
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
    vk.queue.waitIdle();
    sc.views.clear();
    sc.sc.reset();
    surface.reset();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
