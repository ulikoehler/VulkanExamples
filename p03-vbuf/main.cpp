// SPDX-License-Identifier: CC0-1.0
//
// Post 3: real vertex buffers. Same RGB triangle as post 2 — but now
// the vertex data lives in a device-local VkBuffer, uploaded through
// a host-visible staging buffer + vkCmdCopyBuffer.
//
//   ./app                 windowed: RGB triangle
//   ./app --headless o.ppm     one frame -> PPM for check.py
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

/// Interleaved vertex format: the shader reads pos from location 0
/// and color from location 1; the descriptions below map this struct
/// onto those locations.
struct Vertex {
    float pos[2];
    float col[3];
};

static const Vertex kTriangle[] = {
    {{-0.8f, 0.8f}, {1.f, 0.f, 0.f}}, // bottom-left  (NDC +Y down)
    {{0.8f, 0.8f}, {0.f, 1.f, 0.f}},  // bottom-right
    {{0.0f, -0.8f}, {0.f, 0.f, 1.f}}, // top
};

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 vColor;
void main() {
    gl_Position = vec4(inPos, 0.0, 1.0);
    vColor = inColor;
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(vColor, 1.0); }
)GLSL";

/// Upload `data` to a *device-local* vertex buffer.
///
/// The pattern is always the same two buffers:
///   staging  — HOST_VISIBLE memory the CPU can memcpy() into
///   vertex   — DEVICE_LOCAL memory the GPU reads fastest
/// and a vkCmdCopyBuffer in between. Reading CPU-written memory
/// directly is possible but slower and, more importantly, many GPUs
/// place host-visible memory in uncached/limited areas.
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
        // Make the transfer write visible to vertex fetch. The
        // barrier is cheap on a one-time setup command buffer — on a
        // per-frame update it can be folded into submission order
        // (see the frames-in-flight post later in this series).
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

/// Same state block as post 2 — the only difference is that
/// vertexInput now describes a real buffer instead of being empty.
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

    // ---- the new part: describe the vertex buffer layout ---------
    // binding: "vertices come from buffer #0 in vkCmdBindVertexBuffers,
    //           each vertex is sizeof(Vertex), advance per-vertex"
    vk::VertexInputBindingDescription binding{};
    binding.setBinding(0)
        .setStride(sizeof(Vertex))
        .setInputRate(vk::VertexInputRate::eVertex);
    // attributes: "location 0 = vec2 at offsetof(pos),
    //              location 1 = vec3 at offsetof(col)"
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
    o.layout = vk.device->createPipelineLayoutUnique({});
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
                         vk::Buffer vbuf) {
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
    // Bind slot 0 to our buffer — this is what the binding
    // description's `binding = 0` refers to.
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
        win = glfwCreateWindow(640, 480, "vkmini p03", nullptr, nullptr);
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
        hl.render(vk::ImageLayout::eColorAttachmentOptimal,
                  [&](vk::CommandBuffer c) {
                      drawTriangle(c, hl.color.view.get(), hl.extent,
                                   p, vbuf.buf.get());
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

    for (int f = 0; f < 240 && !glfwWindowShouldClose(win); ++f) {
        glfwPollEvents();
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
                     sc.extent, p, vbuf.buf.get());
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
    // Free swapchain+surface before the window they reference.
    vk.queue.waitIdle();
    sc.views.clear();
    sc.sc.reset();
    surface.reset();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
