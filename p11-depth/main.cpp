// SPDX-License-Identifier: CC0-1.0
//
// Post 11: depth testing. Two overlapping quads — the NEAR one is
// drawn FIRST, the FAR one second. Without a depth attachment the
// far quad would wrongly paint over the near one; with
// eD32Sfloat + depthStencilState the near quad wins the overlap.
//
//   ./app                 windowed
//   ./app --headless o.ppm     one frame -> PPM for check.py
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

struct Vertex {
    float pos[2];
};
static const Vertex kQuad[] = {
    {{-0.4f, -0.4f}}, {{0.4f, -0.4f}}, {{0.4f, 0.4f}},
    {{0.4f, 0.4f}},  {{-0.4f, 0.4f}},  {{-0.4f, -0.4f}},
};

struct Push {
    float ofs[2];
    float z;
    float _pad;
    float col[3];
    float _pad2;
};
static_assert(sizeof(Push) == 32);

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) in vec2 inPos;
layout(location = 0) out vec3 vColor;
layout(push_constant) uniform Push {
    vec2 ofs;
    float z;
    vec3 col;
} pc;
void main() {
    gl_Position = vec4(inPos + pc.ofs, pc.z, 1.0);
    vColor = pc.col;
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(vColor, 1.0); }
)GLSL";

static vkmini::Vk::Buffer makeVertexBuffer(vkmini::Vk& vk) {
    vk::DeviceSize size = sizeof(kQuad);
    auto staging = vk.createBuffer(
        size, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(staging.mapped, kQuad, size);
    auto buf = vk.createBuffer(
        size,
        vk::BufferUsageFlagBits::eVertexBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::BufferCopy r{0, 0, size};
        c.copyBuffer(staging.buf.get(), buf.buf.get(), r);
        vk::BufferMemoryBarrier b{};
        b.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(
                vk::AccessFlagBits::eVertexAttributeRead)
            .setBuffer(buf.buf.get())
            .setOffset(0)
            .setSize(size);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                          vk::PipelineStageFlagBits::eVertexInput,
                          {}, {}, b, {});
    });
    return buf;
}

/// Depth attachment = a second image drawn alongside color. It needs
/// its own usage, format, layout (eDepthStencilAttachmentOptimal)
/// and its own barrier — plus depth state in the pipeline.
struct Depth {
    vkmini::Vk::Image img;
};

static Depth makeDepth(vkmini::Vk& vk, uint32_t w, uint32_t h) {
    Depth o;
    o.img = vk.createImage(
        w, h, vk::Format::eD32Sfloat,
        vk::ImageUsageFlagBits::eDepthStencilAttachment);
    return o;
}

struct Pipe {
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
};

/// The pipeline gains depthStencil state — mandatory once the render
/// targets include a depth attachment.
static Pipe makePipeline(vkmini::Vk& vk, vk::Format colorFmt,
                         vk::Format depthFmt) {
    Pipe o;
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "q.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "q.frag");
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
            .setOffset(0),
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
    // THE new state: test depth AND write it, nearest wins (smaller z)
    vk::PipelineDepthStencilStateCreateInfo ds{};
    ds.setDepthTestEnable(true)
        .setDepthWriteEnable(true)
        .setDepthCompareOp(vk::CompareOp::eLess);
    vk::PipelineColorBlendAttachmentState cb{};
    cb.setColorWriteMask(
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    vk::PipelineColorBlendStateCreateInfo blend{};
    blend.setAttachments(cb);
    vk::PushConstantRange pc{};
    pc.setStageFlags(vk::ShaderStageFlagBits::eVertex)
        .setOffset(0)
        .setSize(sizeof(Push));
    vk::PipelineLayoutCreateInfo li{};
    li.setPushConstantRanges(pc);
    o.layout = vk.device->createPipelineLayoutUnique(li);
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(colorFmt)
        .setDepthAttachmentFormat(depthFmt); // declare depth format
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(stages)
        .setPVertexInputState(&vin)
        .setPInputAssemblyState(&ia)
        .setPViewportState(&vp)
        .setPDynamicState(&dyn)
        .setPRasterizationState(&rs)
        .setPMultisampleState(&ms)
        .setPDepthStencilState(&ds) // wire it in
        .setPColorBlendState(&blend)
        .setLayout(o.layout.get())
        .setPNext(&rendering);
    auto res = vk.device->createGraphicsPipelineUnique({}, gi);
    if (res.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    o.pipeline = std::move(res.value);
    return o;
}

/// Record the whole scene: depth barrier + rendering with a depth
/// attachment + both quads (NEAR first, FAR second on purpose).
static void record(vk::CommandBuffer c, vk::ImageView colorView,
                   vk::ImageView depthView, vk::Extent2D extent,
                   const Pipe& p, vk::Buffer vbuf) {
    vk::RenderingAttachmentInfo color{};
    color.setImageView(colorView)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearValue{
            vk::ClearColorValue{std::array{0.05f, 0.05f, 0.08f,
                                           1.f}}});
    // The depth attachment joins the rendering info — own layout,
    // own clear value (1.0 = far plane).
    vk::RenderingAttachmentInfo depth{};
    depth.setImageView(depthView)
        .setImageLayout(
            vk::ImageLayout::eDepthStencilAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setClearValue(
            vk::ClearValue{vk::ClearDepthStencilValue{1.0f, 0}});
    vk::RenderingInfo ri{};
    ri.setRenderArea({{0, 0}, extent})
        .setLayerCount(1)
        .setColorAttachments(color)
        .setPDepthAttachment(&depth);
    c.beginRendering(ri);
    c.bindPipeline(vk::PipelineBindPoint::eGraphics, p.pipeline.get());
    vk::Viewport vp{0, 0, float(extent.width), float(extent.height),
                    0.f, 1.f};
    c.setViewport(0, vp);
    vk::Rect2D sc{{0, 0}, extent};
    c.setScissor(0, sc);
    vk::DeviceSize off = 0;
    c.bindVertexBuffers(0, vbuf, off);
    // NEAR quad FIRST (z=0.3, red, shifted left)
    Push near_{{-0.2f, 0.f}, 0.3f, 0.f, {0.9f, 0.15f, 0.15f}, 0.f};
    c.pushConstants(p.layout.get(), vk::ShaderStageFlagBits::eVertex,
                    0, sizeof(Push), &near_);
    c.draw(6, 1, 0, 0);
    // FAR quad SECOND (z=0.7, blue, shifted right) — would paint over
    // the overlap without depth testing
    Push far{{0.2f, 0.f}, 0.7f, 0.f, {0.15f, 0.25f, 0.9f}, 0.f};
    c.pushConstants(p.layout.get(), vk::ShaderStageFlagBits::eVertex,
                    0, sizeof(Push), &far);
    c.draw(6, 1, 0, 0);
    c.endRendering();
}

static void imgBarrier(vk::CommandBuffer c, vk::Image img,
                       vk::ImageAspectFlags aspect, vk::ImageLayout ol,
                       vk::ImageLayout nl, vk::AccessFlags src,
                       vk::AccessFlags dst, vk::PipelineStageFlags ss,
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
        win = glfwCreateWindow(640, 480, "vkmini p11", nullptr, nullptr);
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
    auto vbuf = makeVertexBuffer(vk);
    auto depth = makeDepth(vk, 640, 480);

    // transition the depth image once into its attachment layout —
    // per frame it's cleared by the attachment's loadOp
    vk.oneTime([&](vk::CommandBuffer c) {
        imgBarrier(c, depth.img.img.get(),
                   vk::ImageAspectFlagBits::eDepth,
                   vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eDepthStencilAttachmentOptimal,
                   {},
                   vk::AccessFlagBits::eDepthStencilAttachmentWrite,
                   vk::PipelineStageFlagBits::eTopOfPipe,
                   vk::PipelineStageFlagBits::eEarlyFragmentTests);
    });

    if (headless) {
        vkmini::Headless hl;
        hl.init(vk, 640, 480);
        Pipe p = makePipeline(vk, vk::Format::eR8G8B8A8Unorm,
                              vk::Format::eD32Sfloat);
        hl.render(vk::ImageLayout::eColorAttachmentOptimal,
                  [&](vk::CommandBuffer c) {
                      record(c, hl.color.view.get(),
                             depth.img.view.get(), hl.extent, p,
                             vbuf.buf.get());
                  });
        hl.savePpm(ppmOut);
        printf("wrote %s\n", ppmOut);
        return 0;
    }

    vkmini::Swapchain sc;
    sc.create(vk, vk.phys, surface.get(), 640, 480);
    Pipe p = makePipeline(vk, sc.format, vk::Format::eD32Sfloat);
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
        record(cmd[0].get(), sc.views[acq.value].get(),
               depth.img.view.get(), sc.extent, p, vbuf.buf.get());
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
