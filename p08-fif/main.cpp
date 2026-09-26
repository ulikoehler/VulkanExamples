// SPDX-License-Identifier: CC0-1.0
//
// Post 8: frames in flight — proper CPU/GPU overlap. Every earlier
// post waited on one global fence each frame, serializing the CPU
// against the GPU. Here each frame slot owns a command buffer, an
// in-flight fence and an acquire semaphore; render-done semaphores
// are per *swapchain image*; and `imagesInFlight` prevents
// re-rendering an image the compositor is still reading.
//
// The content is a rotating colored-corner quad (rotation angle =
// 30° * frame index) so each frame is deterministic and the
// headless check can verify 4 distinct frames.
//
//   ./app                 windowed: spinning quad
//   ./app --headless       renders 4 frames offscreen -> out{0..3}.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>
#include <cmath>

constexpr int kFramesInFlight = 2;
constexpr int kFramesToRender = 4; // headless; windowed loops longer

struct Vertex {
    float pos[2];
    float col[3];
};
static const Vertex kQuad[] = {
    // CCW two triangles; NDC +Y down => -0.5 is upper half.
    {{-0.5f, -0.5f}, {1.f, 0.f, 0.f}}, // TL red
    {{0.5f, -0.5f}, {0.f, 1.f, 0.f}},  // TR green
    {{0.5f, 0.5f}, {0.f, 0.f, 1.f}},   // BR blue
    {{0.5f, 0.5f}, {0.f, 0.f, 1.f}},
    {{-0.5f, 0.5f}, {1.f, 1.f, 0.f}},  // BL yellow
    {{-0.5f, -0.5f}, {1.f, 0.f, 0.f}},
};

struct Push {
    float rot[4];   // 2x2 rotation matrix, column-major
};

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 vColor;
layout(push_constant) uniform Push { mat2 rot; } pc;
void main() {
    gl_Position = vec4(pc.rot * inPos, 0.0, 1.0);
    vColor = inColor;
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(vColor, 1.0); }
)GLSL";

/// Everything owned by one in-flight slot. The rule: a slot's
/// resources may only be reused after its fence has signaled.
struct FrameSlot {
    vk::UniqueCommandBuffer cmd;
    vk::UniqueFence fence;        // CPU waits -> slot free again
    vk::UniqueSemaphore acquired; // per-slot: GPU waits for image
};

static vkmini::Vk::Buffer makeVertexBuffer(vkmini::Vk& vk) {
    vk::DeviceSize size = sizeof(kQuad);
    auto staging = vk.createBuffer(
        size, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(staging.mapped, kQuad, size);
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

static Pipe makePipeline(vkmini::Vk& vk, vk::Format fmt) {
    Pipe o;
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "quad.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "quad.frag");
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
    vk::PushConstantRange pc{};
    pc.setStageFlags(vk::ShaderStageFlagBits::eVertex)
        .setOffset(0)
        .setSize(sizeof(Push));
    vk::PipelineLayoutCreateInfo li{};
    li.setPushConstantRanges(pc);
    o.layout = vk.device->createPipelineLayoutUnique(li);
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

static void imgBarrier(vk::CommandBuffer c, vk::Image img,
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
        .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1,
                              0, 1})
        .setSrcAccessMask(src)
        .setDstAccessMask(dst);
    c.pipelineBarrier(ss, ds, {}, {}, {}, b);
}

/// Record one frame's draw into `img/view` on the slot's command
/// buffer. `frameNo` selects the rotation angle deterministically.
static void recordDraw(vk::CommandBuffer c, vk::Image img,
                       vk::ImageView view, vk::Extent2D extent,
                       const Pipe& p, vk::Buffer vbuf, int frameNo) {
    imgBarrier(c, img, vk::ImageLayout::eUndefined,
               vk::ImageLayout::eColorAttachmentOptimal,
               vk::AccessFlags{},
               vk::AccessFlagBits::eColorAttachmentWrite,
               vk::PipelineStageFlagBits::eTopOfPipe,
               vk::PipelineStageFlagBits::eColorAttachmentOutput);
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
    c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                   p.pipeline.get());
    vk::Viewport vp{0, 0, float(extent.width), float(extent.height),
                    0.f, 1.f};
    c.setViewport(0, vp);
    vk::Rect2D sc{{0, 0}, extent};
    c.setScissor(0, sc);
    float a = frameNo * 30.f * 3.14159265f / 180.f;
    Push push{{std::cos(a), std::sin(a), -std::sin(a), std::cos(a)}};
    c.pushConstants(p.layout.get(), vk::ShaderStageFlagBits::eVertex,
                    0, sizeof(Push), &push);
    vk::DeviceSize off = 0;
    c.bindVertexBuffers(0, vbuf, off);
    c.draw(6, 1, 0, 0);
    c.endRendering();
    imgBarrier(c, img, vk::ImageLayout::eColorAttachmentOptimal,
               vk::ImageLayout::ePresentSrcKHR,
               vk::AccessFlagBits::eColorAttachmentWrite,
               vk::AccessFlags{},
               vk::PipelineStageFlagBits::eColorAttachmentOutput,
               vk::PipelineStageFlagBits::eBottomOfPipe);
}

int main(int argc, char** argv) {
    bool headless = argc > 1 && std::string(argv[1]) == "--headless";

    vkmini::Vk vk;
    GLFWwindow* win = nullptr;
    vk::UniqueSurfaceKHR surface;
    if (!headless) {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        win = glfwCreateWindow(640, 480, "vkmini p08", nullptr, nullptr);
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

    // ---- the frames-in-flight objects ------------------------------
    auto cmds = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::ePrimary,
         kFramesInFlight});
    std::vector<FrameSlot> slots(kFramesInFlight);
    for (int i = 0; i < kFramesInFlight; ++i) {
        slots[i].cmd = std::move(cmds[i]);
        // created SIGNALED: the first frame must not wait on a fence
        // that has never been submitted
        slots[i].fence = vk.device->createFenceUnique(
            {vk::FenceCreateFlagBits::eSignaled});
        slots[i].acquired = vk.device->createSemaphoreUnique({});
    }

    if (headless) {
        // A pretend "swapchain": FOUR offscreen images cycled through
        // the same slot machinery (minus semaphores — nothing to
        // acquire or present), so each rendered frame keeps its own
        // image for the PPM dump: out0..out3.ppm.
        constexpr int kImgs = 4;
        vkmini::Headless hl[kImgs];
        for (auto& h : hl) h.init(vk, 640, 480);
        Pipe p = makePipeline(vk, vk::Format::eR8G8B8A8Unorm);

        std::vector<vk::Fence> imagesInFlight(kImgs);
        for (int f = 0; f < kFramesToRender; ++f) {
            FrameSlot& s = slots[f % kFramesInFlight];
            (void)vk.device->waitForFences(s.fence.get(), VK_TRUE,
                                           UINT64_MAX);
            uint32_t imgIdx = f % kImgs; // deterministic cycle
            if (imagesInFlight[imgIdx])
                (void)vk.device->waitForFences(
                    imagesInFlight[imgIdx], VK_TRUE, UINT64_MAX);
            imagesInFlight[imgIdx] = s.fence.get();
            vk.device->resetFences(s.fence.get());

            auto& hl_ = hl[imgIdx];
            s.cmd->reset();
            s.cmd->begin({vk::CommandBufferUsageFlagBits::
                              eOneTimeSubmit});
            // recordDraw() already transitions the target
            // eUndefined -> eColorAttachment -> ePresentSrc; we only
            // append the readback transition + copy afterwards.
            recordDraw(s.cmd.get(), hl_.color.img.get(),
                       hl_.color.view.get(), hl_.extent, p,
                       vbuf.buf.get(), f);
            hl_.barrier(
                s.cmd.get(), vk::ImageLayout::ePresentSrcKHR,
                vk::ImageLayout::eTransferSrcOptimal,
                vk::AccessFlagBits::eColorAttachmentWrite,
                vk::AccessFlagBits::eTransferRead);
            vk::BufferImageCopy r{};
            r.setImageSubresource(
                 {vk::ImageAspectFlagBits::eColor, 0, 0, 1})
                .setImageExtent({640, 480, 1});
            s.cmd->copyImageToBuffer(
                hl_.color.img.get(),
                vk::ImageLayout::eTransferSrcOptimal,
                hl_.readback.buf.get(), r);
            s.cmd->end();
            vk::SubmitInfo si{};
            si.setCommandBuffers(s.cmd.get());
            vk.queue.submit(si, s.fence.get());
        }
        // drain all slots before touching mapped readback memory
        vk.queue.waitIdle();
        for (int f = 0; f < kFramesToRender; ++f) {
            char name[32];
            snprintf(name, sizeof(name), "out%d.ppm", f);
            hl[f % kImgs].savePpm(name);
            printf("wrote %s\n", name);
        }
        return 0;
    }

    // ---- windowed ---------------------------------------------------
    vkmini::Swapchain sc;
    sc.create(vk, vk.phys, surface.get(), 640, 480);
    Pipe p = makePipeline(vk, sc.format);

    // render-done semaphores live PER SWAPCHAIN IMAGE: an image can
    // still be presenting while a later frame acquires another one,
    // so a single shared "done" semaphore would be signaled by
    // multiple frames — the classic bug this structure avoids.
    std::vector<vk::UniqueSemaphore> renderDone(sc.images.size());
    for (auto& s : renderDone)
        s = vk.device->createSemaphoreUnique({});
    // fence of the frame currently using each swapchain image
    std::vector<vk::Fence> imagesInFlight(sc.images.size());

    for (int f = 0; f < 480 && !glfwWindowShouldClose(win); ++f) {
        glfwPollEvents();
        FrameSlot& s = slots[f % kFramesInFlight];
        // 1. slot free? its last submission must have finished
        (void)vk.device->waitForFences(s.fence.get(), VK_TRUE,
                                       UINT64_MAX);
        // 2. acquire an image — the semaphore signals when the
        //    compositor releases it; only the GPU waits
        auto acq = vk.device->acquireNextImageKHR(
            sc.sc.get(), UINT64_MAX, s.acquired.get(), {});
        uint32_t img = acq.value;
        // 3. if a previous in-flight frame still renders into this
        //    image, wait for it (present may already be done, the
        //    fence guards the submit, not the screen)
        if (imagesInFlight[img])
            (void)vk.device->waitForFences(imagesInFlight[img],
                                           VK_TRUE, UINT64_MAX);
        imagesInFlight[img] = s.fence.get();
        vk.device->resetFences(s.fence.get());

        s.cmd->reset();
        s.cmd->begin({vk::CommandBufferUsageFlagBits::
                          eOneTimeSubmit});
        recordDraw(s.cmd.get(), sc.images[img],
                   sc.views[img].get(), sc.extent, p,
                   vbuf.buf.get(), f);
        s.cmd->end();

        vk::PipelineStageFlags waitStage =
            vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::SubmitInfo si{};
        si.setWaitSemaphores(s.acquired.get())
            .setWaitDstStageMask(waitStage)
            .setCommandBuffers(s.cmd.get())
            .setSignalSemaphores(renderDone[img].get());
        vk.queue.submit(si, s.fence.get());
        vk::PresentInfoKHR pi{};
        pi.setWaitSemaphores(renderDone[img].get())
            .setSwapchains(sc.sc.get())
            .setImageIndices(img);
        (void)vk.queue.presentKHR(pi);
        // no fence wait here — the whole point of frames in flight:
        // the CPU immediately proceeds to prepare the next frame
    }
    // drain all slots AND images before teardown
    vk.queue.waitIdle();
    sc.views.clear();
    sc.sc.reset();
    surface.reset();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
