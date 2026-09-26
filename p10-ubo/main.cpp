// SPDX-License-Identifier: CC0-1.0
//
// Post 10: uniform buffers + MVP matrices. The indexed quad from
// post 9 gets a real 3D transform: a mat4 in a uniform buffer
// (descriptor eUniformBuffer), updated per frame through a
// persistently mapped pointer. Headless renders a fixed tilt so the
// quad projects as a trapezoid the check can measure.
//
//   ./app                 windowed: tumbling quad
//   ./app --headless o.ppm     fixed tilt -> PPM for check.py
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>
#include <chrono>
#include <cmath>

// ---- minimal mat4 (column-major, like GLSL) ------------------------
struct Mat4 {
    float m[16];
    static Mat4 identity() {
        Mat4 r{};
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.f;
        return r;
    }
    static Mat4 perspective(float fovY, float aspect, float zn,
                            float zf) {
        Mat4 r{};
        float t = 1.f / std::tan(fovY * 0.5f);
        r.m[0] = t / aspect;
        r.m[5] = -t; // flip Y: Vulkan clip Y is down
        r.m[10] = zf / (zn - zf);
        r.m[11] = -1.f;
        r.m[14] = zn * zf / (zn - zf);
        return r;
    }
    static Mat4 translate(float x, float y, float z) {
        Mat4 r = identity();
        r.m[12] = x;
        r.m[13] = y;
        r.m[14] = z;
        return r;
    }
    static Mat4 rotateX(float a) {
        Mat4 r = identity();
        r.m[5] = std::cos(a);
        r.m[6] = std::sin(a);
        r.m[9] = -std::sin(a);
        r.m[10] = std::cos(a);
        return r;
    }
    static Mat4 rotateY(float a) {
        Mat4 r = identity();
        r.m[0] = std::cos(a);
        r.m[2] = -std::sin(a);
        r.m[8] = std::sin(a);
        r.m[10] = std::cos(a);
        return r;
    }
    Mat4 operator*(const Mat4& o) const {
        Mat4 r{};
        for (int c = 0; c < 4; ++c)
            for (int rw = 0; rw < 4; ++rw)
                for (int k = 0; k < 4; ++k)
                    r.m[c * 4 + rw] += m[k * 4 + rw] * o.m[c * 4 + k];
        return r;
    }
};

struct Vertex {
    float pos[2];
    float col[3];
};
static const Vertex kVerts[] = {
    {{-0.6f, -0.6f}, {1.f, 0.f, 0.f}},
    {{0.6f, -0.6f}, {0.f, 1.f, 0.f}},
    {{0.6f, 0.6f}, {0.f, 0.f, 1.f}},
    {{-0.6f, 0.6f}, {1.f, 1.f, 0.f}},
};
static const uint16_t kIndices[] = {0, 1, 2, 2, 3, 0};

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 vColor;
// A uniform block: read-only, shared across all invocations of the
// draw — the standard home for matrices.
layout(set = 0, binding = 0) uniform Ubo { mat4 mvp; } ubo;
void main() {
    gl_Position = ubo.mvp * vec4(inPos, 0.0, 1.0);
    vColor = inColor;
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(vColor, 1.0); }
)GLSL";

static vkmini::Vk::Buffer upload(vkmini::Vk& vk, const void* data,
                                 vk::DeviceSize size,
                                 vk::BufferUsageFlags usage) {
    auto staging = vk.createBuffer(
        size, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(staging.mapped, data, size);
    auto buf = vk.createBuffer(
        size, usage | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::BufferCopy r{0, 0, size};
        c.copyBuffer(staging.buf.get(), buf.buf.get(), r);
        vk::BufferMemoryBarrier b{};
        b.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(
                vk::AccessFlagBits::eVertexAttributeRead |
                vk::AccessFlagBits::eIndexRead)
            .setBuffer(buf.buf.get())
            .setOffset(0)
            .setSize(size);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                          vk::PipelineStageFlagBits::eVertexInput,
                          {}, {}, b, {});
    });
    return buf;
}

/// Uniform buffer + descriptor for one mat4.
/// HOST_VISIBLE + persistently mapped: the CPU writes the new matrix
/// each frame with a plain memcpy — the simplest correct UBO update
/// path (GPU-visible because the memory is host-coherent).
struct Ubo {
    vkmini::Vk::Buffer buf;
    Mat4* mapped; // write here, shader reads it
    vk::UniqueDescriptorSetLayout setLayout;
    vk::UniqueDescriptorPool pool;
    vk::DescriptorSet set;
};

static Ubo makeUbo(vkmini::Vk& vk) {
    Ubo o;
    o.buf = vk.createBuffer(
        sizeof(Mat4), vk::BufferUsageFlagBits::eUniformBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    o.mapped = static_cast<Mat4*>(o.buf.mapped);

    vk::DescriptorSetLayoutBinding lb{};
    lb.setBinding(0)
        .setDescriptorType(vk::DescriptorType::eUniformBuffer)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eVertex);
    vk::DescriptorSetLayoutCreateInfo li{};
    li.setBindings(lb);
    o.setLayout = vk.device->createDescriptorSetLayoutUnique(li);
    vk::DescriptorPoolSize ps{vk::DescriptorType::eUniformBuffer, 1};
    vk::DescriptorPoolCreateInfo pi{};
    pi.setMaxSets(1).setPoolSizes(ps);
    o.pool = vk.device->createDescriptorPoolUnique(pi);
    vk::DescriptorSetAllocateInfo ai{};
    ai.setDescriptorPool(o.pool.get())
        .setSetLayouts(o.setLayout.get());
    o.set = vk.device->allocateDescriptorSets(ai).front();
    vk::DescriptorBufferInfo di{};
    di.setBuffer(o.buf.buf.get()).setOffset(0).setRange(sizeof(Mat4));
    vk::WriteDescriptorSet w{};
    w.setDstSet(o.set)
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eUniformBuffer)
        .setBufferInfo(di);
    vk.device->updateDescriptorSets(w, {});
    return o;
}

struct Pipe {
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
};

static Pipe makePipeline(vkmini::Vk& vk, vk::Format fmt,
                         vk::DescriptorSetLayout sl) {
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
    vk::PipelineLayoutCreateInfo li{};
    li.setSetLayouts(sl);
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

static void drawQuad(vk::CommandBuffer c, vk::ImageView target,
                     vk::Extent2D extent, const Pipe& p,
                     vk::DescriptorSet set, vk::Buffer vbuf,
                     vk::Buffer ibuf) {
    vk::RenderingAttachmentInfo att{};
    att.setImageView(target)
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
    c.bindPipeline(vk::PipelineBindPoint::eGraphics, p.pipeline.get());
    c.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                         p.layout.get(), 0, set, {});
    vk::Viewport vp{0, 0, float(extent.width), float(extent.height),
                    0.f, 1.f};
    c.setViewport(0, vp);
    vk::Rect2D sc{{0, 0}, extent};
    c.setScissor(0, sc);
    vk::DeviceSize off = 0;
    c.bindVertexBuffers(0, vbuf, off);
    c.bindIndexBuffer(ibuf, 0, vk::IndexType::eUint16);
    c.drawIndexed(6, 1, 0, 0, 0);
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
        win = glfwCreateWindow(640, 480, "vkmini p10", nullptr, nullptr);
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

    auto vbuf = upload(vk, kVerts, sizeof(kVerts),
                       vk::BufferUsageFlagBits::eVertexBuffer);
    auto ibuf = upload(vk, kIndices, sizeof(kIndices),
                       vk::BufferUsageFlagBits::eIndexBuffer);
    auto ubo = makeUbo(vk);

    const float aspect = 640.f / 480.f;
    if (headless) {
        vkmini::Headless hl;
        hl.init(vk, 640, 480);
        Pipe p = makePipeline(vk, vk::Format::eR8G8B8A8Unorm,
                              ubo.setLayout.get());
        // Fixed deterministic transform: tilt 45deg around X so the
        // quad projects as a trapezoid (top edge narrower).
        *ubo.mapped = Mat4::perspective(1.0f, aspect, 0.1f, 10.f) *
                      Mat4::translate(0.f, 0.f, -2.5f) *
                      Mat4::rotateX(-0.7854f);
        hl.render(vk::ImageLayout::eColorAttachmentOptimal,
                  [&](vk::CommandBuffer c) {
                      drawQuad(c, hl.color.view.get(), hl.extent, p,
                               ubo.set, vbuf.buf.get(),
                               ibuf.buf.get());
                  });
        hl.savePpm(ppmOut);
        printf("wrote %s\n", ppmOut);
        return 0;
    }

    vkmini::Swapchain sc;
    sc.create(vk, vk.phys, surface.get(), 640, 480);
    Pipe p = makePipeline(vk, sc.format, ubo.setLayout.get());
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
        // Per-frame UBO update = one memcpy into mapped memory.
        *ubo.mapped = Mat4::perspective(1.0f, aspect, 0.1f, 10.f) *
                      Mat4::translate(0.f, 0.f, -2.5f) *
                      Mat4::rotateY(t) * Mat4::rotateX(0.6f);

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
        drawQuad(cmd[0].get(), sc.views[acq.value].get(), sc.extent,
                 p, ubo.set, vbuf.buf.get(), ibuf.buf.get());
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
