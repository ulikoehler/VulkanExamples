// SPDX-License-Identifier: CC0-1.0
//
// Post 5: textures. A fullscreen triangle (no vertex buffer) samples
// a checkerboard texture that was generated on the CPU and uploaded
// via staging buffer -> vkCmdCopyBufferToImage -> layout transitions.
// Also the first descriptor set of the series: a combined image
// sampler binding.
//
//   ./app                 windowed: textured fullscreen triangle
//   ./app --headless o.ppm     one frame -> PPM for check.py
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

// ---------------- the texture data (CPU side) ----------------------
// 256x256 RGBA, 8x8 checkerboard of two saturated colors. Nearest
// sampling keeps texel reads exact so the check can compare ==.
constexpr uint32_t kTexW = 256, kTexH = 256, kCell = 32;
static std::vector<uint8_t> makeCheckerboard() {
    std::vector<uint8_t> px(kTexW * kTexH * 4);
    for (uint32_t y = 0; y < kTexH; ++y)
        for (uint32_t x = 0; x < kTexW; ++x) {
            bool a = ((x / kCell) + (y / kCell)) % 2 == 0;
            uint8_t* p = &px[(y * kTexW + x) * 4];
            p[0] = a ? 230 : 40;  // R
            p[1] = a ? 40 : 230;  // G
            p[2] = 60;            // B
            p[3] = 255;
        }
    return px;
}

static const char* kVert = R"GLSL(
#version 460
// Fullscreen triangle: 3 verts cover the whole viewport.
// positions: (-1,-1) (3,-1) (-1,3); uv: (0,0) (2,0) (0,2)
layout(location = 0) out vec2 vUV;
void main() {
    vUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(vUV * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
// A combined image sampler = VkImageView + VkSampler in one binding.
layout(binding = 0) uniform sampler2D tex;
void main() { outColor = texture(tex, vUV); }
)GLSL";

/// Descriptor resources for a single combined-image-sampler.
struct TexBinding {
    vk::UniqueDescriptorSetLayout setLayout;
    vk::UniqueDescriptorPool pool;
    vk::DescriptorSet set;
    vk::UniqueSampler sampler;
};

/// The new aspect of this post: turning a VkImage into something a
/// fragment shader can `texture()` on.
static TexBinding makeTextureBinding(vkmini::Vk& vk,
                                     vk::ImageView texView) {
    TexBinding o;

    // 1. Sampler: nearest filtering so the checkerboard stays crisp
    //    and the check's texel values are exact.
    vk::SamplerCreateInfo si{};
    si.setMagFilter(vk::Filter::eNearest)
        .setMinFilter(vk::Filter::eNearest)
        .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeW(vk::SamplerAddressMode::eClampToEdge);
    o.sampler = vk.device->createSamplerUnique(si);

    // 2. Descriptor set layout: "binding 0 is a combined image
    //    sampler, visible to the fragment stage".
    vk::DescriptorSetLayoutBinding lb{};
    lb.setBinding(0)
        .setDescriptorType(
            vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eFragment);
    vk::DescriptorSetLayoutCreateInfo li{};
    li.setBindings(lb);
    o.setLayout = vk.device->createDescriptorSetLayoutUnique(li);

    // 3. Pool + set (a pool is required even for a single
    //    descriptor).
    vk::DescriptorPoolSize ps{vk::DescriptorType::eCombinedImageSampler,
                              1};
    vk::DescriptorPoolCreateInfo pi{};
    pi.setMaxSets(1).setPoolSizes(ps);
    o.pool = vk.device->createDescriptorPoolUnique(pi);
    vk::DescriptorSetAllocateInfo ai{};
    ai.setDescriptorPool(o.pool.get())
        .setSetLayouts(o.setLayout.get());
    o.set = vk.device->allocateDescriptorSets(ai).front();

    // 4. Point binding 0 at {sampler, imageView, shaderRead layout}.
    vk::DescriptorImageInfo di{};
    di.setSampler(o.sampler.get())
        .setImageView(texView)
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    vk::WriteDescriptorSet w{};
    w.setDstSet(o.set)
        .setDstBinding(0)
        .setDescriptorType(
            vk::DescriptorType::eCombinedImageSampler)
        .setImageInfo(di);
    vk.device->updateDescriptorSets(w, {});
    return o;
}

/// Upload pixel data to a sampled image via staging + barriers.
/// Layout dance, in order:
///   UNDEFINED -> TRANSFER_DST  (write staging data in)
///   TRANSFER_DST -> SHADER_READ (sampler reads it)
/// Both need access-mask pairs matching the pipeline stages.
static vkmini::Vk::Image uploadTexture(vkmini::Vk& vk, uint32_t w,
                                       uint32_t h,
                                       const void* pixels,
                                       size_t size) {
    auto staging = vk.createBuffer(
        size, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(staging.mapped, pixels, size);

    auto img = vk.createImage(w, h, vk::Format::eR8G8B8A8Unorm,
                              vk::ImageUsageFlagBits::eSampled |
                                  vk::ImageUsageFlagBits::eTransferDst);
    vk::ImageSubresourceRange sr{vk::ImageAspectFlagBits::eColor, 0,
                                 1, 0, 1};
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::ImageMemoryBarrier toDst{};
        toDst.setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(img.img.get())
            .setSubresourceRange(sr)
            .setSrcAccessMask({})
            .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::eTransfer, {},
                          {}, {}, toDst);

        vk::BufferImageCopy r{};
        r.setImageSubresource(
             {vk::ImageAspectFlagBits::eColor, 0, 0, 1})
            .setImageExtent({w, h, 1});
        c.copyBufferToImage(staging.buf.get(), img.img.get(),
                            vk::ImageLayout::eTransferDstOptimal, r);

        vk::ImageMemoryBarrier toRead{};
        toRead.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
            .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(img.img.get())
            .setSubresourceRange(sr)
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                          vk::PipelineStageFlagBits::eFragmentShader,
                          {}, {}, {}, toRead);
    });
    return img;
}

struct Pipe {
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
};

/// Pipeline now takes the descriptor set layout — it becomes part of
/// the VkPipelineLayout ("the shader interface").
static Pipe makePipeline(vkmini::Vk& vk, vk::Format colorFormat,
                         vk::DescriptorSetLayout setLayout) {
    Pipe o;
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "fs.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "fs.frag");
    vk::PipelineShaderStageCreateInfo stages[2];
    stages[0].setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(vs.get())
        .setPName("main");
    stages[1].setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(fs.get())
        .setPName("main");
    vk::PipelineVertexInputStateCreateInfo vin{}; // shader-generated
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
    li.setSetLayouts(setLayout); // <- the shader interface
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

static void drawTextured(vk::CommandBuffer c, vk::ImageView target,
                         vk::Extent2D extent, const Pipe& p,
                         vk::DescriptorSet set) {
    vk::RenderingAttachmentInfo att{};
    att.setImageView(target)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearValue{
            vk::ClearColorValue{std::array{0.f, 0.f, 0.f, 1.f}}});
    vk::RenderingInfo ri{};
    ri.setRenderArea({{0, 0}, extent})
        .setLayerCount(1)
        .setColorAttachments(att);
    c.beginRendering(ri);
    c.bindPipeline(vk::PipelineBindPoint::eGraphics, p.pipeline.get());
    // bindDescriptorSets makes `set` visible as set=0 — which is why
    // the shader just says `layout(binding = 0)` (set defaults to 0).
    c.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                         p.layout.get(), 0, set, {});
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
        win = glfwCreateWindow(640, 480, "vkmini p05", nullptr, nullptr);
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

    auto pixels = makeCheckerboard();
    auto tex = uploadTexture(vk, kTexW, kTexH, pixels.data(),
                             pixels.size());
    auto binding = makeTextureBinding(vk, tex.view.get());

    if (headless) {
        vkmini::Headless hl;
        hl.init(vk, 640, 480);
        Pipe p = makePipeline(vk, vk::Format::eR8G8B8A8Unorm,
                              binding.setLayout.get());
        hl.render(vk::ImageLayout::eColorAttachmentOptimal,
                  [&](vk::CommandBuffer c) {
                      drawTextured(c, hl.color.view.get(), hl.extent,
                                   p, binding.set);
                  });
        hl.savePpm(ppmOut);
        printf("wrote %s\n", ppmOut);
        return 0;
    }

    vkmini::Swapchain sc;
    sc.create(vk, vk.phys, surface.get(), 640, 480);
    Pipe p = makePipeline(vk, sc.format, binding.setLayout.get());
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
        drawTextured(cmd[0].get(), sc.views[acq.value].get(),
                     sc.extent, p, binding.set);
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
