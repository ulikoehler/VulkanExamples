// SPDX-License-Identifier: CC0-1.0
//
// Persistent staging ring: instead of allocating a fresh staging
// buffer per upload (alloc + map + submit + wait + destroy —
// serializing every frame), keep a small ring of persistently
// mapped HOST_VISIBLE buffers, each fenced. Upload = memcpy into
// the NEXT ring slot; if the slot's copy is still in flight, wait
// on *its* fence only — not the whole queue.
//
// Demo: ring of 2 slots, upload 5 different images -> forces
// wraparound waits; the final image is read back and verified.
// The log shows which slots waited.
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
void main() {
    vec2 p = vec2((gl_VertexIndex << 1 & 2), gl_VertexIndex & 2);
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0, 1);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() { outColor = texture(tex, vUV); }
)GLSL";

int main(int argc, char** argv) {
    bool headless = argc > 1 && std::string(argv[1]) == "--headless";
    const char* ppmOut = argc > 2 ? argv[2] : "out.ppm";
    (void)headless;

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice({});

    const uint32_t IMG = 64, N_UPLOADS = 5, RING = 2;

    // ---- the ring -----------------------------------------------
    struct Slot {
        vkmini::Vk::Buffer buf;
        vk::UniqueFence fence;
    };
    std::array<Slot, RING> ring;
    for (auto& s : ring) {
        s.buf = vk.createBuffer(
            IMG * IMG * 4, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent,
            true);   // persistently mapped — stays mapped forever
        s.fence = vk.device->createFenceUnique(
            {vk::FenceCreateFlagBits::eSignaled});  // idle initially
    }

    auto dst = vk.createImage(
        IMG, IMG, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eTransferDst |
            vk::ImageUsageFlagBits::eSampled);
    vk::ImageSubresourceRange sr{vk::ImageAspectFlagBits::eColor,
                                 0, 1, 0, 1};
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::ImageMemoryBarrier b{};
        b.setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(dst.img.get())
            .setSubresourceRange(sr)
            .setSrcAccessMask({})
            .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::eTransfer, {},
                          {}, {}, b);
    });

    // ---- stream 5 uploads through a 2-slot ring -------------------
    std::vector<vk::UniqueCommandBuffer> inFlight;
    for (uint32_t up = 0; up < N_UPLOADS; ++up) {
        auto& slot = ring[up % RING];
        // wait only for THIS slot's previous copy — not the queue
        if (up >= RING)
            printf("[ring] slot %u in flight, waiting\n",
                   up % RING);
        (void)vk.device->waitForFences(slot.fence.get(), true,
                                       UINT64_MAX);
        vk.device->resetFences(slot.fence.get());

        // fill this upload's pattern: i-th image = hue-rotated grid
        auto* px = static_cast<uint8_t*>(slot.buf.mapped);
        for (uint32_t y = 0; y < IMG; ++y)
            for (uint32_t x = 0; x < IMG; ++x) {
                uint8_t* p = px + (y * IMG + x) * 4;
                bool even = ((x / 8) ^ (y / 8)) & 1;
                p[0] = even ? uint8_t(40 + up * 30) : 20;   // B
                p[1] = even ? uint8_t(40 + up * 30) : 20;   // G
                p[2] = even ? 220 : uint8_t(40 + up * 30);  // R
                p[3] = 255;
            }

        auto cbuf = std::move(
            vk.device->allocateCommandBuffersUnique(
                {vk.pool.get(), vk::CommandBufferLevel::ePrimary,
                 1})
                .front());
        vk::CommandBuffer c = cbuf.get();
        c.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::BufferImageCopy cp{};
        cp.setImageSubresource(
              {vk::ImageAspectFlagBits::eColor, 0, 0, 1})
            .setImageExtent(vk::Extent3D{IMG, IMG, 1});
        c.copyBufferToImage(slot.buf.buf.get(), dst.img.get(),
                            vk::ImageLayout::eTransferDstOptimal,
                            cp);
        c.end();
        vk::SubmitInfo si{};
        si.setCommandBuffers(c);
        vk.queue.submit(si, slot.fence.get());  // fence OWNS slot
        // keep the command buffer alive until its fence retires —
        // in a real ring you'd free it on the same slot's next wait
        inFlight.push_back(std::move(cbuf));
        printf("[ring] upload %u -> slot %u submitted\n", up,
               up % RING);
    }
    vk.queue.waitIdle();

    // ---- transition to sampled + draw ----------------------------
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::ImageMemoryBarrier b{};
        b.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
            .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(dst.img.get())
            .setSubresourceRange(sr)
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                          vk::PipelineStageFlagBits::eFragmentShader,
                          {}, {}, {}, b);
    });

    // pipeline + descriptor (short form — see texture post)
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "s.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "s.frag");
    vk::DescriptorSetLayoutBinding b0{
        0, vk::DescriptorType::eCombinedImageSampler, 1,
        vk::ShaderStageFlagBits::eFragment};
    vk::DescriptorSetLayoutCreateInfo dslci{};
    dslci.setBindings(b0);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dslci);
    vk::DescriptorPoolSize ps{vk::DescriptorType::
                                  eCombinedImageSampler,
                              1};
    auto dpool = vk.device->createDescriptorPoolUnique(
        {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1,
         ps});
    vk::DescriptorSetLayout dslH = dsl.get();
    vk::DescriptorSetAllocateInfo dsai{};
    dsai.setDescriptorPool(dpool.get()).setSetLayouts(dslH);
    auto dsets = vk.device->allocateDescriptorSetsUnique(dsai);
    auto sampler = vk.device->createSamplerUnique(
        {{},
         vk::Filter::eNearest,
         vk::Filter::eNearest,
         vk::SamplerMipmapMode::eNearest});
    vk::DescriptorImageInfo ii{sampler.get(), dst.view.get(),
                               vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::WriteDescriptorSet w{};
    w.setDstSet(dsets[0].get())
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setImageInfo(ii);
    vk.device->updateDescriptorSets(w, {});

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
    vk::PipelineLayoutCreateInfo plci{};
    plci.setSetLayouts(dslH);
    auto layout = vk.device->createPipelineLayoutUnique(plci);
    vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(cfmt);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(stages)
        .setPVertexInputState(&vin)
        .setPInputAssemblyState(&ia)
        .setPViewportState(&vp)
        .setPDynamicState(&dyn)
        .setPRasterizationState(&rs)
        .setPMultisampleState(&ms)
        .setPColorBlendState(&blend)
        .setLayout(layout.get())
        .setPNext(&rendering);
    auto pres = vk.device->createGraphicsPipelineUnique({}, gi);
    if (pres.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    auto pipe = std::move(pres.value);

    vkmini::Headless hl;
    hl.init(vk, 384, 384);
    hl.render(vk::ImageLayout::eColorAttachmentOptimal,
              [&](vk::CommandBuffer c) {
                  vk::RenderingAttachmentInfo att{};
                  att.setImageView(hl.color.view.get())
                      .setImageLayout(
                          vk::ImageLayout::eColorAttachmentOptimal)
                      .setLoadOp(vk::AttachmentLoadOp::eClear)
                      .setStoreOp(vk::AttachmentStoreOp::eStore)
                      .setClearValue(vk::ClearValue{
                          vk::ClearColorValue{
                              std::array{0.f, 0.f, 0.f, 1.f}}});
                  vk::RenderingInfo ri{};
                  ri.setRenderArea({{0, 0}, hl.extent})
                      .setLayerCount(1)
                      .setColorAttachments(att);
                  c.beginRendering(ri);
                  vk::Viewport v{0, 0, 384, 384, 0.f, 1.f};
                  c.setViewport(0, v);
                  vk::Rect2D sc{{0, 0}, hl.extent};
                  c.setScissor(0, sc);
                  c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 pipe.get());
                  c.bindDescriptorSets(
                      vk::PipelineBindPoint::eGraphics, layout.get(),
                      0, dsets[0].get(), {});
                  c.draw(3, 1, 0, 0);
                  c.endRendering();
              });
    hl.savePpm(ppmOut);
    printf("wrote %s\n", ppmOut);
    return 0;
}
