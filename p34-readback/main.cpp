// SPDX-License-Identifier: CC0-1.0
//
// Non-stalling GPU readback: the naive path maps the readback
// buffer right after submitting — vkMapMemory isn't the problem,
// READING it before the copy retired is. Fence the copy, keep a
// ring of readback buffers, and only read slot N when N frames of
// work have passed since its submission. The queue stays busy.
//
// Demo: 6 frames, each a different uniform color, rendered as fast
// as possible; a 3-slot readback ring lets the CPU lag the GPU by
// up to 3 frames. Every frame's color is verified from the ring.
//
//   ./app     prints timing + verifies frames (check.py parses)
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* kVert = R"GLSL(
#version 460
void main() {
    vec2 p = vec2((gl_VertexIndex << 1 & 2), gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0, 1);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(push_constant) uniform Push { float r, g, b; } pc;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(pc.r, pc.g, pc.b, 1.0); }
)GLSL";

int main() {
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice({});

    const uint32_t W = 64, H = 64, FRAMES = 6, RING = 3;

    // ---- 3 readback slots: buffer + fence each --------------------
    struct Slot {
        vkmini::Vk::Buffer buf;
        vk::UniqueFence fence;
        bool pending = false;
    };
    std::array<Slot, RING> ring;
    for (auto& s : ring) {
        s.buf = vk.createBuffer(
            W * H * 4, vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent,
            true);
        s.fence = vk.device->createFenceUnique({});
    }

    auto img = vk.createImage(
        W, H, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eColorAttachment |
            vk::ImageUsageFlagBits::eTransferSrc);
    vk::ImageSubresourceRange sr{vk::ImageAspectFlagBits::eColor,
                                 0, 1, 0, 1};

    // minimal pipeline (solid-color fullscreen triangle)
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "s.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "s.frag");
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
    vk::PushConstantRange pcr{};
    pcr.setStageFlags(vk::ShaderStageFlagBits::eFragment)
        .setSize(12);
    vk::PipelineLayoutCreateInfo plci{};
    plci.setPushConstantRanges(pcr);
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

    // frame f draws color (f, 255-f*30, 30) — distinguishable
    float cols[FRAMES][3];
    for (uint32_t f = 0; f < FRAMES; ++f) {
        cols[f][0] = float(f) * 40.f / 255.f;
        cols[f][1] = 1.f - float(f) * 0.12f;
        cols[f][2] = 0.2f;
    }

    uint32_t read = 0, waited = 0;
    for (uint32_t f = 0; f < FRAMES; ++f) {
        auto& slot = ring[f % RING];
        if (slot.pending) {
            // read the frame that submitted into THIS slot last —
            // GPU has had RING frames of headroom
            (void)vk.device->waitForFences(slot.fence.get(), true,
                                           UINT64_MAX);
            ++waited;
            uint8_t b = static_cast<uint8_t*>(slot.buf.mapped)[0];
            uint32_t src = f - RING;   // frame that wrote this slot
            printf("[frame %u] read slot %u <- frame %u "
                   "(r=%u)\n", f, f % RING, src, b);
            int want = int(cols[src][0] * 255 + 0.5f);
            if (abs(int(b) - want) > 3) {
                printf("MISMATCH: got %u want %d\n", b, want);
                return 1;
            }
            ++read;
            vk.device->resetFences(slot.fence.get());
        }

        // record + submit this frame's draw -> copy -> fence
        auto cbuf = std::move(
            vk.device->allocateCommandBuffersUnique(
                {vk.pool.get(), vk::CommandBufferLevel::ePrimary, 1})
                .front());
        vk::CommandBuffer c = cbuf.get();
        c.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::ImageMemoryBarrier toDraw{};
        toDraw.setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(img.img.get())
            .setSubresourceRange(sr)
            .setSrcAccessMask({})
            .setDstAccessMask(
                vk::AccessFlagBits::eColorAttachmentWrite);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::
                              eColorAttachmentOutput,
                          {}, {}, {}, toDraw);
        vk::RenderingAttachmentInfo att{};
        att.setImageView(img.view.get())
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore);
        vk::RenderingInfo ri{};
        ri.setRenderArea({{0, 0}, {W, H}})
            .setLayerCount(1)
            .setColorAttachments(att);
        c.beginRendering(ri);
        vk::Viewport v{0, 0, float(W), float(H), 0.f, 1.f};
        c.setViewport(0, v);
        vk::Rect2D sc{{0, 0}, {W, H}};
        c.setScissor(0, sc);
        c.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe.get());
        c.pushConstants(layout.get(),
                        vk::ShaderStageFlagBits::eFragment, 0, 12,
                        cols[f]);
        c.draw(3, 1, 0, 0);
        c.endRendering();
        vk::ImageMemoryBarrier toCopy{};
        toCopy.setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(img.img.get())
            .setSubresourceRange(sr)
            .setSrcAccessMask(
                vk::AccessFlagBits::eColorAttachmentWrite)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        c.pipelineBarrier(vk::PipelineStageFlagBits::
                              eColorAttachmentOutput,
                          vk::PipelineStageFlagBits::eTransfer, {},
                          {}, {}, toCopy);
        vk::BufferImageCopy rc{};
        rc.setImageSubresource(
              {vk::ImageAspectFlagBits::eColor, 0, 0, 1})
            .setImageExtent(vk::Extent3D{W, H, 1});
        c.copyImageToBuffer(img.img.get(),
                            vk::ImageLayout::eTransferSrcOptimal,
                            slot.buf.buf.get(), rc);
        c.end();
        vk::SubmitInfo si{};
        si.setCommandBuffers(c);
        vk.queue.submit(si, slot.fence.get());
        slot.pending = true;
        cbuf.release();   // freed when slot's fence is next waited —
                          // demo keeps it simple: process exit
    }
    // drain remaining slots
    for (uint32_t f = FRAMES; f < FRAMES + RING && read < FRAMES;
         ++f) {
        auto& slot = ring[f % RING];
        if (!slot.pending)
            continue;
        (void)vk.device->waitForFences(slot.fence.get(), true,
                                       UINT64_MAX);
        uint8_t b = static_cast<uint8_t*>(slot.buf.mapped)[0];
        uint32_t src = FRAMES - RING + (f % RING);
        src = f - RING;
        printf("[frame %u] read slot %u <- frame %u (r=%u)\n", f,
               f % RING, src, b);
        int want = int(cols[src][0] * 255 + 0.5f);
        if (abs(int(b) - want) > 3) {
            printf("MISMATCH: got %u want %d\n", b, want);
            return 1;
        }
        ++read;
    }
    printf("done: %u/%u frames verified, %u fence waits "
           "(no waitIdle until drain)\n",
           read, FRAMES, waited);
    return read == FRAMES ? 0 : 1;
}
