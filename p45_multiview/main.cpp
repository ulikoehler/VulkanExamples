// SPDX-License-Identifier: CC0-1.0
//
// VK_KHR_multiview (core in Vulkan 1.1): ONE draw call renders into
// N attachment LAYERS simultaneously; gl_ViewIndex tells the shader
// which layer/eye it's in. This is how stereo VR does left/right in
// a single pass — same geometry, per-view variation.
//
// Demo: layered 2-layer image; one fullscreen triangle, fragment
// colors differ by gl_ViewIndex (layer0 = red, layer1 = blue).
// Readback samples both layers separately.
//
//   ./app --headless out.ppm   (writes layer0->out.ppm, layer1->out_l1.ppm)
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0,
                  (gl_VertexIndex == 2) ? 3.0 : -1.0);
    vUV = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0, 1);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
#extension GL_EXT_multiview : require
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() {
    // gl_ViewIndex = which layer this invocation renders into
    outColor = (gl_ViewIndex == 0) ? vec4(1, 0.2, 0.2, 1)
                                 : vec4(0.2, 0.2, 1, 1);
}
)GLSL";

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk::PhysicalDeviceVulkan11Features f11{};
    f11.setMultiview(VK_TRUE);
    vk.createDevice({}, &f11);

    const uint32_t W = 384, H = 384;

    // layered render target: arrayLayers=2, usage transfer-src for readback
    vk::ImageCreateInfo ici{};
    ici.setImageType(vk::ImageType::e2D)
        .setFormat(vk::Format::eR8G8B8A8Unorm)
        .setExtent({W, H, 1})
        .setMipLevels(1)
        .setArrayLayers(2)          // <- the two "views"
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage(vk::ImageUsageFlagBits::eColorAttachment |
                  vk::ImageUsageFlagBits::eTransferSrc)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined);
    auto img = vk.device->createImageUnique(ici);
    auto req = vk.device->getImageMemoryRequirements(img.get());
    auto mem = vk.device->allocateMemoryUnique(
        {req.size,
         vk.memoryType(req.memoryTypeBits,
                       vk::MemoryPropertyFlagBits::eDeviceLocal)});
    vk.device->bindImageMemory(img.get(), mem.get(), 0);

    // multiview needs ONE view covering all rendered layers —
    // a 2D-array view with layerCount=2.
    vk::ImageViewCreateInfo vi{};
    vi.setImage(img.get())
        .setViewType(vk::ImageViewType::e2DArray)
        .setFormat(vk::Format::eR8G8B8A8Unorm)
        .setSubresourceRange({vk::ImageAspectFlagBits::eColor,
                              0, 1, 0, 2});
    auto view = vk.device->createImageViewUnique(vi);

    // also per-layer views for readback
    vk::UniqueImageView layerViews[2];
    for (int l = 0; l < 2; ++l) {
        vk::ImageViewCreateInfo lv{};
        lv.setImage(img.get())
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(vk::Format::eR8G8B8A8Unorm)
            .setSubresourceRange({vk::ImageAspectFlagBits::eColor,
                                  0, 1, (uint32_t)l, 1});
        layerViews[l] = vk.device->createImageViewUnique(lv);
    }

    // ---- pipeline -------------------------------------------------
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
    vk::PipelineLayoutCreateInfo plci{};
    auto layout = vk.device->createPipelineLayoutUnique(plci);
    vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(cfmt)
        .setViewMask(0b11);   // views 0 and 1
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
    auto pipe = std::move(pres.value);

    // ---- render into both layers, read each back ------------------
    auto rb = vk.createBuffer(
        W * H * 4 * 2, vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    vk::ImageSubresourceRange sr{vk::ImageAspectFlagBits::eColor,
                                 0, 1, 0, 2};
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::ImageMemoryBarrier b{};
        b.setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(img.get())
            .setSubresourceRange(sr)
            .setSrcAccessMask({})
            .setDstAccessMask(
                vk::AccessFlagBits::eColorAttachmentWrite);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::
                              eColorAttachmentOutput,
                          {}, {}, {}, b);

        vk::RenderingAttachmentInfo att{};
        att.setImageView(view.get())
            .setImageLayout(
                vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setClearValue(vk::ClearValue{vk::ClearColorValue{
                std::array{0.f, 0.f, 0.f, 1.f}}});
        vk::RenderingInfo ri{};
        ri.setRenderArea({{0, 0}, {W, H}})
            .setLayerCount(2)          // renders into both layers
            .setViewMask(0b11)         // multiview bits
            .setColorAttachments(att);
        c.beginRendering(ri);
        vk::Viewport v{0, 0, (float)W, (float)H, 0.f, 1.f};
        c.setViewport(0, v);
        vk::Rect2D sc{{0, 0}, {W, H}};
        c.setScissor(0, sc);
        c.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe.get());
        c.draw(3, 1, 0, 0);            // ONE draw -> BOTH layers
        c.endRendering();

        vk::ImageMemoryBarrier b2{};
        b2.setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(img.get())
            .setSubresourceRange(sr)
            .setSrcAccessMask(
                vk::AccessFlagBits::eColorAttachmentWrite)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        c.pipelineBarrier(vk::PipelineStageFlagBits::
                              eColorAttachmentOutput,
                          vk::PipelineStageFlagBits::eTransfer,
                          {}, {}, {}, b2);
        for (int l = 0; l < 2; ++l) {
            vk::BufferImageCopy cp{};
            cp.setImageSubresource({vk::ImageAspectFlagBits::eColor,
                                    0, (uint32_t)l, 1})
                .setImageExtent({W, H, 1})
                .setBufferOffset((size_t)l * W * H * 4);
            c.copyImageToBuffer(img.get(),
                                vk::ImageLayout::eTransferSrcOptimal,
                                rb.buf.get(), cp);
        }
    });

    // write both layers as PPMs
    const uint8_t* d = (const uint8_t*)rb.mapped;
    for (int l = 0; l < 2; ++l) {
        char nm[64];
        snprintf(nm, 64, l ? "%s_l%d.ppm" : "%s",
                 argv[argc - 1], l);
        FILE* f = fopen(nm, "wb");
        fprintf(f, "P6\n%u %u\n255\n", W, H);
        for (uint32_t i = 0; i < W * H; ++i) {
            const uint8_t* p = d + l * W * H * 4 + i * 4;
            uint8_t rgb[3] = {p[0], p[1], p[2]};
            fwrite(rgb, 1, 3, f);
        }
        fclose(f);
        printf("layer%d -> %s\n", l, nm);
    }
    return 0;
}
