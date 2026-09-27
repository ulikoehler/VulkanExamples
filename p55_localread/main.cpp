// SPDX-License-Identifier: CC0-1.0
//
// VK_KHR_dynamic_rendering_local_read: read a color attachment from
// the fragment shader WHILE still inside a dynamic render — the
// subpass-input pattern, without renderpass objects. On tile GPUs
// the data never leaves tile memory; on desktop it's the standard
// in-pass feedback trick (blend modes, accumulation, post-fx chains).
//
// Demo: draw 1 writes a gradient; a by-region barrier flips the
// attachment to RENDERING_LOCAL_READ; draw 2 reads its own pixel
// via subpassLoad() and writes the inverted color back to the same
// attachment — all in one vkCmdBeginRendering.
//
//   ./app --headless out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc

#include "vkmini.hpp"

static const char* kVertGrad = R"GLSL(
#version 460
layout(location = 0) out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0,
                  (gl_VertexIndex == 2) ? 3.0 : -1.0);
    vUV = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0, 1);
}
)GLSL";
static const char* kFragGrad = R"GLSL(
#version 460
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(vUV, 0.25, 1); }
)GLSL";

// draw 2: read the CURRENT pixel (attachment is in
// RENDERING_LOCAL_READ) and write the inverse.
static const char* kFragInv = R"GLSL(
#version 460
layout(input_attachment_index = 0, binding = 0)
    uniform subpassInput prev;
layout(location = 0) out vec4 outColor;
void main() {
    vec4 p = subpassLoad(prev);          // this fragment's own pixel
    outColor = vec4(1.0 - p.rgb, 1.0);
}
)GLSL";

int main(int argc, char** argv) {
    const char* ppm = "out.ppm";
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--headless") && i + 1 < argc)
            ppm = argv[++i];

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR lr{};
    lr.setDynamicRenderingLocalRead(true);
    vk::PhysicalDeviceFeatures2 f2{};
    f2.setPNext(&lr);
    vk.phys.getFeatures2(&f2);
    if (!lr.dynamicRenderingLocalRead) {
        printf("SKIP: no dynamic_rendering_local_read\n");
        return 0;
    }
    lr.setPNext(nullptr);
    vk.createDevice(
        {VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME}, &lr);
    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr, vk.device.get());

    constexpr uint32_t W = 256, H = 256;
    // one image: color attachment AND input attachment
    auto img = vk.createImage(W, H, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eColorAttachment |
        vk::ImageUsageFlagBits::eInputAttachment |
        vk::ImageUsageFlagBits::eTransferSrc);

    // input-attachment descriptor (no sampler — subpassLoad, not tex)
    vk::DescriptorSetLayoutBinding b0{0,
        vk::DescriptorType::eInputAttachment, 1,
        vk::ShaderStageFlagBits::eFragment};
    vk::DescriptorSetLayoutCreateInfo dslci{};
    dslci.setBindings(b0);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dslci);
    vk::DescriptorPoolSize dps{vk::DescriptorType::eInputAttachment, 1};
    auto dpool = vk.device->createDescriptorPoolUnique({{}, 1, dps});
    auto dsets = vk.device->allocateDescriptorSetsUnique(
        vk::DescriptorSetAllocateInfo{}.setDescriptorPool(dpool.get())
            .setSetLayouts(dsl.get()));
    // NOTE: the imageLayout in the descriptor must match the layout
    // at draw time -> eRenderingLocalReadKHR
    vk::DescriptorImageInfo dii{{}, img.view.get(),
        vk::ImageLayout::eRenderingLocalRead};
    vk::WriteDescriptorSet w{};
    w.setDstSet(dsets[0].get()).setDstBinding(0)
     .setDescriptorType(vk::DescriptorType::eInputAttachment)
     .setImageInfo(dii);
    vk.device->updateDescriptorSets(w, {});

    // two pipelines: gradient writer + inverter reader
    auto mkPipe = [&](const char* frag, bool withInput) {
        auto vs = vk.shader(kVertGrad, shaderc_vertex_shader, "v");
        auto fs = vk.shader(frag, shaderc_fragment_shader, "f");
        vk::PipelineShaderStageCreateInfo stg[2];
        stg[0].setStage(vk::ShaderStageFlagBits::eVertex)
            .setModule(vs.get()).setPName("main");
        stg[1].setStage(vk::ShaderStageFlagBits::eFragment)
            .setModule(fs.get()).setPName("main");
        vk::PipelineVertexInputStateCreateInfo vin{};
        vk::PipelineInputAssemblyStateCreateInfo ia{};
        ia.setTopology(vk::PrimitiveTopology::eTriangleList);
        vk::PipelineViewportStateCreateInfo vps{};
        vps.setViewportCount(1).setScissorCount(1);
        std::array dyn{vk::DynamicState::eViewport,
                       vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dsi{};
        dsi.setDynamicStates(dyn);
        vk::PipelineRasterizationStateCreateInfo rs{};
        rs.setPolygonMode(vk::PolygonMode::eFill)
          .setCullMode(vk::CullModeFlagBits::eNone).setLineWidth(1.f);
        vk::PipelineMultisampleStateCreateInfo ms{};
        ms.setRasterizationSamples(vk::SampleCountFlagBits::e1);
        vk::PipelineColorBlendAttachmentState cba{};
        cba.setColorWriteMask(vk::ColorComponentFlagBits::eR |
                              vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB |
                              vk::ColorComponentFlagBits::eA);
        vk::PipelineColorBlendStateCreateInfo blend{};
        blend.setAttachments(cba);
        vk::PipelineLayoutCreateInfo plci{};
        if (withInput) plci.setSetLayouts(dsl.get());
        auto layout = vk.device->createPipelineLayoutUnique(plci);
        vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
        vk::PipelineRenderingCreateInfo pr{};
        pr.setColorAttachmentFormats(cfmt);
        vk::GraphicsPipelineCreateInfo gi{};
        gi.setStages(stg).setPVertexInputState(&vin)
          .setPInputAssemblyState(&ia).setPViewportState(&vps)
          .setPDynamicState(&dsi).setPRasterizationState(&rs)
          .setPMultisampleState(&ms).setPColorBlendState(&blend)
          .setLayout(layout.get()).setPNext(&pr);
        struct R { vk::UniquePipeline p; vk::UniquePipelineLayout l; };
        R r;
        r.l = std::move(layout);
        r.p = std::move(
            vk.device->createGraphicsPipelineUnique({}, gi).value);
        return r;
    };
    auto pGrad = mkPipe(kFragGrad, false);
    auto pInv = mkPipe(kFragInv, true);

    // ---- render: gradient -> local-read barrier -> invert ---------
    auto cmds = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::ePrimary, 1});
    auto& c = cmds[0];
    c->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::ImageMemoryBarrier pre{};
    pre.setOldLayout(vk::ImageLayout::eUndefined)
       .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
       .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
       .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
       .setImage(img.img.get())
       .setSubresourceRange({vk::ImageAspectFlagBits::eColor,0,1,0,1})
       .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
    c->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, {}, {}, pre);
    vk::RenderingAttachmentInfo att{};
    att.setImageView(img.view.get())
       .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
       .setLoadOp(vk::AttachmentLoadOp::eClear)
       .setStoreOp(vk::AttachmentStoreOp::eStore);
    vk::RenderingInfo ri{};
    ri.setRenderArea({{0,0},{W,H}}).setLayerCount(1)
      .setColorAttachments(att);
    c->beginRendering(ri);
    c->bindPipeline(vk::PipelineBindPoint::eGraphics, pGrad.p.get());
    vk::Viewport vp{0,0,float(W),float(H),0,1};
    vk::Rect2D sc{{0,0},{W,H}};
    c->setViewport(0, vp); c->setScissor(0, sc);
    c->draw(3, 1, 0, 0);

    // the local-read barrier: by-region dep, layout -> LOCAL_READ.
    // RENDERING_LOCAL_READ allows BOTH input reads AND color writes
    // of the same pixel — the in-pass feedback layout.
    vk::ImageMemoryBarrier2 mid{};
    mid.setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
       .setNewLayout(vk::ImageLayout::eRenderingLocalRead)
       .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
       .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
       .setImage(img.img.get())
       .setSubresourceRange(
           {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})
       .setSrcStageMask(
           vk::PipelineStageFlagBits2::eColorAttachmentOutput)
       .setSrcAccessMask(
           vk::AccessFlagBits2::eColorAttachmentWrite)
       .setDstStageMask(
           vk::PipelineStageFlagBits2::eFragmentShader |
           vk::PipelineStageFlagBits2::eColorAttachmentOutput)
       .setDstAccessMask(
           vk::AccessFlagBits2::eInputAttachmentRead |
           vk::AccessFlagBits2::eColorAttachmentWrite);
    vk::DependencyInfo dep{};
    dep.setDependencyFlags(vk::DependencyFlagBits::eByRegion)
       .setImageMemoryBarriers(mid);
    c->pipelineBarrier2(dep);

    // attachment imageLayout for the local-read portion of the render
    att.setImageLayout(vk::ImageLayout::eRenderingLocalRead);
    c->bindPipeline(vk::PipelineBindPoint::eGraphics, pInv.p.get());
    c->bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                          pInv.l.get(), 0, dsets[0].get(), {});
    c->draw(3, 1, 0, 0);
    c->endRendering();

    // readback
    auto rb = vk.createBuffer(W * H * 4,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);
    vk::ImageMemoryBarrier post{};
    post.setOldLayout(vk::ImageLayout::eRenderingLocalRead)
        .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(img.img.get())
        .setSubresourceRange({vk::ImageAspectFlagBits::eColor,0,1,0,1})
        .setSrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
        .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
    c->pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, post);
    vk::BufferImageCopy r{};
    r.setImageSubresource({vk::ImageAspectFlagBits::eColor,0,0,1})
     .setImageExtent({W, H, 1});
    c->copyImageToBuffer(img.img.get(),
        vk::ImageLayout::eTransferSrcOptimal, rb.buf.get(), r);
    c->end();
    vk::SubmitInfo si{}; si.setCommandBuffers(c.get());
    vk.queue.submit(si);
    vk.queue.waitIdle();

    FILE* f = fopen(ppm, "wb");
    fprintf(f, "P6\n%u %u\n255\n", W, H);
    auto* b = static_cast<uint8_t*>(rb.mapped);
    for (uint32_t i = 0; i < W * H; ++i)
        fwrite(b + i * 4, 1, 3, f);
    fclose(f);
    printf("wrote %s\n", ppm);
    return 0;
}
