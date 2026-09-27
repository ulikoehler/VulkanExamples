// SPDX-License-Identifier: CC0-1.0
//
// VK_EXT_descriptor_buffer: descriptors become bytes in a plain
// VkBuffer you own — no descriptor pools, no descriptor sets, no
// vkUpdateDescriptorSets. You memcpy a descriptor blob into
// device-visible memory and bind it with two commands.
//
// This is the "everything is a buffer" endgame of descriptor
// management: swapping a texture = rewriting a few bytes + a memory
// barrier, which is exactly what a video wall wants when tiles swap
// sources every frame.
//
//   ./app --headless out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc

#include "vkmini.hpp"

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
layout(binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(texture(tex, vUV).rgb, 1); }
)GLSL";

int main(int argc, char** argv) {
    const char* ppm = "out.ppm";
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--headless") && i + 1 < argc)
            ppm = argv[++i];

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    // check support before enabling
    vk::PhysicalDeviceDescriptorBufferFeaturesEXT dbFeat{};
    dbFeat.setDescriptorBuffer(true);
    vk::PhysicalDeviceFeatures2 f2{};
    f2.setPNext(&dbFeat);
    vk.phys.getFeatures2(&f2);
    if (!dbFeat.descriptorBuffer) {
        printf("SKIP: VK_EXT_descriptor_buffer not supported\n");
        return 0;
    }
    dbFeat.setPNext(nullptr);
    vk.createDevice({VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME},
                    &dbFeat);
    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr, vk.device.get());

    // ---- a small 2-color test texture -----------------------------
    auto tex = vk.createImage(64, 64, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst);
    std::vector<uint8_t> px(64 * 64 * 4);
    for (int y = 0; y < 64; ++y) for (int x = 0; x < 64; ++x) {
        auto* p = &px[(y * 64 + x) * 4];
        bool a = ((x / 8 + y / 8) & 1) == 0;
        p[0] = a ? 255 : 30; p[1] = a ? 60 : 200; p[2] = a ? 0 : 90;
        p[3] = 255;
    }
    auto st = vk.createBuffer(px.size(), vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);
    memcpy(st.mapped, px.data(), px.size());
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::ImageMemoryBarrier b{};
        b.setOldLayout(vk::ImageLayout::eUndefined)
         .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
         .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
         .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
         .setImage(tex.img.get())
         .setSubresourceRange({vk::ImageAspectFlagBits::eColor,0,1,0,1})
         .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, b);
        vk::BufferImageCopy r{};
        r.setImageSubresource({vk::ImageAspectFlagBits::eColor,0,0,1})
         .setImageExtent({64, 64, 1});
        c.copyBufferToImage(st.buf.get(), tex.img.get(),
            vk::ImageLayout::eTransferDstOptimal, r);
        b.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
         .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
         .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
         .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, b);
    });

    // ---- set layout flagged for descriptor-buffer use -------------
    vk::DescriptorSetLayoutBinding b0{0,
        vk::DescriptorType::eCombinedImageSampler, 1,
        vk::ShaderStageFlagBits::eFragment};
    vk::DescriptorSetLayoutCreateInfo dslci{};
    dslci.setFlags(
            vk::DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT)
        .setBindings(b0);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dslci);

    // layout's footprint in the descriptor buffer + per-binding ofs
    vk::DeviceSize setSize =
        vk.device->getDescriptorSetLayoutSizeEXT(dsl.get(), dldi);
    vk::DeviceSize b0ofs =
        vk.device->getDescriptorSetLayoutBindingOffsetEXT(
            dsl.get(), 0, dldi);
    auto dbProps =
        vk.phys.getProperties2<
            vk::PhysicalDeviceProperties2,
            vk::PhysicalDeviceDescriptorBufferPropertiesEXT>()
            .get<vk::PhysicalDeviceDescriptorBufferPropertiesEXT>();

    // ---- the descriptor buffer itself ------------------------------
    auto descBuf = vk.createBuffer(setSize,
        vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT |
        vk::BufferUsageFlagBits::eSamplerDescriptorBufferEXT |
        vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);

    // write the combined-image-sampler descriptor bytes into it:
    // vkGetDescriptorEXT fills a blob sized for this implementation
    auto sampler = vk.device->createSamplerUnique(
        {{}, vk::Filter::eNearest, vk::Filter::eNearest});
    vk::DescriptorImageInfo dii{sampler.get(), tex.view.get(),
        vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorGetInfoEXT dgi{};
    vk::DescriptorDataEXT dd{};
    dd.setPCombinedImageSampler(&dii);
    dgi.setType(vk::DescriptorType::eCombinedImageSampler)
       .setData(dd);
    size_t blobSz = dbProps.combinedImageSamplerDescriptorSize;
    vk.device->getDescriptorEXT(dgi, blobSz,
        static_cast<char*>(descBuf.mapped) + b0ofs, dldi);
    printf("descriptor blob: %zu bytes at set-offset %zu\n",
           blobSz, size_t(b0ofs));

    // ---- pipeline --------------------------------------------------
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "d.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "d.frag");
    vk::PipelineShaderStageCreateInfo stg[2];
    stg[0].setStage(vk::ShaderStageFlagBits::eVertex).setModule(vs.get())
        .setPName("main");
    stg[1].setStage(vk::ShaderStageFlagBits::eFragment).setModule(fs.get())
        .setPName("main");
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
    plci.setSetLayouts(dsl.get());
    auto layout = vk.device->createPipelineLayoutUnique(plci);
    vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
    vk::PipelineRenderingCreateInfo pr{};
    pr.setColorAttachmentFormats(cfmt);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(stg).setPVertexInputState(&vin)
      .setPInputAssemblyState(&ia).setPViewportState(&vps)
      .setPDynamicState(&dsi).setPRasterizationState(&rs)
      .setPMultisampleState(&ms).setPColorBlendState(&blend)
      .setLayout(layout.get()).setPNext(&pr)
      // pipelines using descriptor buffers need the flag
      .setFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT);
    auto pipe = std::move(
        vk.device->createGraphicsPipelineUnique({}, gi).value);

    // ---- draw ------------------------------------------------------
    vkmini::Headless hl;
    hl.init(vk, 256, 256);
    hl.render(vk::ImageLayout::eColorAttachmentOptimal,
              [&](vk::CommandBuffer c) {
        vk::RenderingAttachmentInfo att{};
        att.setImageView(hl.color.view.get())
           .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
           .setLoadOp(vk::AttachmentLoadOp::eClear)
           .setStoreOp(vk::AttachmentStoreOp::eStore);
        vk::RenderingInfo ri{};
        ri.setRenderArea({{0,0},{256,256}}).setLayerCount(1)
          .setColorAttachments(att);
        c.beginRendering(ri);
        c.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe.get());
        vk::Viewport vp{0,0,256.f,256.f,0,1};
        vk::Rect2D sc{{0,0},{256,256}};
        c.setViewport(0, vp); c.setScissor(0, sc);

        // bind the buffer, then select (bufferIndex, setOffset)
        vk::DescriptorBufferBindingInfoEXT bi{};
        bi.setAddress(vk.device->getBufferAddress(descBuf.buf.get()))
          .setUsage(vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT |
                    vk::BufferUsageFlagBits::eSamplerDescriptorBufferEXT);
        c.bindDescriptorBuffersEXT(bi, dldi);
        uint32_t bufIdx = 0; vk::DeviceSize ofs = 0;
        c.setDescriptorBufferOffsetsEXT(
            vk::PipelineBindPoint::eGraphics, layout.get(), 0,
            bufIdx, ofs, dldi);
        c.draw(3, 1, 0, 0);
        c.endRendering();
    });
    hl.savePpm(ppm);
    printf("wrote %s\n", ppm);
    return 0;
}
