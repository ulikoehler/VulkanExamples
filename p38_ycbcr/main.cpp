// SPDX-License-Identifier: CC0-1.0
//
// YCbCr video frames via VK_KHR_sampler_ycbcr_conversion (core in
// Vulkan 1.1): a multiplanar image + a sampler carrying an
// IMMUTABLE conversion — the fragment shader calls texture() once
// and receives RGB, no manual YUV math.
//
// Demo: build a 64x64 NV12-style image
// (VK_FORMAT_G8_B8R8_2PLANE_420_UNORM: Y plane + interleaved CbCr
// half-res plane), fill 4 quadrants with known BT.601 narrow-range
// YUV colors (red/green/blue/white), sample it -> quadrant colors
// come out as RGB.
//
//   ./app --headless o.ppm     one frame -> PPM for check.py
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
layout(binding = 0) uniform sampler2D tex;   // ycbcr-converting
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() {
    // ONE texture() call: luma+chroma fetch, chroma upsample and
    // YCbCr->RGB matrix all happen inside the sampling hardware.
    outColor = texture(tex, vUV);
}
)GLSL";

int main() {
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    // enable samplerYcbcrConversion (Vulkan 1.1 feature)
    vk::PhysicalDeviceVulkan11Features f11{};
    f11.setSamplerYcbcrConversion(VK_TRUE);
    vk.createDevice({}, &f11);

    const uint32_t W = 64, H = 64;
    const vk::Format fmt = vk::Format::eG8B8R82Plane420Unorm;

    // the conversion object: BT.601 narrow-range ("studio swing"),
    // Cb=plane1 R channel ... for G8B8R8 the CbCr order is fixed.
    vk::SamplerYcbcrConversionCreateInfo ci{};
    ci.setFormat(fmt)
        .setYcbcrModel(vk::SamplerYcbcrModelConversion::eYcbcr601)
        .setYcbcrRange(vk::SamplerYcbcrRange::eItuNarrow)
        .setComponents(vk::ComponentMapping{})
        .setXChromaOffset(vk::ChromaLocation::eCositedEven)
        .setYChromaOffset(vk::ChromaLocation::eCositedEven)
        .setChromaFilter(vk::Filter::eNearest)
        .setForceExplicitReconstruction(VK_FALSE);
    auto conv = vk.device->createSamplerYcbcrConversionUnique(ci);

    // ---- multiplanar image ----------------------------------------
    vk::ImageCreateInfo ii{};
    ii.setImageType(vk::ImageType::e2D)
        .setFormat(fmt)
        .setExtent({W, H, 1})
        .setMipLevels(1)
        .setArrayLayers(1)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setFlags(vk::ImageCreateFlagBits::eMutableFormat)
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage(vk::ImageUsageFlagBits::eTransferDst |
                  vk::ImageUsageFlagBits::eSampled)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined);
    auto img = vk.device->createImageUnique(ii);
    auto req = vk.device->getImageMemoryRequirements(img.get());
    auto mem = vk.device->allocateMemoryUnique(
        {req.size,
         vk.memoryType(
             req.memoryTypeBits,
             vk::MemoryPropertyFlagBits::eDeviceLocal)});
    vk.device->bindImageMemory(img.get(), mem.get(), 0);

    // ---- fill planes: 4 quadrants of known BT.601-narrow YUV -----
    // (Y, Cb, Cr) values are the standard studio-swing encodings:
    //   red   (255,0,0)     -> ( 82,  90, 240)
    //   green (0,255,0)     -> (146,  54,  34)
    //   blue  (0,0,255)     -> ( 41, 240, 110)
    //   white (255,255,255) -> (235, 128, 128)
    std::array<uint8_t, W * H> yplane;
    std::array<uint8_t, (W / 2) * (H / 2) * 2> cbcr;
    struct { uint8_t y, cb, cr; } quad[4] = {
        {82, 90, 240}, {146, 54, 34}, {41, 240, 110}, {235, 128, 128}};
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            int q = (x >= W / 2) + 2 * (y >= H / 2);
            yplane[y * W + x] = quad[q].y;
        }
    for (uint32_t y = 0; y < H / 2; ++y)
        for (uint32_t x = 0; x < W / 2; ++x) {
            int q = (x >= W / 4) + 2 * (y >= H / 4);
            cbcr[(y * (W / 2) + x) * 2 + 0] = quad[q].cb;
            cbcr[(y * (W / 2) + x) * 2 + 1] = quad[q].cr;
        }

    auto sty = vk.createBuffer(
        yplane.size(), vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(sty.mapped, yplane.data(), yplane.size());
    auto stc = vk.createBuffer(
        cbcr.size(), vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(stc.mapped, cbcr.data(), cbcr.size());

    vk::ImageSubresourceRange sr{vk::ImageAspectFlagBits::eColor,
                                 0, 1, 0, 1};
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::ImageMemoryBarrier toDst{};
        toDst.setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(img.get())
            .setSubresourceRange(sr)   // color aspect covers ALL planes
            .setSrcAccessMask({})
            .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::eTransfer, {},
                          {}, {}, toDst);
        // plane 0: luma, full res, 1 byte/texel
        vk::BufferImageCopy cy{};
        cy.setImageSubresource({vk::ImageAspectFlagBits::ePlane0, 0,
                                0, 1})
            .setImageExtent({W, H, 1});
        // plane 1: interleaved CbCr, HALF res, 2 bytes/texel
        vk::BufferImageCopy cc{};
        cc.setImageSubresource({vk::ImageAspectFlagBits::ePlane1, 0,
                                0, 1})
            .setImageExtent({W / 2, H / 2, 1});
        c.copyBufferToImage(sty.buf.get(), img.get(),
                            vk::ImageLayout::eTransferDstOptimal, cy);
        c.copyBufferToImage(stc.buf.get(), img.get(),
                            vk::ImageLayout::eTransferDstOptimal, cc);
        vk::ImageMemoryBarrier toRead{};
        toRead.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
            .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(img.get())
            .setSubresourceRange(sr)
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                          vk::PipelineStageFlagBits::eFragmentShader,
                          {}, {}, {}, toRead);
    });

    // ---- view + sampler both chain the conversion ----------------
    vk::SamplerYcbcrConversionInfo convInfo{};
    convInfo.setConversion(conv.get());
    vk::ImageViewCreateInfo vi{};
    vi.setImage(img.get())
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(fmt)
        .setSubresourceRange(sr)
        .setPNext(&convInfo);
    auto view = vk.device->createImageViewUnique(vi);
    vk::SamplerCreateInfo si{};
    si.setMagFilter(vk::Filter::eNearest)
        .setMinFilter(vk::Filter::eNearest)
        .setMipmapMode(vk::SamplerMipmapMode::eNearest)
        .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
        .setPNext(&convInfo);   // conversion lives IN the sampler
    auto sampler = vk.device->createSamplerUnique(si);

    // ---- descriptor ----------------------------------------------
    // spec rule: a sampler carrying a ycbcr conversion must be an
    // IMMUTABLE sampler baked into the set layout; the descriptor
    // write then only carries the image view.
    vk::Sampler imSamp = sampler.get();
    vk::DescriptorSetLayoutBinding binding{
        0, vk::DescriptorType::eCombinedImageSampler, 1,
        vk::ShaderStageFlagBits::eFragment, &imSamp};
    vk::DescriptorSetLayoutCreateInfo dslci{};
    dslci.setBindings(binding);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dslci);
    vk::DescriptorPoolSize ps{
        vk::DescriptorType::eCombinedImageSampler, 4};
    auto dpool = vk.device->createDescriptorPoolUnique(
        {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 2,
         ps});
    vk::DescriptorSetLayout dslH = dsl.get();
    vk::DescriptorSetAllocateInfo dsai{};
    dsai.setDescriptorPool(dpool.get()).setSetLayouts(dslH);
    auto dsets = vk.device->allocateDescriptorSetsUnique(dsai);
    vk::DescriptorImageInfo ii2{nullptr, view.get(),
                                vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::WriteDescriptorSet w{};
    w.setDstSet(dsets[0].get())
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setImageInfo(ii2);
    vk.device->updateDescriptorSets(w, {});

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
    hl.savePpm("out.ppm");
    printf("wrote out.ppm\n");
    return 0;
}
