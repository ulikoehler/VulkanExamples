// SPDX-License-Identifier: CC0-1.0
//
// Bindless textures with VK_EXT_descriptor_indexing: a runtime-sized
// descriptor array (`sampler2D tex[]`) indexed PER FRAGMENT with
// nonuniformEXT — something the dynamically-uniform index in the
// texture-array post could not do.
//
// Demo: ONE fullscreen draw; the fragment shader derives the texture
// index from its own screen position (2x2 quadrant -> tex 0..3).
// 4 textures, 1 draw, per-pixel index. Needs:
//   shaderSampledImageArrayNonUniformIndexing,
//   runtimeDescriptorArray, descriptorBindingPartiallyBound,
//   descriptorBindingVariableDescriptorCount
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
    // fullscreen triangle: vertices (-1,-1) (3,-1) (-1,3)
    vec2 p = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0,
                  (gl_VertexIndex == 2) ? 3.0 : -1.0);
    vUV = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0, 1);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
#extension GL_EXT_nonuniform_qualifier : require
layout(binding = 0) uniform sampler2D tex[];   // UNSIZED array
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() {
    // per-fragment index derived from the pixel position —
    // non-uniform across the draw, so nonuniformEXT is required.
    int idx = int(vUV.x > 0.5) + 2 * int(vUV.y > 0.5);
    outColor = texture(tex[nonuniformEXT(idx)], fract(vUV * 8.0));
}
)GLSL";

int main(int argc, char** argv) {
    bool headless = argc > 1 && std::string(argv[1]) == "--headless";
    const char* ppmOut = argc > 2 ? argv[2] : "out.ppm";
    (void)headless;

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    // ---- descriptor indexing features ----------------------------
    vk::PhysicalDeviceDescriptorIndexingFeatures di{};
    di.setShaderSampledImageArrayNonUniformIndexing(VK_TRUE)
        .setRuntimeDescriptorArray(VK_TRUE)
        .setDescriptorBindingPartiallyBound(VK_TRUE)
        .setDescriptorBindingVariableDescriptorCount(VK_TRUE)
        .setDescriptorBindingSampledImageUpdateAfterBind(VK_TRUE);
    vk::PhysicalDeviceVulkan12Features f12{};
    // nothing needed from v12 here; the chain just shows the pattern
    f12.setPNext(&di);
    vk.createDevice({}, &f12);

    // ---- 4 solid-color 8x8 textures ------------------------------
    uint32_t px[4] = {0xFF3333FFu, 0xFF33FF33u, 0xFFFF3333u,
                      0xFFFFFFFFu};  // ABGR little-endian: R,G,B,W
    std::array<vkmini::Vk::Image, 4> texs;
    for (int i = 0; i < 4; ++i) {
        std::array<uint32_t, 64> dat;
        dat.fill(px[i]);
        auto st = vk.createBuffer(
            sizeof(dat), vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent,
            true);
        memcpy(st.mapped, dat.data(), sizeof(dat));
        texs[i] = vk.createImage(
            8, 8, vk::Format::eR8G8B8A8Unorm,
            vk::ImageUsageFlagBits::eTransferDst |
                vk::ImageUsageFlagBits::eSampled);
        vk::ImageSubresourceRange sr{vk::ImageAspectFlagBits::eColor,
                                     0, 1, 0, 1};
        vk.oneTime([&](vk::CommandBuffer c) {
            vk::ImageMemoryBarrier toDst{};
            toDst.setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
                .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setImage(texs[i].img.get())
                .setSubresourceRange(sr)
                .setSrcAccessMask({})
                .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
            c.pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
                toDst);
            vk::BufferImageCopy cp{};
            cp.setImageSubresource(
                  {vk::ImageAspectFlagBits::eColor, 0, 0, 1})
                .setImageExtent(vk::Extent3D{8, 8, 1});
            c.copyBufferToImage(st.buf.get(), texs[i].img.get(),
                                vk::ImageLayout::eTransferDstOptimal,
                                cp);
            vk::ImageMemoryBarrier toRead{};
            toRead.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
                .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
                .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setImage(texs[i].img.get())
                .setSubresourceRange(sr)
                .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
            c.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eFragmentShader, {}, {},
                {}, toRead);
        });
    }

    // ---- runtime-sized binding: count N at ALLOCATE time ---------
    const uint32_t kTexCount = 4;
    vk::DescriptorSetLayoutBinding binding{
        0, vk::DescriptorType::eCombinedImageSampler,
        kTexCount,  // upper bound of the runtime array
        vk::ShaderStageFlagBits::eFragment};
    vk::DescriptorBindingFlags bflags =
        vk::DescriptorBindingFlagBits::ePartiallyBound |
        vk::DescriptorBindingFlagBits::eVariableDescriptorCount |
        vk::DescriptorBindingFlagBits::eUpdateAfterBind;
    vk::DescriptorSetLayoutBindingFlagsCreateInfo bfci{};
    bfci.setBindingFlags(bflags);
    vk::DescriptorSetLayoutCreateInfo dslci{};
    dslci.setBindings(binding)
        .setFlags(vk::DescriptorSetLayoutCreateFlagBits::
                      eUpdateAfterBindPool)
        .setPNext(&bfci);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dslci);
    vk::DescriptorPoolSize ps{
        vk::DescriptorType::eCombinedImageSampler, kTexCount};
    auto dpool = vk.device->createDescriptorPoolUnique(
        {{vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind |
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet},
         1, ps});
    vk::DescriptorSetLayout dslH = dsl.get();
    uint32_t varCount = kTexCount;  // actual size of THIS set's array
    vk::DescriptorSetVariableDescriptorCountAllocateInfo vdai{};
    vdai.setDescriptorCounts(varCount);
    vk::DescriptorSetAllocateInfo dsai{};
    dsai.setDescriptorPool(dpool.get())
        .setSetLayouts(dslH)
        .setPNext(&vdai);
    auto dsets = vk.device->allocateDescriptorSetsUnique(dsai);

    auto sampler = vk.device->createSamplerUnique(
        {{},
         vk::Filter::eNearest,
         vk::Filter::eNearest,
         vk::SamplerMipmapMode::eNearest,
         vk::SamplerAddressMode::eRepeat,
         vk::SamplerAddressMode::eRepeat,
         vk::SamplerAddressMode::eRepeat});

    std::array<vk::DescriptorImageInfo, 4> infos;
    for (int i = 0; i < 4; ++i)
        infos[i] = {sampler.get(), texs[i].view.get(),
                    vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::WriteDescriptorSet w{};
    w.setDstSet(dsets[0].get())
        .setDstBinding(0)
        .setDescriptorType(
            vk::DescriptorType::eCombinedImageSampler)
        .setImageInfo(infos);
    vk.device->updateDescriptorSets(w, {});

    // ---- pipeline ------------------------------------------------
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
                              std::array{0.05f, 0.05f, 0.08f,
                                         1.f}}});
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
                  c.draw(3, 1, 0, 0);   // ONE draw, 4 textures
                  c.endRendering();
              });
    hl.savePpm(ppmOut);
    printf("wrote %s\n", ppmOut);
    return 0;
}
