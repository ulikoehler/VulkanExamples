// SPDX-License-Identifier: CC0-1.0
//
// Compute post-filter: a graphics pass renders a sharp checkerboard
// into image A (color attachment), a COMPUTE pass box-blurs it into
// image B (storage image, eGeneral), and B is copied back out.
// Compute as a post-processing stage — the texture-sampler read +
// image2D write in one dispatch, no second graphics pipeline.
//
//   ./app --headless o.ppm     one frame -> PPM for check.py
//   ./app                      windowed
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

// sharp 8x8 checkerboard, NDC -1..+1 over 64x64
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
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() {
    bool e = ((int(vUV.x * 8) ^ int(vUV.y * 8)) & 1) != 0;
    outColor = vec4(e ? vec3(0.9) : vec3(0.05), 1.0);
}
)GLSL";

// 5x5 box blur: texture() the input, imageStore() the result
static const char* kComp = R"GLSL(
#version 460
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0) uniform sampler2D src;
layout(binding = 1, rgba8) uniform writeonly image2D dst;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    vec2 ts = 1.0 / vec2(imageSize(dst));
    vec3 acc = vec3(0);
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx)
            acc += texture(src, (vec2(p) + vec2(dx, dy)) * ts).rgb;
    imageStore(dst, p, vec4(acc / 25.0, 1.0));
}
)GLSL";

int main(int argc, char** argv) {
    bool headless = argc > 1 && std::string(argv[1]) == "--headless";
    const char* ppmOut = argc > 2 ? argv[2] : "out.ppm";
    (void)headless;

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice({});

    const uint32_t S = 128;
    vk::ImageSubresourceRange sr{vk::ImageAspectFlagBits::eColor,
                                 0, 1, 0, 1};

    // image A: graphics renders the checkerboard into it
    auto imgA = vk.createImage(
        S, S, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eColorAttachment |
            vk::ImageUsageFlagBits::eSampled);
    // image B: compute writes the blurred result
    auto imgB = vk.createImage(
        S, S, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eTransferSrc);

    // ---- graphics pipeline (checkerboard) -------------------------
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
    auto glayout = vk.device->createPipelineLayoutUnique({});
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
        .setLayout(glayout.get())
        .setPNext(&rendering);
    auto gpres = vk.device->createGraphicsPipelineUnique({}, gi);
    if (gpres.result != vk::Result::eSuccess)
        throw std::runtime_error("graphics pipeline failed");
    auto gpipe = std::move(gpres.value);

    // ---- compute pipeline (blur) -----------------------------------
    auto cs = vk.shader(kComp, shaderc_compute_shader, "s.comp");
    std::array cbds{
        vk::DescriptorSetLayoutBinding{
            0, vk::DescriptorType::eCombinedImageSampler, 1,
            vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{
            1, vk::DescriptorType::eStorageImage, 1,
            vk::ShaderStageFlagBits::eCompute}};
    vk::DescriptorSetLayoutCreateInfo cdslci{};
    cdslci.setBindings(cbds);
    auto cdsl = vk.device->createDescriptorSetLayoutUnique(cdslci);
    vk::DescriptorSetLayout cdslH = cdsl.get();
    vk::PipelineLayoutCreateInfo cplci{};
    cplci.setSetLayouts(cdslH);
    auto clayout = vk.device->createPipelineLayoutUnique(cplci);
    vk::ComputePipelineCreateInfo cpi{};
    cpi.stage.setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(cs.get())
        .setPName("main");
    cpi.setLayout(clayout.get());
    auto cres = vk.device->createComputePipelineUnique({}, cpi);
    if (cres.result != vk::Result::eSuccess)
        throw std::runtime_error("compute pipeline failed");
    auto cpipe = std::move(cres.value);

    // descriptors: binding0 = imgA sampled, binding1 = imgB storage
    vk::DescriptorPoolSize ps[2]{
        {vk::DescriptorType::eCombinedImageSampler, 1},
        {vk::DescriptorType::eStorageImage, 1}};
    auto dpool = vk.device->createDescriptorPoolUnique(
        {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 2,
         ps});
    vk::DescriptorSetAllocateInfo dsai{};
    dsai.setDescriptorPool(dpool.get()).setSetLayouts(cdslH);
    auto dsets = vk.device->allocateDescriptorSetsUnique(dsai);
    auto sampler = vk.device->createSamplerUnique(
        {{},
         vk::Filter::eNearest,
         vk::Filter::eNearest,
         vk::SamplerMipmapMode::eNearest,
         vk::SamplerAddressMode::eClampToEdge,
         vk::SamplerAddressMode::eClampToEdge,
         vk::SamplerAddressMode::eClampToEdge});
    vk::DescriptorImageInfo ia_{sampler.get(), imgA.view.get(),
                               vk::ImageLayout::
                                   eShaderReadOnlyOptimal};
    vk::DescriptorImageInfo ib_{VK_NULL_HANDLE, imgB.view.get(),
                               vk::ImageLayout::eGeneral};
    vk::WriteDescriptorSet w[2];
    w[0].setDstSet(dsets[0].get())
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setImageInfo(ia_);
    w[1].setDstSet(dsets[0].get())
        .setDstBinding(1)
        .setDescriptorType(vk::DescriptorType::eStorageImage)
        .setImageInfo(ib_);
    vk.device->updateDescriptorSets(w, {});

    // ---- one command buffer: draw -> barrier -> blur -> readback --
    vkmini::Headless hl;
    hl.init(vk, S, S);
    auto cbuf = std::move(
        vk.device->allocateCommandBuffersUnique(
            {vk.pool.get(), vk::CommandBufferLevel::ePrimary, 1})
            .front());
    vk::CommandBuffer c = cbuf.get();
    c.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // imgA: undefined -> color attachment
    vk::ImageMemoryBarrier b1{};
    b1.setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(imgA.img.get())
        .setSubresourceRange(sr)
        .setSrcAccessMask({})
        .setDstAccessMask(
            vk::AccessFlagBits::eColorAttachmentWrite);
    c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                      vk::PipelineStageFlagBits::
                          eColorAttachmentOutput,
                      {}, {}, {}, b1);

    vk::RenderingAttachmentInfo att{};
    att.setImageView(imgA.view.get())
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearValue{vk::ClearColorValue{
            std::array{0.f, 0.f, 0.f, 1.f}}});
    vk::RenderingInfo ri{};
    ri.setRenderArea({{0, 0}, {S, S}})
        .setLayerCount(1)
        .setColorAttachments(att);
    c.beginRendering(ri);
    vk::Viewport v{0, 0, float(S), float(S), 0.f, 1.f};
    c.setViewport(0, v);
    vk::Rect2D sc{{0, 0}, {S, S}};
    c.setScissor(0, sc);
    c.bindPipeline(vk::PipelineBindPoint::eGraphics, gpipe.get());
    c.draw(3, 1, 0, 0);
    c.endRendering();

    // imgA: attachment -> shader read; imgB: undefined -> general
    vk::ImageMemoryBarrier b2{};
    b2.setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(imgA.img.get())
        .setSubresourceRange(sr)
        .setSrcAccessMask(
            vk::AccessFlagBits::eColorAttachmentWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    vk::ImageMemoryBarrier b3{};
    b3.setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eGeneral)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(imgB.img.get())
        .setSubresourceRange(sr)
        .setSrcAccessMask({})
        .setDstAccessMask(vk::AccessFlagBits::eShaderWrite);
    std::array brs{b2, b3};
    c.pipelineBarrier(vk::PipelineStageFlagBits::
                          eColorAttachmentOutput,
                      vk::PipelineStageFlagBits::eComputeShader, {},
                      {}, {}, brs);

    c.bindPipeline(vk::PipelineBindPoint::eCompute, cpipe.get());
    c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                         clayout.get(), 0, dsets[0].get(), {});
    c.dispatch(S / 16, S / 16, 1);

    // imgB: general -> transfer src, copy to the readback buffer
    vk::ImageMemoryBarrier b4{};
    b4.setOldLayout(vk::ImageLayout::eGeneral)
        .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(imgB.img.get())
        .setSubresourceRange(sr)
        .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
        .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
    c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                      vk::PipelineStageFlagBits::eTransfer, {}, {},
                      {}, b4);
    vk::BufferImageCopy r{};
    r.setImageSubresource(
          {vk::ImageAspectFlagBits::eColor, 0, 0, 1})
        .setImageExtent(vk::Extent3D{S, S, 1});
    c.copyImageToBuffer(imgB.img.get(),
                        vk::ImageLayout::eTransferSrcOptimal,
                        hl.readback.buf.get(), r);
    c.end();
    vk::SubmitInfo si{};
    si.setCommandBuffers(c);
    auto f = vk.device->createFenceUnique({});
    vk.queue.submit(si, f.get());
    (void)vk.device->waitForFences(f.get(), true, UINT64_MAX);
    printf("wrote %s\n", ppmOut);
    hl.savePpm(ppmOut);
    return 0;
}
