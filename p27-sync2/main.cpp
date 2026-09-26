// SPDX-License-Identifier: CC0-1.0
//
// Synchronization2 (core in Vulkan 1.3, VK_KHR_synchronization2):
// vkCmdPipelineBarrier2 replaces the legacy vkCmdPipelineBarrier
// with VkImageMemoryBarrier2/VkBufferMemoryBarrier2 inside a
// VkDependencyInfo — the win is *precise* stage+access masks:
// "COMPUTE_SHADER wrote -> FRAGMENT_SHADER samples" instead of
// ALL_COMMANDS -> ALL_COMMANDS.
//
// Demo: compute writes a plasma into a storage image -> ONE
// barrier2 (compute write -> fragment sampled read) -> graphics
// pass samples it -> readback. Same workload as the storage-image
// post, every transition via barrier2.
//
//   ./app --headless o.ppm     one frame -> PPM for check.py
//   ./app                      windowed
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* kComp = R"GLSL(
#version 460
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0, rgba8) uniform writeonly image2D img;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(p) / 256.0;
    float v = 0.5 + 0.5 * sin(uv.x * 24.0) * cos(uv.y * 24.0);
    imageStore(img, p, vec4(v, uv.x, uv.y, 1.0));
}
)GLSL";

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
    // synchronization2 is a FEATURE: enabled like everything else
    vk::PhysicalDeviceSynchronization2Features sync2{};
    sync2.setSynchronization2(true);
    vk.createDevice({}, &sync2);

    // ---- storage image (compute writes, graphics samples) -------
    auto img = vk.createImage(
        256, 256, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eSampled);
    vk::ImageSubresourceRange sr{vk::ImageAspectFlagBits::eColor,
                                 0, 1, 0, 1};

    // ---- compute pipeline ----------------------------------------
    auto cs = vk.shader(kComp, shaderc_compute_shader, "s.comp");
    vk::DescriptorSetLayoutBinding cBind{
        0, vk::DescriptorType::eStorageImage, 1,
        vk::ShaderStageFlagBits::eCompute};
    vk::DescriptorSetLayoutCreateInfo cdslci{};
    cdslci.setBindings(cBind);
    auto cdsl = vk.device->createDescriptorSetLayoutUnique(cdslci);
    vk::PipelineLayoutCreateInfo cplci{};
    vk::DescriptorSetLayout cdslH = cdsl.get();
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

    // ---- graphics pipeline (samples the image) -------------------
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "s.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "s.frag");
    vk::DescriptorSetLayoutBinding gBind{
        0, vk::DescriptorType::eCombinedImageSampler, 1,
        vk::ShaderStageFlagBits::eFragment};
    vk::DescriptorSetLayoutCreateInfo gdslci{};
    gdslci.setBindings(gBind);
    auto gdsl = vk.device->createDescriptorSetLayoutUnique(gdslci);
    vk::PipelineLayoutCreateInfo gplci{};
    vk::DescriptorSetLayout gdslH = gdsl.get();
    gplci.setSetLayouts(gdslH);
    auto glayout = vk.device->createPipelineLayoutUnique(gplci);

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
    auto pres = vk.device->createGraphicsPipelineUnique({}, gi);
    if (pres.result != vk::Result::eSuccess)
        throw std::runtime_error("graphics pipeline failed");
    auto gpipe = std::move(pres.value);

    // ---- descriptors ----------------------------------------------
    vk::DescriptorPoolSize ps[2]{
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eCombinedImageSampler, 1}};
    auto dpool = vk.device->createDescriptorPoolUnique(
        {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 2,
         ps});
    vk::DescriptorSetLayout layouts[2]{cdsl.get(), gdsl.get()};
    vk::DescriptorSetAllocateInfo dsai{};
    dsai.setDescriptorPool(dpool.get()).setSetLayouts(layouts);
    auto dsets = vk.device->allocateDescriptorSetsUnique(dsai);
    auto sampler = vk.device->createSamplerUnique(
        {{},
         vk::Filter::eNearest,
         vk::Filter::eNearest,
         vk::SamplerMipmapMode::eNearest});
    vk::DescriptorImageInfo cimg{VK_NULL_HANDLE, img.view.get(),
                                 vk::ImageLayout::eGeneral};
    vk::DescriptorImageInfo gimg{sampler.get(), img.view.get(),
                                 vk::ImageLayout::eGeneral};
    vk::WriteDescriptorSet w[2];
    w[0].setDstSet(dsets[0].get())
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eStorageImage)
        .setImageInfo(cimg);
    w[1].setDstSet(dsets[1].get())
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setImageInfo(gimg);
    vk.device->updateDescriptorSets(w, {});

    // ---- one command buffer: compute -> barrier2 -> draw ----------
    vkmini::Headless hl;
    hl.init(vk, 384, 384);
    hl.render(
        vk::ImageLayout::eColorAttachmentOptimal,
        [&](vk::CommandBuffer c) {
            // 1) UNDEFINED -> GENERAL for storage write
            //    (barrier2: everything is a VkDependencyInfo)
            vk::ImageMemoryBarrier2 toGen{};
            toGen.setSrcStageMask(vk::PipelineStageFlagBits2::eNone)
                .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                .setDstStageMask(
                    vk::PipelineStageFlagBits2::eComputeShader)
                .setDstAccessMask(
                    vk::AccessFlagBits2::eShaderStorageWrite)
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eGeneral)
                .setImage(img.img.get())
                .setSubresourceRange(sr);
            vk::DependencyInfo dep{};
            dep.setImageMemoryBarriers(toGen);
            c.pipelineBarrier2(dep);

            c.bindPipeline(vk::PipelineBindPoint::eCompute,
                           cpipe.get());
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                 clayout.get(), 0, dsets[0].get(),
                                 {});
            c.dispatch(256 / 16, 256 / 16, 1);

            // 2) THE precise barrier: COMPUTE writes are done,
            //    FRAGMENT_SHADER may sample. Still eGeneral layout —
            //    a barrier2 can be layout-free (pure execution dep).
            vk::ImageMemoryBarrier2 c2f{};
            c2f.setSrcStageMask(
                    vk::PipelineStageFlagBits2::eComputeShader)
                .setSrcAccessMask(
                    vk::AccessFlagBits2::eShaderStorageWrite)
                .setDstStageMask(
                    vk::PipelineStageFlagBits2::eFragmentShader)
                .setDstAccessMask(vk::AccessFlagBits2::eShaderRead)
                .setOldLayout(vk::ImageLayout::eGeneral)
                .setNewLayout(vk::ImageLayout::eGeneral)
                .setImage(img.img.get())
                .setSubresourceRange(sr);
            vk::DependencyInfo dep2{};
            dep2.setImageMemoryBarriers(c2f);
            c.pipelineBarrier2(dep2);

            // 3) draw fullscreen triangle sampling the image
            vk::RenderingAttachmentInfo att{};
            att.setImageView(hl.color.view.get())
                .setImageLayout(
                    vk::ImageLayout::eColorAttachmentOptimal)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setClearValue(vk::ClearValue{vk::ClearColorValue{
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
                           gpipe.get());
            c.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                 glayout.get(), 0, dsets[1].get(),
                                 {});
            c.draw(3, 1, 0, 0);
            c.endRendering();
        });
    hl.savePpm(ppmOut);
    printf("wrote %s\n", ppmOut);
    return 0;
}
