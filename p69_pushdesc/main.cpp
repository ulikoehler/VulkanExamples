// SPDX-License-Identifier: CC0-1.0
//
// VK_KHR_push_descriptor — descriptors written straight into the
// command buffer. No VkDescriptorPool, no vkAllocateDescriptorSets,
// no vkUpdateDescriptorSets: the app hands vkCmdPushDescriptorSetKHR
// an ordinary VkWriteDescriptorSet at record time.
//
//   ./app --headless out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0, 1);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec2 vUV;
layout(set = 0, binding = 0) uniform sampler2D tex;
layout(location = 0) out vec4 outColor;
void main() { outColor = texture(tex, vUV); }
)GLSL";

/// Same checkerboard upload as post 5 (staging -> TRANSFER_DST ->
/// SHADER_READ). Nothing push-descriptor-specific here.
static vkmini::Vk::Image uploadTexture(vkmini::Vk& vk, uint32_t w,
                                       uint32_t h, const void* px,
                                       size_t size) {
    auto staging = vk.createBuffer(
        size, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(staging.mapped, px, size);
    auto img = vk.createImage(
        w, h, vk::Format::eR8G8B8A8Unorm,
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
        r.setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
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

int main(int argc, char** argv) {
    const char* ppm = "out.ppm";
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--headless") && i + 1 < argc)
            ppm = argv[++i];

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    // The extension has no feature bit — but the PROPERTY limits how
    // many descriptors one push may carry.
    auto props = vk.phys.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDevicePushDescriptorPropertiesKHR>();
    auto& pd = props.get<
        vk::PhysicalDevicePushDescriptorPropertiesKHR>();
    printf("maxPushDescriptors = %u\n", pd.maxPushDescriptors);
    if (pd.maxPushDescriptors < 1) {
        printf("SKIP: no push descriptor support\n");
        return 0;
    }
    vk.createDevice({VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME});
    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr, vk.device.get());

    // 8x8 checkerboard texture.
    const uint32_t TW = 64, TH = 64;
    std::vector<uint32_t> px(TW * TH);
    for (uint32_t y = 0; y < TH; ++y)
        for (uint32_t x = 0; x < TW; ++x)
            px[y * TW + x] = ((x / 8 + y / 8) & 1)
                                 ? 0xFF00FF00u       // green
                                 : 0xFF0000FFu;      // red (RGBA)
    auto tex = uploadTexture(vk, TW, TH, px.data(), px.size() * 4);

    vk::SamplerCreateInfo sci{};
    sci.setMagFilter(vk::Filter::eNearest)
        .setMinFilter(vk::Filter::eNearest);
    auto sampler = vk.device->createSamplerUnique(sci);

    // The set layout is marked PUSH_DESCRIPTOR — it can never back an
    // allocated VkDescriptorSet; it only describes what a push may
    // contain (and goes into the pipeline layout as usual).
    vk::DescriptorSetLayoutBinding b0{};
    b0.setBinding(0)
        .setDescriptorType(
            vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eFragment);
    vk::DescriptorSetLayoutCreateInfo sli{};
    sli.setFlags(vk::DescriptorSetLayoutCreateFlagBits::
                     ePushDescriptorKHR)
        .setBindings(b0);
    auto setLayout =
        vk.device->createDescriptorSetLayoutUnique(sli);

    auto vs = vk.shader(kVert, shaderc_vertex_shader, "v.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "f.frag");
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
    vk::PipelineLayoutCreateInfo li{};
    li.setSetLayouts(setLayout.get());
    auto layout = vk.device->createPipelineLayoutUnique(li);
    vk::PipelineRenderingCreateInfo rendering{};
    std::array cf{vk::Format::eR8G8B8A8Unorm};
    rendering.setColorAttachmentFormats(cf);
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

    vkmini::Headless h;
    h.init(vk, 320, 240);
    h.render(vk::ImageLayout::eColorAttachmentOptimal,
             [&](vk::CommandBuffer c) {
        vk::RenderingAttachmentInfo att{};
        att.setImageView(h.color.view.get())
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore);
        vk::RenderingInfo ri{};
        ri.setRenderArea({{0, 0}, h.extent})
            .setLayerCount(1)
            .setColorAttachments(att);
        c.beginRendering(ri);
        c.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe.get());

        // ---- THE extension: write the descriptor at record time ----
        vk::DescriptorImageInfo ii{};
        ii.setSampler(sampler.get())
            .setImageView(tex.view.get())
            .setImageLayout(
                vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::WriteDescriptorSet wr{};
        wr.setDstSet({})  // no set exists — dstSet is ignored
            .setDstBinding(0)
            .setDescriptorType(
                vk::DescriptorType::eCombinedImageSampler)
            .setImageInfo(ii);
        c.pushDescriptorSetKHR(
            vk::PipelineBindPoint::eGraphics, layout.get(), 0, wr,
            dldi);

        vk::Viewport vpv{0, 0, float(h.extent.width),
                         float(h.extent.height), 0, 1};
        c.setViewport(0, vpv);
        c.setScissor(0, vk::Rect2D{{0, 0}, h.extent});
        c.draw(3, 1, 0, 0);
        c.endRendering();
    });
    h.savePpm(ppm);
    printf("done: pushed sampler+image at record time, no pool, no set\n");
    return 0;
}
