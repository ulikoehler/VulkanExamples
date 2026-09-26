// SPDX-License-Identifier: CC0-1.0
//
// VK_EXT_host_image_copy: vkCopyMemoryToImageEXT / vkCopyImageToMemoryEXT
// — copying between a host pointer and a VkImage WITHOUT a command
// buffer, staging buffer, or queue submission. The driver does the
// layout handling internally (it's allowed to be slow — this is the
// simple path, not the fast path).
//
// Demo: CPU array -> vkCopyMemoryToImageEXT -> sample -> PPM;
// then vkCopyImageToMemoryEXT reads the image back and we memcmp.
//
//   ./app --headless o.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>
#include <cstring>

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
void main() { outColor = texture(tex, vUV); }
)GLSL";

int main() {
    setbuf(stdout, nullptr);
    vkmini::Vk vk;
    vk.createInstance({});
    // pick the first device that SUPPORTS host image copies —
    // typically iGPUs where "host memory" already IS GPU-visible.
    for (auto& d : vk.instance->enumeratePhysicalDevices()) {
        auto exts = d.enumerateDeviceExtensionProperties();
        for (auto& e : exts)
            if (!strcmp(e.extensionName,
                        VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME)) {
                vk.phys = d;
                break;
            }
        if (vk.phys) break;
    }
    if (!vk.phys) {
        printf("SKIP: no device supports VK_EXT_host_image_copy\n");
        return 0;
    }
    printf("device: %s\n",
           vk.phys.getProperties().deviceName.data());

    vk::PhysicalDeviceHostImageCopyFeaturesEXT hic{};
    hic.setHostImageCopy(VK_TRUE);
    vk::PhysicalDeviceVulkan13Features f13{};
    f13.setPNext(&hic);
    vk.createDevice({VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME}, &f13);

    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr, vk.device.get());

    const uint32_t W = 8, H = 8;
    std::array<uint32_t, W * H> src;
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x)
            src[y * W + x] = ((x + y) & 1) ? 0xFF00FF00u
                                         : 0xFF0000FFu;

    // usage bit eHostTransferEXT is required for host copies
    auto tex = vk.createImage(
        W, H, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eHostTransferEXT);

    // host copies need eGeneral (or a layout listed in
    // VkPhysicalDeviceHostImageCopyPropertiesEXT::pCopyDstLayouts)
    vk::ImageSubresourceRange sr{vk::ImageAspectFlagBits::eColor,
                                 0, 1, 0, 1};
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::ImageMemoryBarrier b{};
        b.setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eGeneral)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(tex.img.get())
            .setSubresourceRange(sr);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::eBottomOfPipe,
                          {}, {}, {}, b);
    });

    // ---- THE HOST COPY: no command buffer involved ---------------
    vk::MemoryToImageCopyEXT region{};
    region.setPHostPointer(src.data())
        .setMemoryRowLength(0)      // 0 = tightly packed
        .setMemoryImageHeight(0)
        .setImageSubresource(
            {vk::ImageAspectFlagBits::eColor, 0, 0, 1})
        .setImageExtent({W, H, 1});
    vk::CopyMemoryToImageInfoEXT cmi{};
    cmi.setDstImage(tex.img.get())
        .setDstImageLayout(vk::ImageLayout::eGeneral)
        .setRegions(region);
    vk.device->copyMemoryToImageEXT(cmi, dldi);
    printf("host->image copy done (no command buffer)\n");

    // ---- sample it -----------------------------------------------
    auto sampler = vk.device->createSamplerUnique(
        {{},
         vk::Filter::eNearest,
         vk::Filter::eNearest,
         vk::SamplerMipmapMode::eNearest,
         vk::SamplerAddressMode::eClampToEdge,
         vk::SamplerAddressMode::eClampToEdge,
         vk::SamplerAddressMode::eClampToEdge});
    vk::DescriptorSetLayoutBinding binding{
        0, vk::DescriptorType::eCombinedImageSampler, 1,
        vk::ShaderStageFlagBits::eFragment};
    vk::DescriptorSetLayoutCreateInfo dslci{};
    dslci.setBindings(binding);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dslci);
    vk::DescriptorPoolSize ps{
        vk::DescriptorType::eCombinedImageSampler, 1};
    auto dpool = vk.device->createDescriptorPoolUnique(
        {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1,
         ps});
    vk::DescriptorSetLayout dslH = dsl.get();
    vk::DescriptorSetAllocateInfo dsai{};
    dsai.setDescriptorPool(dpool.get()).setSetLayouts(dslH);
    auto dsets = vk.device->allocateDescriptorSetsUnique(dsai);
    vk::DescriptorImageInfo dii{sampler.get(), tex.view.get(),
                                vk::ImageLayout::eGeneral};
    vk::WriteDescriptorSet w{};
    w.setDstSet(dsets[0].get())
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setImageInfo(dii);
    vk.device->updateDescriptorSets(w, {});

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
                      .setStoreOp(vk::AttachmentStoreOp::eStore);
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

    // ---- readback via vkCopyImageToMemoryEXT ---------------------
    std::array<uint32_t, W * H> back{};
    vk::ImageToMemoryCopyEXT r2{};
    r2.setPHostPointer(back.data())
        .setImageSubresource(
            {vk::ImageAspectFlagBits::eColor, 0, 0, 1})
        .setImageExtent({W, H, 1});
    vk::CopyImageToMemoryInfoEXT imc{};
    imc.setSrcImage(tex.img.get())
        .setSrcImageLayout(vk::ImageLayout::eGeneral)
        .setRegions(r2);
    vk.device->copyImageToMemoryEXT(imc, dldi);
    int diff = memcmp(src.data(), back.data(), sizeof(src));
    printf("image->host roundtrip: %s\n", diff ? "MISMATCH" : "exact");
    printf("wrote out.ppm\n");
    return diff != 0;
}
