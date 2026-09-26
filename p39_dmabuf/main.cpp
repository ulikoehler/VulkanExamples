// SPDX-License-Identifier: CC0-1.0
//
// Zero-copy DMA-Buf sharing: image A exports its memory as a
// dma-buf fd (vkGetMemoryFdKHR), image B imports the SAME fd via
// VK_EXT_external_memory_dma_buf + VK_EXT_image_drm_format_modifier.
// Content written through A is sampled through B — no CPU copy.
// This is the same mechanism a video decoder's dma-buf export uses;
// here both endpoints are Vulkan so the demo is self-contained.
//
//   ./app --headless o.ppm     one frame -> PPM for check.py
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>
#include <unistd.h>

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
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice({VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                     VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
                     VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME});

    // extension commands need dynamic dispatch
    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr, vk.device.get());

    const uint32_t W = 64, H = 64;
    const vk::Format fmt = vk::Format::eR8G8B8A8Unorm;

    // ---- discover supported DRM modifiers for this format --------
    vk::DrmFormatModifierPropertiesListEXT modList{};
    vk::FormatProperties2 fp2{};
    fp2.pNext = &modList;
    // first query the count
    vk.phys.getFormatProperties2(fmt, &fp2);
    std::vector<vk::DrmFormatModifierPropertiesEXT> mods(
        modList.drmFormatModifierCount);
    modList.pDrmFormatModifierProperties = mods.data();
    vk.phys.getFormatProperties2(fmt, &fp2);
    uint64_t modifier = 0;
    for (auto& m : mods)
        printf("  mod 0x%016lx planes %u feats 0x%x\n",
               (uint64_t)m.drmFormatModifier,
               m.drmFormatModifierPlaneCount,
               (uint32_t)m.drmFormatModifierTilingFeatures);
    for (auto& m : mods) {
        auto need = vk::FormatFeatureFlagBits::eSampledImage |
                    vk::FormatFeatureFlagBits::eTransferDst;
        if ((m.drmFormatModifierTilingFeatures & need) == need &&
            m.drmFormatModifierPlaneCount == 1) {
            modifier = m.drmFormatModifier;
            break;
        }
    }
    if (!modifier && !mods.empty())
        modifier = mods[0].drmFormatModifier;
    printf("modifier: 0x%016lx (%zu supported)\n", modifier,
           mods.size());
    fflush(stdout);

    const vk::ImageUsageFlags kUsage =
        vk::ImageUsageFlagBits::eSampled |
        vk::ImageUsageFlagBits::eTransferDst;

    // ---- image A: exportable, DRM-modifier tiled -----------------
    vk::ExternalMemoryImageCreateInfo emi{};
    emi.setHandleTypes(
        vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT);
    vk::ImageDrmFormatModifierListCreateInfoEXT mli{};
    mli.setDrmFormatModifiers(modifier);
    emi.setPNext(&mli);
    vk::ImageCreateInfo ici{};
    ici.setImageType(vk::ImageType::e2D)
        .setFormat(fmt)
        .setExtent({W, H, 1})
        .setMipLevels(1)
        .setArrayLayers(1)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eDrmFormatModifierEXT)
        .setUsage(kUsage)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setPNext(&emi);
    auto imgA = vk.device->createImageUnique(ici);
    auto reqA =
        vk.device->getImageMemoryRequirements(imgA.get());
    vk::ExportMemoryAllocateInfo exAlloc{};
    exAlloc.setHandleTypes(
        vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT);
    vk::MemoryAllocateInfo mai{};
    mai.setAllocationSize(reqA.size)
        .setMemoryTypeIndex(vk.memoryType(
            reqA.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eDeviceLocal))
        .setPNext(&exAlloc);
    auto memA = vk.device->allocateMemoryUnique(mai);
    vk.device->bindImageMemory(imgA.get(), memA.get(), 0);

    // ---- export: memory -> dma-buf fd ------------------------------
    vk::MemoryGetFdInfoKHR fdInfo{};
    fdInfo.setMemory(memA.get())
        .setHandleType(
            vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT);
    int fd = vk.device->getMemoryFdKHR(fdInfo, dldi);
    printf("exported dma-buf fd %d\n", fd);
    fflush(stdout);

    // ---- which modifier + plane layout did the driver pick? ------
    auto modProps = vk.device->getImageDrmFormatModifierPropertiesEXT(
        imgA.get(), dldi);
    vk::ImageSubresource plane0{vk::ImageAspectFlagBits::eMemoryPlane0EXT,
                                0, 0};
    vk::SubresourceLayout planeLayout =
        vk.device->getImageSubresourceLayout(imgA.get(), plane0);
    printf("selected modifier 0x%016lx, rowPitch %llu, size %llu\n",
           (uint64_t)modProps.drmFormatModifier,
           (unsigned long long)planeLayout.rowPitch,
           (unsigned long long)planeLayout.size);
    fflush(stdout);

    // ---- fill A through a normal staging upload -------------------
    // DRM-modifier images may only use eGeneral/ePresentSrc layouts.
    std::array<uint32_t, W * H> src;
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x)
            src[y * W + x] =
                ((x / 8 + y / 8) & 1) ? 0xFF33FF33u : 0xFF3333FFu;
    auto st = vk.createBuffer(
        src.size() * 4, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(st.mapped, src.data(), src.size() * 4);
    vk::ImageSubresourceRange sr{vk::ImageAspectFlagBits::eColor,
                                 0, 1, 0, 1};
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::ImageMemoryBarrier b{};
        b.setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eGeneral)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(imgA.get())
            .setSubresourceRange(sr)
            .setSrcAccessMask({})
            .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::eTransfer, {},
                          {}, {}, b);
        vk::BufferImageCopy cp{};
        cp.setImageSubresource({vk::ImageAspectFlagBits::eColor, 0,
                                0, 1})
            .setImageExtent({W, H, 1});
        c.copyBufferToImage(st.buf.get(), imgA.get(),
                            vk::ImageLayout::eGeneral, cp);
        vk::ImageMemoryBarrier b2{};
        b2.setOldLayout(vk::ImageLayout::eGeneral)
            .setNewLayout(vk::ImageLayout::eGeneral)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(imgA.get())
            .setSubresourceRange(sr)
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                          vk::PipelineStageFlagBits::eFragmentShader,
                          {}, {}, {}, b2);
    });

    // ---- image B: IMPORT the same fd -------------------------------
    vk::ImageDrmFormatModifierExplicitCreateInfoEXT expl{};
    vk::SubresourceLayout impPlane{planeLayout.offset,
                                   planeLayout.size,
                                   planeLayout.rowPitch, 0, 0};
    expl.setDrmFormatModifier(modProps.drmFormatModifier)
        .setDrmFormatModifierPlaneCount(1)
        .setPPlaneLayouts(&impPlane);
    vk::ExternalMemoryImageCreateInfo emiB{};
    emiB.setHandleTypes(
        vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT)
        .setPNext(&expl);
    vk::ImageCreateInfo iciB = ici;
    iciB.setPNext(&emiB);
    auto imgB = vk.device->createImageUnique(iciB);
    auto reqB = vk.device->getImageMemoryRequirements(imgB.get());
    vk::ImportMemoryFdInfoKHR imp{};
    imp.setHandleType(
           vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT)
        .setFd(dup(fd));   // ownership: Vulkan consumes the fd
    // modifier-explicit imports must be dedicated allocations
    vk::MemoryDedicatedAllocateInfo ded{};
    ded.setImage(imgB.get());
    imp.setPNext(&ded);
    vk::MemoryAllocateInfo maiB{};
    maiB.setAllocationSize(reqB.size)
        .setMemoryTypeIndex(vk.memoryType(
            reqB.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eDeviceLocal))
        .setPNext(&imp);
    auto memB = vk.device->allocateMemoryUnique(maiB);
    vk.device->bindImageMemory(imgB.get(), memB.get(), 0);
    // `imp.fd` was consumed by the driver; the original stays ours:
    close(fd);

    vk::ImageViewCreateInfo vi{};
    vi.setImage(imgB.get())
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(fmt)
        .setSubresourceRange(sr);
    auto viewB = vk.device->createImageViewUnique(vi);

    // imported images are born eGeneral-per-spec; no upload needed —
    // B sees A's pixels because it IS A's memory.
    vk::ImageMemoryBarrier initB{};
    initB.setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eGeneral)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(imgB.get())
        .setSubresourceRange(sr)
        .setSrcAccessMask({})
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    vk.oneTime([&](vk::CommandBuffer c) {
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::eFragmentShader,
                          {}, {}, {}, initB);
    });

    // ---- descriptor + pipeline + fullscreen draw ------------------
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
    vk::DescriptorImageInfo dii{sampler.get(), viewB.get(),
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
    printf("wrote out.ppm\n");
    return 0;
}
