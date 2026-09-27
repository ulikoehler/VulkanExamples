// SPDX-License-Identifier: CC0-1.0
//
// FFmpeg + VA-API decode -> DMA-Buf -> VkImage, zero copies.
// This is the same path a camera wall uses for RTSP streams:
//   H.264 file -> FFmpeg demux -> VA-API decode (GPU)
//     -> av_hwframe_map -> AVDRMFrameDescriptor (dma-buf fds)
//     -> per-plane VkImage import -> sampled in a fragment shader
//
//   ./app --headless o.ppm     first decoded frame -> PPM
//   ./app --headless o.ppm myfile.h264
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app \
//             -lvulkan -lglfw -lshaderc -lavcodec -lavformat -lavutil
//
// Prints SKIP when VA-API or the import is unavailable.

#include "vkmini.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/pixdesc.h>
}
#include <drm/drm_fourcc.h>
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

// NV12: binding 0 = Y plane (R8), binding 1 = UV plane (R8G8).
static const char* kFrag = R"GLSL(
#version 460
layout(binding = 0) uniform sampler2D texY;
layout(binding = 1) uniform sampler2D texUV;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() {
    float Y = texture(texY, vUV).r;
    vec2  UV = texture(texUV, vUV).rg;   // interleaved U,V half-res
    vec3 yuv = vec3(Y, UV.x, UV.y) - vec3(0.0625, 0.5, 0.5);
    // BT.601 limited range; mat3 is column-major
    vec3 rgb = clamp(mat3( 1.164,  1.164,  1.164,
                           0.000, -0.392,  2.017,
                           1.596, -0.813,  0.000) * yuv, 0.0, 1.0);
    outColor = vec4(rgb, 1);
}
)GLSL";

/// Import one dma-buf plane as a single-plane VkImage (R8 / R8G8).
struct ImportedPlane {
    vk::UniqueImage img;
    vk::UniqueDeviceMemory mem;
    vk::UniqueImageView view;
};

static ImportedPlane importPlane(vkmini::Vk& vk,
                                 vk::detail::DispatchLoaderDynamic& dldi,
                                 int fd, uint32_t offset,
                                 uint32_t pitch, uint64_t modifier,
                                 vk::Format fmt, uint32_t w, uint32_t h) {
    // which memory types can host this foreign fd?
    vk::MemoryFdPropertiesKHR fdProps =
        vk.device->getMemoryFdPropertiesKHR(
            vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT, fd, dldi);

    vk::SubresourceLayout plane{};
    plane.offset = offset;
    plane.size = uint64_t(pitch) * h;
    plane.rowPitch = pitch;
    plane.arrayPitch = plane.size;  // modifier validation wants these
    plane.depthPitch = plane.size;
    vk::ImageDrmFormatModifierExplicitCreateInfoEXT expl{};
    expl.setDrmFormatModifier(modifier)
        .setDrmFormatModifierPlaneCount(1)
        .setPPlaneLayouts(&plane);
    vk::ExternalMemoryImageCreateInfo emi{};
    emi.setHandleTypes(
            vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT)
        .setPNext(&expl);
    vk::ImageCreateInfo ici{};
    ici.setImageType(vk::ImageType::e2D)
        .setFormat(fmt)
        .setExtent({w, h, 1})
        .setMipLevels(1)
        .setArrayLayers(1)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eDrmFormatModifierEXT)
        .setUsage(vk::ImageUsageFlagBits::eSampled)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setPNext(&emi);

    ImportedPlane p;
    p.img = vk.device->createImageUnique(ici);
    auto req = vk.device->getImageMemoryRequirements(p.img.get());
    uint32_t typeBits = req.memoryTypeBits & fdProps.memoryTypeBits;
    if (!typeBits) typeBits = req.memoryTypeBits; // fd props too strict
    // dma-buf memory may live in GTT/host-visible heaps — do NOT
    // require DEVICE_LOCAL here.

    vk::ImportMemoryFdInfoKHR imp{};
    imp.setHandleType(vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT)
        .setFd(dup(fd)); // import consumes the fd — keep ours
    vk::MemoryDedicatedAllocateInfo ded{};
    ded.setImage(p.img.get()); // required for explicit-modifier imports
    imp.setPNext(&ded);
    vk::MemoryAllocateInfo mai{};
    // the plane sits at `offset` inside the dma-buf object — the
    // import must cover offset + plane size, not just req.size
    mai.setAllocationSize(std::max(req.size, uint64_t(offset) + plane.size))
        .setMemoryTypeIndex(vk.memoryType(typeBits, {}))
        .setPNext(&imp);
    p.mem = vk.device->allocateMemoryUnique(mai);
    vk.device->bindImageMemory(p.img.get(), p.mem.get(), 0);

    vk::ImageViewCreateInfo vi{};
    vi.setImage(p.img.get())
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(fmt)
        .setSubresourceRange(
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    p.view = vk.device->createImageViewUnique(vi);
    return p;
}

int main(int argc, char** argv) {
    const char* ppm = "out.ppm";
    const char* input = "test.h264";
    const char* vaapiDev = "/dev/dri/renderD129";
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--headless") && i + 1 < argc)
            ppm = argv[++i];
        else if (!strcmp(argv[i], "--vaapi") && i + 1 < argc)
            vaapiDev = argv[++i];
        else if (argv[i][0] != '-')
            input = argv[i];

    // ---------------- FFmpeg: demux + VA-API decode --------------
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, input, nullptr, nullptr) < 0) {
        printf("SKIP: cannot open %s\n", input); return 0;
    }
    avformat_find_stream_info(fmt, nullptr);
    int vstream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1,
                                 nullptr, 0);
    const AVCodec* codec =
        avcodec_find_decoder(fmt->streams[vstream]->codecpar->codec_id);
    AVCodecContext* dec = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(dec, fmt->streams[vstream]->codecpar);

    AVBufferRef* hwDev = nullptr;
    if (av_hwdevice_ctx_create(&hwDev, AV_HWDEVICE_TYPE_VAAPI,
                               vaapiDev, nullptr, 0) < 0) {
        printf("SKIP: no usable VA-API device\n"); return 0;
    }
    dec->hw_device_ctx = av_buffer_ref(hwDev);
    dec->get_format = [](AVCodecContext*, const AVPixelFormat* f) {
        while (*f != AV_PIX_FMT_NONE && *f != AV_PIX_FMT_VAAPI) ++f;
        return *f; // force VAAPI or give up
    };
    if (avcodec_open2(dec, codec, nullptr) < 0) {
        printf("SKIP: decoder open failed\n"); return 0;
    }

    // decode through the whole clip; keep the LAST complete frame —
    // the VAAPI surface must be finished before we map its dma-buf
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* last = av_frame_alloc();
    AVFrame* mapped = av_frame_alloc();
    int nFrames = 0;
    auto drain = [&] {
        while (avcodec_receive_frame(dec, frame) == 0) {
            av_frame_unref(last);
            av_frame_move_ref(last, frame); // keep the surface alive
            ++nFrames;
        }
    };
    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index != vstream) { av_packet_unref(pkt); continue; }
        avcodec_send_packet(dec, pkt);
        av_packet_unref(pkt);
        drain();
    }
    avcodec_send_packet(dec, nullptr); // flush
    drain();
    if (!nFrames) { printf("SKIP: no frame decoded\n"); return 0; }
    frame = last;
    printf("decoded %d frames, last is %dx%d fmt=%s\n", nFrames,
           frame->width, frame->height,
           av_get_pix_fmt_name((AVPixelFormat)frame->format));

    // hw frame -> DRM-PRIME descriptor (dma-buf fds per plane)
    mapped->format = AV_PIX_FMT_DRM_PRIME; // request DRM-PRIME mapping
    int mapErr = av_hwframe_map(mapped, frame, AV_HWFRAME_MAP_READ);
    if (mapErr < 0 || mapped->format != AV_PIX_FMT_DRM_PRIME) {
        char eb[AV_ERROR_MAX_STRING_SIZE]{};
        av_strerror(mapErr, eb, sizeof(eb));
        printf("SKIP: hwframe_map failed: %s (got fmt=%s)\n", eb,
               av_get_pix_fmt_name((AVPixelFormat)mapped->format));
        return 0;
    }
    auto* desc =
        reinterpret_cast<AVDRMFrameDescriptor*>(mapped->data[0]);
    printf("drm desc: %d layers, %d objects\n", desc->nb_layers,
           desc->nb_objects);
    for (int l = 0; l < desc->nb_layers; ++l)
        for (int p = 0; p < desc->layers[l].nb_planes; ++p)
            printf("  layer%d plane%d: obj%d off=%u pitch=%u mod=0x%lx\n",
                   l, p, desc->layers[l].planes[p].object_index,
                   (unsigned)desc->layers[l].planes[p].offset,
                   (unsigned)desc->layers[l].planes[p].pitch,
                   (uint64_t)desc->objects[desc->layers[l].planes[p]
                                               .object_index]
                       .format_modifier);

    // ---------------- Vulkan -------------------------------------
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice({VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                     VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
                     VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME});
    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr, vk.device.get());

    // VAAPI NV12 arrives as two single-plane layers:
    //   layer 0: R8   (luma, WxH)
    //   layer 1: GR88 (interleaved U,V — that's R8G8, W/2 x H/2)
    if (desc->nb_layers < 2) {
        printf("SKIP: expected per-plane layers, got %d\n",
               desc->nb_layers);
        return 0;
    }
    auto fdOf = [&](int layer) {
        return desc->objects[desc->layers[layer].planes[0]
                                 .object_index].fd;
    };
    auto modOf = [&](int layer) {
        return desc->objects[desc->layers[layer].planes[0]
                                 .object_index].format_modifier;
    };
    const uint32_t W = frame->width, Hh = frame->height;
    auto imgY = importPlane(
        vk, dldi, fdOf(0), desc->layers[0].planes[0].offset,
        desc->layers[0].planes[0].pitch, modOf(0),
        vk::Format::eR8Unorm, W, Hh);
    auto imgUV = importPlane(
        vk, dldi, fdOf(1), desc->layers[1].planes[0].offset,
        desc->layers[1].planes[0].pitch, modOf(1),
        vk::Format::eR8G8Unorm, W / 2, Hh / 2);
    printf("imported 2 planes as VkImages (modifier 0x%lx)\n",
           (uint64_t)modOf(0));

    // descriptor set with both plane views
    auto sampler = vk.device->createSamplerUnique(
        {{}, vk::Filter::eLinear, vk::Filter::eLinear,
         vk::SamplerMipmapMode::eNearest,
         vk::SamplerAddressMode::eClampToEdge,
         vk::SamplerAddressMode::eClampToEdge,
         vk::SamplerAddressMode::eClampToEdge});
    std::array bindings{
        vk::DescriptorSetLayoutBinding{
            0, vk::DescriptorType::eCombinedImageSampler, 1,
            vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{
            1, vk::DescriptorType::eCombinedImageSampler, 1,
            vk::ShaderStageFlagBits::eFragment}};
    vk::DescriptorSetLayoutCreateInfo dslci{};
    dslci.setBindings(bindings);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dslci);
    vk::DescriptorPoolSize ps{vk::DescriptorType::eCombinedImageSampler,
                              2};
    auto dpool = vk.device->createDescriptorPoolUnique(
        {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1, ps});
    vk::DescriptorSetLayout dslH = dsl.get();
    vk::DescriptorSetAllocateInfo dsai{};
    dsai.setDescriptorPool(dpool.get()).setSetLayouts(dslH);
    auto dsets = vk.device->allocateDescriptorSetsUnique(dsai);
    std::array imgs{vk::DescriptorImageInfo{sampler.get(),
                                            imgY.view.get(),
                                            vk::ImageLayout::eGeneral},
                    vk::DescriptorImageInfo{sampler.get(),
                                            imgUV.view.get(),
                                            vk::ImageLayout::eGeneral}};
    std::array wr{
        vk::WriteDescriptorSet{dsets[0].get(), 0, 0,
                               vk::DescriptorType::eCombinedImageSampler,
                               imgs[0]},
        vk::WriteDescriptorSet{dsets[0].get(), 1, 0,
                               vk::DescriptorType::eCombinedImageSampler,
                               imgs[1]}};
    vk.device->updateDescriptorSets(wr, {});

    // transition both imported planes UNDEFINED -> GENERAL
    vk.oneTime([&](vk::CommandBuffer c) {
        std::array bs{
            vk::ImageMemoryBarrier{},
            vk::ImageMemoryBarrier{}};
        for (int i = 0; i < 2; ++i) {
            bs[i].setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eGeneral)
                .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setImage(i ? imgUV.img.get() : imgY.img.get())
                .setSubresourceRange(
                    {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})
                .setSrcAccessMask({})
                .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        }
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::eFragmentShader,
                          {}, {}, {}, bs);
    });

    // ---------------- pipeline + draw ----------------------------
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "s.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "s.frag");
    vk::PipelineShaderStageCreateInfo stages[2];
    stages[0].setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(vs.get()).setPName("main");
    stages[1].setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(fs.get()).setPName("main");
    vk::PipelineVertexInputStateCreateInfo vin{};
    vk::PipelineInputAssemblyStateCreateInfo ia{};
    ia.setTopology(vk::PrimitiveTopology::eTriangleList);
    vk::PipelineViewportStateCreateInfo vps{};
    vps.setViewportCount(1).setScissorCount(1);
    std::array dynSt{vk::DynamicState::eViewport,
                     vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dyn{};
    dyn.setDynamicStates(dynSt);
    vk::PipelineRasterizationStateCreateInfo rs{};
    rs.setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eNone)
        .setFrontFace(vk::FrontFace::eCounterClockwise)
        .setLineWidth(1.0f);
    vk::PipelineMultisampleStateCreateInfo ms{};
    ms.setRasterizationSamples(vk::SampleCountFlagBits::e1);
    vk::PipelineColorBlendAttachmentState cb{};
    cb.setColorWriteMask(vk::ColorComponentFlagBits::eR |
                         vk::ColorComponentFlagBits::eG |
                         vk::ColorComponentFlagBits::eB |
                         vk::ColorComponentFlagBits::eA);
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
        .setPVertexInputState(&vin).setPInputAssemblyState(&ia)
        .setPViewportState(&vps).setPDynamicState(&dyn)
        .setPRasterizationState(&rs).setPMultisampleState(&ms)
        .setPColorBlendState(&blend).setLayout(layout.get())
        .setPNext(&rendering);
    auto pres = vk.device->createGraphicsPipelineUnique({}, gi);
    if (pres.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    auto pipe = std::move(pres.value);

    vkmini::Headless hl;
    hl.init(vk, W, Hh);
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
                      .setLayerCount(1).setColorAttachments(att);
                  c.beginRendering(ri);
                  vk::Viewport v{0, 0, float(W), float(Hh), 0.f, 1.f};
                  c.setViewport(0, v);
                  vk::Rect2D sc{{0,0}, hl.extent}; c.setScissor(0, sc);
                  c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 pipe.get());
                  c.bindDescriptorSets(
                      vk::PipelineBindPoint::eGraphics, layout.get(),
                      0, dsets[0].get(), {});
                  c.draw(3, 1, 0, 0);
                  c.endRendering();
              });
    hl.savePpm(ppm);
    printf("wrote %s\n", ppm);
    return 0;
}
