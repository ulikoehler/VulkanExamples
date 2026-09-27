// SPDX-License-Identifier: CC0-1.0
//
// Zero-copy webcam capture: V4L2 exports its frame buffer as a
// dma-buf fd (VIDIOC_EXPBUF), which is imported directly as a
// VkImage — the CPU never sees a pixel. YUYV 4:2:2 lands in a
// VK_FORMAT_G8B8G8R8_422_UNORM image (each texel covers two
// luma samples) and the fragment shader does YUV->RGB.
//
//   ./app --headless o.ppm     one captured frame -> PPM for check.py
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
//
// Prints SKIP and exits 0 when no V4L2 device or no dmabuf export.

#include "vkmini.hpp"

#include <drm/drm_fourcc.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
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

// The dmabuf is imported as R8_UNORM: the 4-byte YUYV unit becomes
// four R8 texels. texelFetch hits raw bytes — parity picks the luma,
// the aligned 4-byte group supplies chroma. (The nicer
// G8B8G8R8_422_UNORM route is legal but unsupported on RADV — import
// as R8 is the portable fallback.)
static const char* kFrag = R"GLSL(
#version 460
layout(binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform PC { float width; float height; } pc;
float byteAt(uint bi, uint y) {                  // bi = byte index in row
    return texelFetch(tex, ivec2(int(bi), int(y)), 0).r * 255.0;
}
void main() {
    uint x = uint(vUV.x * pc.width);             // pixel column
    uint y = uint(vUV.y * pc.height);
    uint base = (x >> 1) * 4;                    // YUYV group: [Y0 U Y1 V]
    float Y = byteAt(base + (x & 1u) * 2u, y);
    float U = byteAt(base + 1u, y);
    float V = byteAt(base + 3u, y);
    vec3 yuv = vec3(Y, U, V) / 255.0 - vec3(0.0625, 0.5, 0.5);
    // BT.601 limited range, GLSL column-major: columns = (Y,U,V) coeffs
    vec3 rgb = clamp(mat3( 1.164,  1.164,  1.164,
                           0.000, -0.392,  2.017,
                           1.596, -0.813,  0.000) * yuv, 0.0, 1.0);
    outColor = vec4(rgb, 1);
}
)GLSL";

int main(int argc, char** argv) {
    const char* ppm = nullptr;
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--headless") && i + 1 < argc)
            ppm = argv[++i];

    // ---------------- V4L2: open, negotiate YUYV, export fd ------
    const uint32_t W = 640, H = 480;
    int vfd = open("/dev/video0", O_RDWR | O_CLOEXEC);
    if (vfd < 0) { printf("SKIP: no /dev/video0\n"); return 0; }

    v4l2_format vfmt{};
    vfmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    vfmt.fmt.pix = {.width = W, .height = H,
                    .pixelformat = V4L2_PIX_FMT_YUYV,
                    .field = V4L2_FIELD_ANY};
    if (ioctl(vfd, VIDIOC_S_FMT, &vfmt) < 0 ||
        vfmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
        printf("SKIP: camera has no YUYV\n");
        return 0;
    }

    v4l2_requestbuffers req{};
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(vfd, VIDIOC_REQBUFS, &req) < 0) {
        printf("SKIP: REQBUFS failed\n"); return 0;
    }
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(vfd, VIDIOC_QUERYBUF, &buf) < 0) {
        printf("SKIP: QUERYBUF failed\n"); return 0;
    }
    v4l2_exportbuffer eb{};
    eb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    eb.flags = O_RDONLY | O_CLOEXEC;
    if (ioctl(vfd, VIDIOC_EXPBUF, &eb) < 0) {
        printf("SKIP: VIDIOC_EXPBUF unsupported\n"); return 0;
    }
    int dmafd = eb.fd;
    printf("dmabuf fd %d, pitch %u, size %u\n", dmafd,
           vfmt.fmt.pix.bytesperline, vfmt.fmt.pix.sizeimage);

    // queue + stream on, dequeue one frame
    if (ioctl(vfd, VIDIOC_QBUF, &buf) < 0 ||
        ioctl(vfd, VIDIOC_STREAMON, &buf.type) < 0) {
        printf("SKIP: stream start failed\n"); return 0;
    }
    v4l2_buffer dq{};
    dq.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    dq.memory = V4L2_MEMORY_MMAP;
    if (ioctl(vfd, VIDIOC_DQBUF, &dq) < 0) {
        printf("SKIP: capture failed\n"); return 0;
    }
    printf("captured frame %ux%u\n", vfmt.fmt.pix.width,
           vfmt.fmt.pix.height);

    // ---------------- Vulkan: import the dma-buf ------------------
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice({VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                     VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
                     VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME});

    // the dmabuf is a linear byte buffer — import it as R8_UNORM with
    // twice the pixel width (2 bytes per YUYV pixel)
    const vk::Format imgFmt = vk::Format::eR8Unorm;
    vk::DrmFormatModifierPropertiesListEXT modList{};
    vk::FormatProperties2 fp2{};
    fp2.pNext = &modList;
    vk.phys.getFormatProperties2(imgFmt, &fp2);
    std::vector<vk::DrmFormatModifierPropertiesEXT> mods(
        modList.drmFormatModifierCount);
    modList.pDrmFormatModifierProperties = mods.data();
    vk.phys.getFormatProperties2(imgFmt, &fp2);
    bool linearOk = false;
    for (auto& m : mods)
        if (m.drmFormatModifier == DRM_FORMAT_MOD_LINEAR &&
            (m.drmFormatModifierTilingFeatures &
             vk::FormatFeatureFlagBits::eSampledImage))
            linearOk = true;
    if (!linearOk) { printf("SKIP: R8+LINEAR not samplable\n"); return 0; }

    vk::SubresourceLayout plane{};
    plane.offset = 0;
    plane.size = vfmt.fmt.pix.sizeimage;
    plane.rowPitch = vfmt.fmt.pix.bytesperline;
    vk::ImageDrmFormatModifierExplicitCreateInfoEXT expl{};
    expl.setDrmFormatModifier(DRM_FORMAT_MOD_LINEAR)
        .setDrmFormatModifierPlaneCount(1)
        .setPPlaneLayouts(&plane);
    vk::ExternalMemoryImageCreateInfo emi{};
    emi.setHandleTypes(
           vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT)
        .setPNext(&expl);
    vk::ImageCreateInfo ici{};
    ici.setImageType(vk::ImageType::e2D)
        .setFormat(imgFmt)
        .setExtent({W * 2, H, 1}) // texels = bytes (2 per pixel)
        .setMipLevels(1)
        .setArrayLayers(1)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eDrmFormatModifierEXT)
        .setUsage(vk::ImageUsageFlagBits::eSampled)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setPNext(&emi);
    auto img = vk.device->createImageUnique(ici);
    auto ireq = vk.device->getImageMemoryRequirements(img.get());

    vk::ImportMemoryFdInfoKHR imp{};
    imp.setHandleType(
           vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT)
        .setFd(dup(dmafd));
    vk::MemoryDedicatedAllocateInfo ded{};
    ded.setImage(img.get());
    imp.setPNext(&ded);
    vk::MemoryAllocateInfo mai{};
    mai.setAllocationSize(ireq.size)
        .setMemoryTypeIndex(
            vk.memoryType(ireq.memoryTypeBits,
                          vk::MemoryPropertyFlagBits::eDeviceLocal))
        .setPNext(&imp);
    auto mem = vk.device->allocateMemoryUnique(mai);
    vk.device->bindImageMemory(img.get(), mem.get(), 0);
    printf("imported dma-buf as VkImage\n");

    vk::ImageViewCreateInfo vi{};
    vi.setImage(img.get())
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(imgFmt)
        .setSubresourceRange(
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    auto view = vk.device->createImageViewUnique(vi);

    vk::ImageMemoryBarrier initB{};
    initB.setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eGeneral)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(img.get())
        .setSubresourceRange(
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})
        .setSrcAccessMask({})
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    vk.oneTime([&](vk::CommandBuffer c) {
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::eFragmentShader,
                          {}, {}, {}, initB);
    });

    // ---------------- descriptors + pipeline ----------------------
    auto sampler = vk.device->createSamplerUnique(
        {{}, vk::Filter::eNearest, vk::Filter::eNearest,
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
    vk::DescriptorPoolSize ps{vk::DescriptorType::eCombinedImageSampler,
                              1};
    auto dpool = vk.device->createDescriptorPoolUnique(
        {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1, ps});
    vk::DescriptorSetLayout dslH = dsl.get();
    vk::DescriptorSetAllocateInfo dsai{};
    dsai.setDescriptorPool(dpool.get()).setSetLayouts(dslH);
    auto dsets = vk.device->allocateDescriptorSetsUnique(dsai);
    vk::DescriptorImageInfo dii{sampler.get(), view.get(),
                                vk::ImageLayout::eGeneral};
    vk::WriteDescriptorSet wr{};
    wr.setDstSet(dsets[0].get())
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setImageInfo(dii);
    vk.device->updateDescriptorSets(wr, {});

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
    vk::PushConstantRange pcr{vk::ShaderStageFlagBits::eFragment, 0, 8};
    vk::PipelineLayoutCreateInfo plci{};
    plci.setSetLayouts(dslH).setPushConstantRanges(pcr);
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
    hl.init(vk, W, H);
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
                  vk::Viewport v{0, 0, float(W), float(H), 0.f, 1.f};
                  c.setViewport(0, v);
                  vk::Rect2D sc{{0, 0}, hl.extent};
                  c.setScissor(0, sc);
                  c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 pipe.get());
                  c.bindDescriptorSets(
                      vk::PipelineBindPoint::eGraphics, layout.get(),
                      0, dsets[0].get(), {});
                  float wh[2] = {float(W), float(H)};
                  c.pushConstants(layout.get(),
                                  vk::ShaderStageFlagBits::eFragment,
                                  0, 8, wh);
                  c.draw(3, 1, 0, 0);
                  c.endRendering();
              });
    hl.savePpm(ppm ? ppm : "out.ppm");
    printf("wrote %s\n", ppm ? ppm : "out.ppm");

    v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(vfd, VIDIOC_STREAMOFF, &t);
    close(dmafd);
    close(vfd);
    return 0;
}
