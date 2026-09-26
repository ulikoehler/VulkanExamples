// SPDX-License-Identifier: CC0-1.0
//
// Post 20: encode the swapchain readback as PNG via libpng.
// Same capture as the screenshot post (eTransferSrc usage +
// layout transitions + copyImageToBuffer), but the host-visible
// buffer is written as a real .png instead of PPM — including the
// BGRA/10-bit decode and RGBA channel shuffling libpng needs.
//
//   xvfb-run -a ./app            windowed, saves shot.png at frame 60
//   python3 check.py shot.png    decode PNG, verify pixels
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc -lpng

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

#include <png.h>

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) out vec3 vColor;
void main() {
    vec2 p[3] = vec2[](vec2(-0.8, 0.7), vec2(0.8, 0.7),
                       vec2(0.0, -0.8));
    vec3 c[3] = vec3[](vec3(1, 0, 0), vec3(0, 1, 0),
                       vec3(0, 0, 1));
    gl_Position = vec4(p[gl_VertexIndex], 0.0, 1.0);
    vColor = c[gl_VertexIndex];
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(vColor, 1.0); }
)GLSL";

struct SwapchainEx {
    vk::UniqueSwapchainKHR sc;
    std::vector<vk::Image> images;
    std::vector<vk::UniqueImageView> views;
    vk::Format format;
    vk::Extent2D extent;
    bool canReadback = false;
};

/// Same negotiation as vkmini::Swapchain — PLUS the crucial extra:
/// eTransferSrc usage (gated on what the surface supports).
static SwapchainEx createSwapchain(vkmini::Vk& vk,
                                   vk::SurfaceKHR surface,
                                   uint32_t w, uint32_t h) {
    SwapchainEx o;
    auto caps = vk.phys.getSurfaceCapabilitiesKHR(surface);
    o.extent = caps.currentExtent.width != 0xFFFFFFFFu
                   ? caps.currentExtent
                   : vk::Extent2D{w, h};
    // Prefer a plain 8-bit format when offered — the readback code
    // below handles B8G8R8A8/R8G8B8A8 AND packed 10-bit, but a
    // simple format keeps the hot path obvious.
    auto fmts = vk.phys.getSurfaceFormatsKHR(surface);
    o.format = fmts.front().format;
    for (auto& f : fmts)
        if (f.format == vk::Format::eB8G8R8A8Unorm ||
            f.format == vk::Format::eR8G8B8A8Unorm)
            o.format = f.format;
    // THE gate: not every surface allows reading its images back.
    auto wantUsage = vk::ImageUsageFlags{
        vk::ImageUsageFlagBits::eColorAttachment};
    if (caps.supportedUsageFlags &
        vk::ImageUsageFlagBits::eTransferSrc) {
        wantUsage |= vk::ImageUsageFlagBits::eTransferSrc;
        o.canReadback = true;
    } else {
        printf("[demo] surface does NOT support TransferSrc — "
               "readback fallback would be needed\n");
    }
    uint32_t n = caps.minImageCount + 1;
    if (caps.maxImageCount) n = std::min(n, caps.maxImageCount);
    vk::SwapchainCreateInfoKHR si{};
    si.setSurface(surface)
        .setMinImageCount(n)
        .setImageFormat(o.format)
        .setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear)
        .setImageExtent(o.extent)
        .setImageArrayLayers(1)
        .setImageUsage(wantUsage)
        .setImageSharingMode(vk::SharingMode::eExclusive)
        .setPreTransform(caps.currentTransform)
        .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
        .setPresentMode(vk::PresentModeKHR::eFifo)
        .setClipped(VK_TRUE);
    o.sc = vk.device->createSwapchainKHRUnique(si);
    o.images = vk.device->getSwapchainImagesKHR(o.sc.get());
    for (auto img : o.images) {
        vk::ImageViewCreateInfo vi{};
        vi.setImage(img)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(o.format)
            .setSubresourceRange(
                {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        o.views.push_back(vk.device->createImageViewUnique(vi));
    }
    return o;
}

static void imgBarrier(vk::CommandBuffer c, vk::Image img,
                       vk::ImageLayout ol, vk::ImageLayout nl,
                       vk::AccessFlags src, vk::AccessFlags dst,
                       vk::PipelineStageFlags ss,
                       vk::PipelineStageFlags ds) {
    vk::ImageMemoryBarrier b{};
    b.setOldLayout(ol)
        .setNewLayout(nl)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(img)
        .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1,
                              0, 1})
        .setSrcAccessMask(src)
        .setDstAccessMask(dst);
    c.pipelineBarrier(ss, ds, {}, {}, {}, b);
}

int main() {
    vkmini::Vk vk;
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* win =
        glfwCreateWindow(512, 512, "vkmini p20", nullptr, nullptr);
    uint32_t n = 0;
    const char** e = glfwGetRequiredInstanceExtensions(&n);
    vk.createInstance({e, e + n});
    VkSurfaceKHR raw;
    glfwCreateWindowSurface(vk.instance.get(), win, nullptr, &raw);
    auto surface = vk::UniqueSurfaceKHR(raw, vk.instance.get());
    vk.pickPhysicalDevice();
    vk.createDevice({VK_KHR_SWAPCHAIN_EXTENSION_NAME});

    int w = 0, h = 0;
    glfwGetFramebufferSize(win, &w, &h);
    auto sc = createSwapchain(vk, surface.get(), uint32_t(w),
                              uint32_t(h));

    // host-visible readback buffer (BGRA8 swapchain = 4 bpp)
    auto readback = vk.createBuffer(
        vk::DeviceSize(sc.extent.width) * sc.extent.height * 4,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);

    auto vs = vk.shader(kVert, shaderc_vertex_shader, "t.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "t.frag");
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
    auto layout = vk.device->createPipelineLayoutUnique({});
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(sc.format);
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
    auto pipeRes = vk.device->createGraphicsPipelineUnique({}, gi);
    if (pipeRes.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    auto pipe = std::move(pipeRes.value);

    auto cmd = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::ePrimary, 1});
    auto acqSem = vk.device->createSemaphoreUnique({});
    auto doneSem = vk.device->createSemaphoreUnique({});
    auto fence = vk.device->createFenceUnique({});

    constexpr int kShotFrame = 60;
    for (int f = 0; f < 120 && !glfwWindowShouldClose(win); ++f) {
        glfwPollEvents();
        vk.device->resetFences(fence.get());
        auto acq = vk.device->acquireNextImageKHR(
            sc.sc.get(), UINT64_MAX, acqSem.get(), {});
        vk::Image img = sc.images[acq.value];
        cmd[0]->reset();
        cmd[0]->begin(
            {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        imgBarrier(cmd[0].get(), img, vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eColorAttachmentOptimal, {},
                   vk::AccessFlagBits::eColorAttachmentWrite,
                   vk::PipelineStageFlagBits::eTopOfPipe,
                   vk::PipelineStageFlagBits::eColorAttachmentOutput);
        vk::RenderingAttachmentInfo att{};
        att.setImageView(sc.views[acq.value].get())
            .setImageLayout(
                vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setClearValue(vk::ClearValue{vk::ClearColorValue{
                std::array{0.05f, 0.05f, 0.08f, 1.f}}});
        vk::RenderingInfo ri{};
        ri.setRenderArea({{0, 0}, sc.extent})
            .setLayerCount(1)
            .setColorAttachments(att);
        cmd[0]->beginRendering(ri);
        cmd[0]->bindPipeline(vk::PipelineBindPoint::eGraphics,
                             pipe.get());
        vk::Viewport vp{0, 0, float(sc.extent.width),
                        float(sc.extent.height), 0.f, 1.f};
        cmd[0]->setViewport(0, vp);
        vk::Rect2D sci{{0, 0}, sc.extent};
        cmd[0]->setScissor(0, sci);
        cmd[0]->draw(3, 1, 0, 0);
        cmd[0]->endRendering();

        if (f == kShotFrame && sc.canReadback) {
            // THE screenshot: attachment -> copy source -> copy ->
            // back to present layout. All before this frame presents.
            imgBarrier(cmd[0].get(), img,
                       vk::ImageLayout::eColorAttachmentOptimal,
                       vk::ImageLayout::eTransferSrcOptimal,
                       vk::AccessFlagBits::eColorAttachmentWrite,
                       vk::AccessFlagBits::eTransferRead,
                       vk::PipelineStageFlagBits::
                           eColorAttachmentOutput,
                       vk::PipelineStageFlagBits::eTransfer);
            vk::BufferImageCopy r{};
            r.setImageSubresource(
                 {vk::ImageAspectFlagBits::eColor, 0, 0, 1})
                .setImageExtent({sc.extent.width, sc.extent.height,
                                 1});
            cmd[0]->copyImageToBuffer(img,
                                      vk::ImageLayout::
                                          eTransferSrcOptimal,
                                      readback.buf.get(), r);
        }
        imgBarrier(cmd[0].get(), img,
                   // when the shot happened, the image is in
                   // TransferSrc; otherwise still ColorAttachment
                   (f == kShotFrame && sc.canReadback)
                       ? vk::ImageLayout::eTransferSrcOptimal
                       : vk::ImageLayout::eColorAttachmentOptimal,
                   vk::ImageLayout::ePresentSrcKHR,
                   f == kShotFrame
                       ? vk::AccessFlags{
                             vk::AccessFlagBits::eTransferRead}
                       : vk::AccessFlags{vk::AccessFlagBits::
                                             eColorAttachmentWrite},
                   {},
                   f == kShotFrame
                       ? vk::PipelineStageFlags{
                             vk::PipelineStageFlagBits::eTransfer}
                       : vk::PipelineStageFlags{
                             vk::PipelineStageFlagBits::
                                 eColorAttachmentOutput},
                   vk::PipelineStageFlagBits::eBottomOfPipe);
        cmd[0]->end();
        vk::PipelineStageFlags waitStage =
            vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::SubmitInfo si{};
        si.setWaitSemaphores(acqSem.get())
            .setWaitDstStageMask(waitStage)
            .setCommandBuffers(cmd[0].get())
            .setSignalSemaphores(doneSem.get());
        vk.queue.submit(si, fence.get());
        vk::PresentInfoKHR pi{};
        pi.setWaitSemaphores(doneSem.get())
            .setSwapchains(sc.sc.get())
            .setImageIndices(acq.value);
        (void)vk.queue.presentKHR(pi);
        (void)vk.device->waitForFences(fence.get(), VK_TRUE,
                                       UINT64_MAX);
    }
    vk.queue.waitIdle();

    // ---- write the shot as PNG ---------------------------------
    {
        // decode the (BGRA/RGBA/10-bit) readback into plain RGB
        // rows for libpng
        std::vector<uint8_t> rgb(
            size_t(sc.extent.width) * sc.extent.height * 3);
        auto* px = static_cast<const uint8_t*>(readback.mapped);
        for (uint32_t i = 0; i < sc.extent.width * sc.extent.height;
             ++i) {
            const uint8_t* p = px + i * 4;
            if (sc.format == vk::Format::eB8G8R8A8Unorm ||
                sc.format == vk::Format::eB8G8R8A8Srgb) {
                rgb[i * 3 + 0] = p[2];
                rgb[i * 3 + 1] = p[1];
                rgb[i * 3 + 2] = p[0];
            } else if (sc.format == vk::Format::eR8G8B8A8Unorm) {
                memcpy(&rgb[i * 3], p, 3);
            } else { // packed 10-bit
                uint32_t v;
                memcpy(&v, p, 4);
                bool a2b = sc.format ==
                           vk::Format::eA2B10G10R10UnormPack32;
                rgb[i * 3 + 0] = uint8_t(
                    (a2b ? v : v >> 20) & 0x3FF) >> 2;
                rgb[i * 3 + 1] =
                    uint8_t((v >> 10) & 0x3FF) >> 2;
                rgb[i * 3 + 2] = uint8_t(
                    (a2b ? v >> 20 : v) & 0x3FF) >> 2;
            }
        }

        FILE* fp = fopen("shot.png", "wb");
        png_struct* png =
            png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr,
                                    nullptr, nullptr);
        png_info* info = png_create_info_struct(png);
        png_init_io(png, fp);
        png_set_IHDR(png, info, sc.extent.width, sc.extent.height,
                     8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT,
                     PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png, info);
        for (uint32_t y = 0; y < sc.extent.height; ++y)
            png_write_row(png, rgb.data() + y * sc.extent.width * 3);
        png_write_end(png, info);
        png_destroy_write_struct(&png, &info);
        fclose(fp);
    }
    printf("wrote shot.png (%ux%u, format %s)\n", sc.extent.width,
           sc.extent.height, vk::to_string(sc.format).c_str());
    sc.views.clear();
    sc.sc.reset();
    surface.reset();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
