// SPDX-License-Identifier: CC0-1.0
//
// HDR output in Vulkan: the swapchain's (format, colorSpace) pair
// decides what reaches the display. VK_EXT_swapchain_colorspace
// (instance ext) lets you ask for HDR spaces:
//   A2B10G10R10UnormPack32 + VK_COLOR_SPACE_HDR10_ST2084_EXT  = HDR10
//   R16G16B16A16Sfloat      + VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT
//                                                           = scRGB
// No HDR display -> the pair won't be offered -> fall back to SDR.
//
// Demo: enumerate surface format/colorspace pairs, prefer HDR10,
// fall back to BGRA8+sRGB, render one gradient frame, log the pick.
//
//   ./app --windowed    (xvfb-run -a ./app --windowed works headless)
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

// gradient ramping past SDR 1.0 — on a real HDR10 swapchain these
// values map to nit levels; on SDR they just clamp to white.
static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() {
    float v = vUV.x * 4.0;   // 0..4, beyond SDR range
    outColor = vec4(v, v * 0.5, v * 0.25, 1.0);
}
)GLSL";

static const char* fmtName(vk::Format f) {
    switch (f) {
    case vk::Format::eB8G8R8A8Unorm: return "B8G8R8A8Unorm";
    case vk::Format::eR8G8B8A8Unorm: return "R8G8B8A8Unorm";
    case vk::Format::eA2B10G10R10UnormPack32:
        return "A2B10G10R10UnormPack32";
    case vk::Format::eA2R10G10B10UnormPack32:
        return "A2R10G10B10UnormPack32";
    case vk::Format::eR16G16B16A16Sfloat:
        return "R16G16B16A16Sfloat";
    default: return "other";
    }
}
static const char* csName(vk::ColorSpaceKHR c) {
    switch (c) {
    case vk::ColorSpaceKHR::eSrgbNonlinear: return "SRGB_NONLINEAR";
    case vk::ColorSpaceKHR::eHdr10St2084EXT: return "HDR10_ST2084";
    case vk::ColorSpaceKHR::eHdr10HlgEXT: return "HDR10_HLG";
    case vk::ColorSpaceKHR::eExtendedSrgbLinearEXT:
        return "EXTENDED_SRGB_LINEAR(scRGB)";
    case vk::ColorSpaceKHR::eDisplayP3NonlinearEXT:
        return "DISPLAY_P3";
    case vk::ColorSpaceKHR::eBt2020LinearEXT: return "BT2020_LINEAR";
    default: return "other";
    }
}

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    bool hold = false;
    for (int i = 1; i < argc; ++i)
        hold |= !strcmp(argv[i], "--hold");
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    auto* win = glfwCreateWindow(384, 384, "hdr", nullptr, nullptr);
    vkmini::Vk vk;
    uint32_t n = 0;
    auto req = glfwGetRequiredInstanceExtensions(&n);
    std::vector<const char*> ie(req, req + n);
    ie.push_back("VK_EXT_swapchain_colorspace");  // unlocks non-sRGB
    vk.createInstance(ie);
    VkSurfaceKHR sraw;
    glfwCreateWindowSurface(vk.instance.get(), win, nullptr, &sraw);
    vk::UniqueSurfaceKHR surface(sraw, vk.instance.get());
    vk.pickPhysicalDevice();
    vk.createDevice({"VK_KHR_swapchain"});

    // enumerate every (format, colorSpace) the surface offers
    auto fmts = vk.phys.getSurfaceFormatsKHR(surface.get());
    for (auto& f : fmts)
        printf("offered: %-24s %s\n", fmtName(f.format),
               csName(f.colorSpace));

    // preference list: HDR10 first, then scRGB, then SDR
    struct Want { vk::Format f; vk::ColorSpaceKHR c; const char* tag; };
    const Want want[] = {
        {vk::Format::eA2B10G10R10UnormPack32,
         vk::ColorSpaceKHR::eHdr10St2084EXT, "HDR10"},
        {vk::Format::eR16G16B16A16Sfloat,
         vk::ColorSpaceKHR::eExtendedSrgbLinearEXT, "scRGB"},
        {vk::Format::eB8G8R8A8Unorm,
         vk::ColorSpaceKHR::eSrgbNonlinear, "SDR"},
    };
    vk::SurfaceFormatKHR pick{};
    const char* pickedTag = "SDR";
    for (auto& w : want) {
        for (auto& f : fmts)
            if (f.format == w.f && f.colorSpace == w.c) {
                pick = f;
                pickedTag = w.tag;
                goto done;
            }
    }
    pick = fmts.front();   // last resort: whatever's there
    pickedTag = "fallback";
done:
    printf("picked: %s (%s + %s)\n", pickedTag,
           fmtName(pick.format), csName(pick.colorSpace));

    // swapchain with the picked pair (manual, not vkmini::Swapchain
    // — the point is the format/colorSpace choice)
    auto caps = vk.phys.getSurfaceCapabilitiesKHR(surface.get());
    vk::Extent2D extent = caps.currentExtent;
    if (extent.width == 0xFFFFFFFFu) {  // "surface doesn't care"
        extent.width = std::clamp(384u, caps.minImageExtent.width,
                                  caps.maxImageExtent.width);
        extent.height = std::clamp(384u, caps.minImageExtent.height,
                                   caps.maxImageExtent.height);
    }
    printf("extent %ux%u (caps current %ux%u)\n", extent.width,
           extent.height, caps.currentExtent.width,
           caps.currentExtent.height);
    vk::SwapchainCreateInfoKHR si{};
    uint32_t nImg = caps.minImageCount + 1;
    if (caps.maxImageCount) nImg = std::min(nImg, caps.maxImageCount);
    si.setSurface(surface.get())
        .setMinImageCount(nImg)
        .setImageFormat(pick.format)
        .setImageColorSpace(pick.colorSpace)
        .setImageExtent(extent)
        .setImageArrayLayers(1)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
        .setPreTransform(caps.currentTransform)
        .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
        .setPresentMode(vk::PresentModeKHR::eFifo)
        .setClipped(VK_TRUE);
    // A listed pair is no guarantee creation succeeds — compositors
    // can advertise spaces they can't actually present. Retry with
    // plain SDR if the HDR pick throws.
    vk::UniqueSwapchainKHR sc;
    try {
        sc = vk.device->createSwapchainKHRUnique(si);
    } catch (const vk::SystemError& e) {
        printf("HDR swapchain create failed (%s) -> SDR retry\n",
               e.what());
        pick = {vk::Format::eB8G8R8A8Unorm,
                vk::ColorSpaceKHR::eSrgbNonlinear};
        pickedTag = "SDR";
        si.setImageFormat(pick.format)
            .setImageColorSpace(pick.colorSpace);
        sc = vk.device->createSwapchainKHRUnique(si);
    }
    printf("using: %s (%s + %s)\n", pickedTag, fmtName(pick.format),
           csName(pick.colorSpace));
    auto images = vk.device->getSwapchainImagesKHR(sc.get());
    std::vector<vk::UniqueImageView> views;
    for (auto im : images) {
        vk::ImageViewCreateInfo vi{};
        vi.setImage(im)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(pick.format)
            .setSubresourceRange({vk::ImageAspectFlagBits::eColor,
                                  0, 1, 0, 1});
        views.push_back(vk.device->createImageViewUnique(vi));
    }

    // simple pipeline, render one frame, present
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
    auto layout = vk.device->createPipelineLayoutUnique(plci);
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(pick.format);
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

    auto sem = vk.device->createSemaphoreUnique({});
    auto done = vk.device->createSemaphoreUnique({});
    auto idx = vk.device->acquireNextImageKHR(sc.get(), UINT64_MAX,
                                              sem.get(), nullptr);
    auto cmds = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::ePrimary, 1});
    auto& cmd = *cmds[0];
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::ImageSubresourceRange sr{vk::ImageAspectFlagBits::eColor,
                                 0, 1, 0, 1};
    vk::ImageMemoryBarrier b{};
    b.setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(images[idx.value])
        .setSubresourceRange(sr)
        .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                        vk::PipelineStageFlagBits::eColorAttachmentOutput,
                        {}, {}, {}, b);
    vk::RenderingAttachmentInfo att{};
    att.setImageView(views[idx.value].get())
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearValue{vk::ClearColorValue{
            std::array{0.f, 0.f, 0.f, 1.f}}});
    vk::RenderingInfo ri{};
    ri.setRenderArea({{0, 0}, extent})
        .setLayerCount(1)
        .setColorAttachments(att);
    cmd.beginRendering(ri);
    vk::Viewport v{0, 0, (float)extent.width,
                   (float)extent.height, 0.f, 1.f};
    cmd.setViewport(0, v);
    vk::Rect2D sc2{{0, 0}, extent};
    cmd.setScissor(0, sc2);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe.get());
    cmd.draw(3, 1, 0, 0);
    cmd.endRendering();
    vk::ImageMemoryBarrier b2{};
    b2.setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(images[idx.value])
        .setSubresourceRange(sr)
        .setSrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                        vk::PipelineStageFlagBits::eBottomOfPipe,
                        {}, {}, {}, b2);
    cmd.end();
    vk::SubmitInfo sub{};
    vk::PipelineStageFlags ws =
        vk::PipelineStageFlagBits::eColorAttachmentOutput;
    sub.setWaitSemaphores(sem.get())
        .setWaitDstStageMask(ws)
        .setCommandBuffers(cmd)
        .setSignalSemaphores(done.get());
    vk.queue.submit(sub);
    vk::PresentInfoKHR pi{};
    pi.setWaitSemaphores(done.get())
        .setSwapchains(sc.get())
        .setImageIndices(idx.value);
    (void)vk.queue.presentKHR(pi);
    vk.device->waitIdle();
    printf("presented on %s swapchain\n", pickedTag);
    if (hold) sleep(4);  // keep the window up for viewing/screenshots
    return 0;
}
