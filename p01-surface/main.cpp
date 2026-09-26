// SPDX-License-Identifier: CC0-1.0
//
// Post 1: create a Vulkan surface via GLFW, a swapchain on it, and
// show it works by clearing every frame to a solid color.
//
//   ./app              windowed: clears the window, alternating color
//   ./app --headless o.ppm   offscreen: same clear, dumped to a PPM
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

int main(int argc, char** argv) {
    bool headless = argc > 1 && std::string(argv[1]) == "--headless";
    const char* ppmOut = argc > 2 ? argv[2] : "out.ppm";

    // ---- 1. GLFW window -------------------------------------------------
    // GLFW_NO_API: GLFW must not create an OpenGL context — the window
    // is just a platform handle we hand to Vulkan.
    if (!headless) {
        if (!glfwInit()) throw std::runtime_error("glfwInit failed");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    }

    // ---- 2. Instance -----------------------------------------------------
    // The surface extension (VK_KHR_surface + a platform WSI extension
    // like VK_KHR_xcb_surface) is dictated by the window system — GLFW
    // tells us exactly which instance extensions it needs.
    vkmini::Vk vk;
    std::vector<const char*> instExts;
    if (!headless) {
        uint32_t n = 0;
        const char** glfwExt = glfwGetRequiredInstanceExtensions(&n);
        instExts.assign(glfwExt, glfwExt + n);
    }
    vk.createInstance(instExts);

    // ---- 3. The surface itself -------------------------------------------
    // glfwCreateWindowSurface picks the right VK_KHR_*_surface
    // extension under the hood and returns a VkSurfaceKHR — an opaque
    // handle for "the drawable area of this window". Nothing is
    // rendered into a surface directly; it exists only so a *swapchain*
    // can be bound to it.
    GLFWwindow* win = nullptr;
    vk::UniqueSurfaceKHR surface;
    if (!headless) {
        win = glfwCreateWindow(640, 480, "vkmini p01", nullptr, nullptr);
        VkSurfaceKHR raw;
        if (glfwCreateWindowSurface(vk.instance.get(), win, nullptr,
                                    &raw) != VK_SUCCESS)
            throw std::runtime_error("surface creation failed");
        surface = vk::UniqueSurfaceKHR(raw, vk.instance.get());
    }

    // ---- 4. Physical + logical device ------------------------------------
    vk.pickPhysicalDevice();
    // A device that *presents* needs the swapchain extension.
    vk.createDevice(headless
                        ? std::vector<const char*>{}
                        : std::vector<const char*>{VK_KHR_SWAPCHAIN_EXTENSION_NAME});

    // ---- 5. Present-support check ----------------------------------------
    // Not every queue family can present to every surface. Ask Vulkan.
    if (!headless) {
        vk::Bool32 canPresent =
            vk.phys.getSurfaceSupportKHR(vk.qfam, surface.get());
        if (!canPresent)
            throw std::runtime_error("queue family cannot present");
    }

    // ---- 6. Swapchain -----------------------------------------------------
    // A swapchain owns a ring of images the compositor flips through.
    // We ask for double-buffered, FIFO (vsync), BGRA8.
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> scImages;
    vk::Format scFormat = vk::Format::eB8G8R8A8Unorm;
    vk::Extent2D scExtent{640, 480};
    if (!headless) {
        auto caps = vk.phys.getSurfaceCapabilitiesKHR(surface.get());
        // currentExtent == 0xFFFFFFFF means "the surface does not
        // dictate a size" — pick the GLFW framebuffer size, clamped
        // into the surface's allowed extent range. Using the sentinel
        // verbatim asks for a ~4G x ~4G swapchain (instant
        // OUT_OF_HOST_MEMORY — a classic gotcha).
        if (caps.currentExtent.width == 0xFFFFFFFFu) {
            int fw = 0, fh = 0;
            glfwGetFramebufferSize(win, &fw, &fh);
            scExtent.width = std::clamp<uint32_t>(
                fw, caps.minImageExtent.width,
                caps.maxImageExtent.width);
            scExtent.height = std::clamp<uint32_t>(
                fh, caps.minImageExtent.height,
                caps.maxImageExtent.height);
        } else {
            scExtent = caps.currentExtent;
        }
        // pick a surface-supported format instead of hoping for BGRA
        auto fmts = vk.phys.getSurfaceFormatsKHR(surface.get());
        scFormat = fmts.front().format;
        uint32_t nImages = caps.minImageCount + 1;
        if (caps.maxImageCount) // 0 means "no limit"
            nImages = std::min(nImages, caps.maxImageCount);
        vk::SwapchainCreateInfoKHR si{};
        si.setSurface(surface.get())
            .setMinImageCount(nImages)
            .setImageFormat(scFormat)
            .setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear)
            .setImageExtent(scExtent)
            .setImageArrayLayers(1)
            .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment |
                           vk::ImageUsageFlagBits::eTransferDst)
            .setImageSharingMode(vk::SharingMode::eExclusive)
            .setPreTransform(caps.currentTransform)
            .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
            .setPresentMode(vk::PresentModeKHR::eFifo)
            .setClipped(VK_TRUE);
        swapchain = vk.device->createSwapchainKHRUnique(si);
        scImages = vk.device->getSwapchainImagesKHR(swapchain.get());
    }

    // ---- 7. Render --------------------------------------------------------
    // No pipelines yet: vkCmdClearColorImage fills an image directly.
    // It needs eTransferDst usage (set on the swapchain above) and is
    // the cheapest way to prove the whole chain works.
    const vk::ClearColorValue colors[] = {
        std::array{0.2f, 0.4f, 0.8f, 1.0f}, // blue
        std::array{0.8f, 0.3f, 0.2f, 1.0f}, // red
    };

    if (headless) {
        vkmini::Headless hl;
        hl.init(vk, 640, 480);
        // headless.render() brackets our commands with the layout
        // transitions the clear needs.
        hl.render(vk::ImageLayout::eTransferDstOptimal,
                  [&](vk::CommandBuffer c) {
                      vk::ImageSubresourceRange sr{
                          vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
                      c.clearColorImage(hl.color.img.get(),
                                        vk::ImageLayout::eTransferDstOptimal,
                                        colors[0], sr);
                  });
        hl.savePpm(ppmOut);
        printf("wrote %s\n", ppmOut);
        return 0;
    }

    // Windowed: acquire → clear → present, ~2s of alternating color.
    auto cmd = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::ePrimary, 1});
    auto acqSem = vk.device->createSemaphoreUnique({});
    auto doneSem = vk.device->createSemaphoreUnique({});
    auto fence = vk.device->createFenceUnique({});

    for (int frame = 0; frame < 120 && !glfwWindowShouldClose(win);
         ++frame) {
        glfwPollEvents();
        vk.device->resetFences(fence.get());
        auto acq = vk.device->acquireNextImageKHR(
            swapchain.get(), UINT64_MAX, acqSem.get(), {});
        vk::Image img = scImages[acq.value];

        cmd[0]->reset();
        cmd[0]->begin(
            {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        // swapchain images arrive in UNDEFINED-ish state: transition
        // to TRANSFER_DST, clear, transition to PRESENT.
        vk::ImageSubresourceRange sr{
            vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        vk::ImageMemoryBarrier toDst{};
        toDst.setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(img)
            .setSubresourceRange(sr)
            .setSrcAccessMask({})
            .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
        cmd[0]->pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, toDst);
        cmd[0]->clearColorImage(
            img, vk::ImageLayout::eTransferDstOptimal,
            colors[frame & 1], sr);
        vk::ImageMemoryBarrier toPresent{};
        toPresent.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
            .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(img)
            .setSubresourceRange(sr)
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask({});
        cmd[0]->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eBottomOfPipe,
                                {}, {}, {}, toPresent);
        cmd[0]->end();

        vk::PipelineStageFlags waitStage =
            vk::PipelineStageFlagBits::eTransfer;
        vk::SubmitInfo si{};
        si.setWaitSemaphores(acqSem.get())
            .setWaitDstStageMask(waitStage)
            .setCommandBuffers(cmd[0].get())
            .setSignalSemaphores(doneSem.get());
        vk.queue.submit(si, fence.get());
        vk::PresentInfoKHR pi{};
        pi.setWaitSemaphores(doneSem.get())
            .setSwapchains(swapchain.get())
            .setImageIndices(acq.value);
        (void)vk.queue.presentKHR(pi);
        (void)vk.device->waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
    }
    // Destroy order matters: swapchain and surface reference the
    // platform window — free them before glfwDestroyWindow, or the
    // driver's WSI teardown touches a dead wl_surface/xcb window.
    vk.queue.waitIdle();
    swapchain.reset();
    surface.reset();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
