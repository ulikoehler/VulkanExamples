// SPDX-License-Identifier: CC0-1.0
//
// Post 15: mipmaps with vkCmdBlitImage. Level 0 is uploaded from a
// staging buffer; every other level is generated ON THE GPU by
// blitting (linear-filtered downsample) from the previous level,
// each with its own layout transitions. A sampler with a real
// mipLodRange then picks levels — two quads sample the same texture
// at forced LOD 4 and LOD 8 to prove the chain contains correct
// downsampled data.
//
//   ./app                 windowed
//   ./app --headless o.ppm     one frame -> PPM for check.py
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) out vec2 vUV;
layout(push_constant) uniform Push { vec2 ofs; float lod; } pc;
void main() {
    vec2 p[6] = vec2[](vec2(-0.5,-0.5), vec2(0.5,-0.5), vec2(0.5,0.5),
                       vec2(-0.5,-0.5), vec2(0.5,0.5),  vec2(-0.5,0.5));
    vec2 q = p[gl_VertexIndex];
    gl_Position = vec4(q * 0.9 + pc.ofs, 0.0, 1.0);
    vUV = q + 0.5;
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec2 ofs; float lod; } pc;
void main() {
    // textureLod forces the exact mip level — that makes the test
    // independent of screen-space derivatives.
    outColor = textureLod(tex, vUV, pc.lod);
}
)GLSL";

struct Push {
    float ofs[2];
    float lod;
    float _pad;
};
static_assert(sizeof(Push) == 16);

/// Texture with a full mip chain — 256x256 -> 9 levels. The view
/// must span ALL levels (that's vkmini's default), the sampler must
/// allow the range, and each level needs TRANSFER_SRC/DST layouts
/// during generation.
static vkmini::Vk::Image makeTexture(vkmini::Vk& vk, uint32_t w,
                                     uint32_t h) {
    vkmini::Vk::Image o;
    uint32_t levels = 0;
    for (uint32_t s = w; s; s >>= 1) ++levels;
    vk::ImageCreateInfo ii{};
    ii.setImageType(vk::ImageType::e2D)
        .setFormat(vk::Format::eR8G8B8A8Unorm)
        .setExtent({w, h, 1})
        .setMipLevels(levels)                  // <- the chain
        .setArrayLayers(1)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage(vk::ImageUsageFlagBits::eSampled |
                  vk::ImageUsageFlagBits::eTransferDst |
                  vk::ImageUsageFlagBits::eTransferSrc)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined);
    o.img = vk.device->createImageUnique(ii);
    auto req = vk.device->getImageMemoryRequirements(o.img.get());
    o.mem = vk.device->allocateMemoryUnique(
        {req.size, vk.memoryType(
             req.memoryTypeBits,
             vk::MemoryPropertyFlagBits::eDeviceLocal)});
    vk.device->bindImageMemory(o.img.get(), o.mem.get(), 0);
    vk::ImageViewCreateInfo vi{};
    vi.setImage(o.img.get())
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(vk::Format::eR8G8B8A8Unorm)
        .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0,
                              levels, 0, 1}); // ALL levels
    o.view = vk.device->createImageViewUnique(vi);
    return o;
}

/// Upload level 0 + generate levels 1..N with blits.
static void uploadAndGenerateMips(vkmini::Vk& vk,
                                  const vkmini::Vk::Image& tex,
                                  uint32_t w, uint32_t h,
                                  const void* pixels) {
    auto staging = vk.createBuffer(
        vk::DeviceSize(w) * h * 4, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(staging.mapped, pixels, vk::DeviceSize(w) * h * 4);

    vk.oneTime([&](vk::CommandBuffer c) {
        // level 0: undefined -> transfer dst, copy in
        auto barrier = [&](uint32_t base, vk::ImageLayout ol,
                           vk::ImageLayout nl, vk::AccessFlags src,
                           vk::AccessFlags dst,
                           vk::PipelineStageFlags ss,
                           vk::PipelineStageFlags ds) {
            vk::ImageMemoryBarrier b{};
            b.setOldLayout(ol)
                .setNewLayout(nl)
                .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setImage(tex.img.get())
                .setSubresourceRange(
                    {vk::ImageAspectFlagBits::eColor, base, 1, 0, 1})
                .setSrcAccessMask(src)
                .setDstAccessMask(dst);
            c.pipelineBarrier(ss, ds, {}, {}, {}, b);
        };
        barrier(0, vk::ImageLayout::eUndefined,
                vk::ImageLayout::eTransferDstOptimal, {},
                vk::AccessFlagBits::eTransferWrite,
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eTransfer);
        vk::BufferImageCopy r{};
        r.setImageSubresource(
            {vk::ImageAspectFlagBits::eColor, 0, 0, 1});
        r.setImageExtent({w, h, 1});
        c.copyBufferToImage(staging.buf.get(), tex.img.get(),
                            vk::ImageLayout::eTransferDstOptimal, r);
        // level 0 is now the blit SOURCE
        barrier(0, vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eTransferSrcOptimal,
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eTransferRead,
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eTransfer);

        // generate each level from the previous one
        uint32_t levels = 0;
        for (uint32_t s = w; s; s >>= 1) ++levels;
        int32_t mw = int32_t(w), mh = int32_t(h);
        for (uint32_t l = 1; l < levels; ++l) {
            barrier(l, vk::ImageLayout::eUndefined,
                    vk::ImageLayout::eTransferDstOptimal, {},
                    vk::AccessFlagBits::eTransferWrite,
                    vk::PipelineStageFlagBits::eTopOfPipe,
                    vk::PipelineStageFlagBits::eTransfer);
            vk::ImageBlit bl{};
            bl.setSrcSubresource(
                {vk::ImageAspectFlagBits::eColor, l - 1, 0, 1});
            bl.setSrcOffsets({vk::Offset3D{0, 0, 0},
                              vk::Offset3D{mw, mh, 1}});
            int32_t nw = std::max(1, mw / 2), nh = std::max(1, mh / 2);
            bl.setDstSubresource(
                {vk::ImageAspectFlagBits::eColor, l, 0, 1});
            bl.setDstOffsets({vk::Offset3D{0, 0, 0},
                              vk::Offset3D{nw, nh, 1}});
            c.blitImage(tex.img.get(),
                        vk::ImageLayout::eTransferSrcOptimal,
                        tex.img.get(),
                        vk::ImageLayout::eTransferDstOptimal, bl,
                        vk::Filter::eLinear);
            // this level becomes the next source + final read layout
            barrier(l, vk::ImageLayout::eTransferDstOptimal,
                    vk::ImageLayout::eTransferSrcOptimal,
                    vk::AccessFlagBits::eTransferWrite,
                    vk::AccessFlagBits::eTransferRead,
                    vk::PipelineStageFlagBits::eTransfer,
                    vk::PipelineStageFlagBits::eTransfer);
            mw = nw;
            mh = nh;
        }
        // whole chain -> shader read
        barrier(0, vk::ImageLayout::eTransferSrcOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::AccessFlagBits::eTransferRead,
                vk::AccessFlagBits::eShaderRead,
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eFragmentShader);
        // levels 1..N are still TransferSrc — fix them in one barrier
        vk::ImageMemoryBarrier b{};
        b.setOldLayout(vk::ImageLayout::eTransferSrcOptimal)
            .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(tex.img.get())
            .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 1,
                                  levels - 1, 0, 1})
            .setSrcAccessMask(vk::AccessFlagBits::eTransferRead)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                          vk::PipelineStageFlagBits::eFragmentShader,
                          {}, {}, {}, b);
    });
}

struct Pipe {
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
    vk::UniqueDescriptorSetLayout dsLayout;
    vk::UniqueDescriptorPool pool;
    vk::UniqueDescriptorSet ds;
    vk::UniqueSampler sampler;
};

static Pipe makePipeline(vkmini::Vk& vk, vk::Format fmt,
                         vk::ImageView view) {
    Pipe o;
    vk::DescriptorSetLayoutBinding bnd{};
    bnd.setBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eFragment);
    o.dsLayout = vk.device->createDescriptorSetLayoutUnique(
        {{}, bnd});
    vk::PushConstantRange pcr{};
    pcr.setStageFlags(vk::ShaderStageFlagBits::eVertex |
                      vk::ShaderStageFlagBits::eFragment)
        .setOffset(0)
        .setSize(sizeof(Push));
    o.layout = vk.device->createPipelineLayoutUnique(
        {{}, o.dsLayout.get(), pcr});

    auto vs = vk.shader(kVert, shaderc_vertex_shader, "m.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "m.frag");
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
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(fmt);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(stages)
        .setPVertexInputState(&vin)
        .setPInputAssemblyState(&ia)
        .setPViewportState(&vp)
        .setPDynamicState(&dyn)
        .setPRasterizationState(&rs)
        .setPMultisampleState(&ms)
        .setPColorBlendState(&blend)
        .setLayout(o.layout.get())
        .setPNext(&rendering);
    auto res = vk.device->createGraphicsPipelineUnique({}, gi);
    if (res.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    o.pipeline = std::move(res.value);

    // sampler + descriptor — the mipmap-capable sampler is the other
    // half of the story: mipmapMode eLinear, lod range 0..1000.
    vk::SamplerCreateInfo sci{};
    sci.setMagFilter(vk::Filter::eLinear)
        .setMinFilter(vk::Filter::eLinear)
        .setMipmapMode(vk::SamplerMipmapMode::eLinear)
        .setMinLod(0.f)
        .setMaxLod(1000.f); // allow the whole chain
    o.sampler = vk.device->createSamplerUnique(sci);
    vk::DescriptorPoolSize ps{vk::DescriptorType::eCombinedImageSampler,
                              1};
    o.pool = vk.device->createDescriptorPoolUnique(
        {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1,
         ps});
    auto sets = vk.device->allocateDescriptorSetsUnique(
        {o.pool.get(), o.dsLayout.get()});
    o.ds = std::move(sets[0]);
    vk::DescriptorImageInfo di{};
    di.setSampler(o.sampler.get())
        .setImageView(view)
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    vk::WriteDescriptorSet wr{};
    wr.setDstSet(o.ds.get())
        .setDstBinding(0)
        .setDescriptorType(
            vk::DescriptorType::eCombinedImageSampler)
        .setImageInfo(di);
    vk.device->updateDescriptorSets(wr, {});
    return o;
}

static void record(vk::CommandBuffer c, vk::ImageView view,
                   vk::Extent2D extent, const Pipe& p) {
    vk::RenderingAttachmentInfo att{};
    att.setImageView(view)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearValue{
            vk::ClearColorValue{std::array{0.05f, 0.05f, 0.08f,
                                           1.f}}});
    vk::RenderingInfo ri{};
    ri.setRenderArea({{0, 0}, extent})
        .setLayerCount(1)
        .setColorAttachments(att);
    c.beginRendering(ri);
    c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                   p.pipeline.get());
    c.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                         p.layout.get(), 0, p.ds.get(), {});
    vk::Viewport vp{0, 0, float(extent.width), float(extent.height),
                    0.f, 1.f};
    c.setViewport(0, vp);
    vk::Rect2D sc{{0, 0}, extent};
    c.setScissor(0, sc);
    // LEFT quad: forced LOD 4 (16x16 mip — quadrants still distinct)
    Push lo{{-0.48f, 0.f}, 4.f, 0.f};
    c.pushConstants(p.layout.get(),
                    vk::ShaderStageFlagBits::eVertex |
                        vk::ShaderStageFlagBits::eFragment,
                    0, sizeof(Push), &lo);
    c.draw(6, 1, 0, 0);
    // RIGHT quad: forced LOD 8 (1x1 mip — one average color)
    Push hi{{0.48f, 0.f}, 8.f, 0.f};
    c.pushConstants(p.layout.get(),
                    vk::ShaderStageFlagBits::eVertex |
                        vk::ShaderStageFlagBits::eFragment,
                    0, sizeof(Push), &hi);
    c.draw(6, 1, 0, 0);
    c.endRendering();
}

static void imgBarrier(vk::CommandBuffer c, vk::Image img,
                       vk::ImageAspectFlags aspect,
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
        .setSubresourceRange({aspect, 0, 1, 0, 1})
        .setSrcAccessMask(src)
        .setDstAccessMask(dst);
    c.pipelineBarrier(ss, ds, {}, {}, {}, b);
}

int main(int argc, char** argv) {
    bool headless = argc > 1 && std::string(argv[1]) == "--headless";
    const char* ppmOut = argc > 2 ? argv[2] : "out.ppm";

    vkmini::Vk vk;
    GLFWwindow* win = nullptr;
    vk::UniqueSurfaceKHR surface;
    if (!headless) {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        win = glfwCreateWindow(512, 256, "vkmini p15", nullptr, nullptr);
    }
    std::vector<const char*> instExts;
    if (!headless) {
        uint32_t n = 0;
        const char** e = glfwGetRequiredInstanceExtensions(&n);
        instExts.assign(e, e + n);
    }
    vk.createInstance(instExts);
    if (!headless) {
        VkSurfaceKHR raw;
        glfwCreateWindowSurface(vk.instance.get(), win, nullptr, &raw);
        surface = vk::UniqueSurfaceKHR(raw, vk.instance.get());
    }
    vk.pickPhysicalDevice();
    vk.createDevice(headless ? std::vector<const char*>{}
                             : std::vector<const char*>{
                                   VK_KHR_SWAPCHAIN_EXTENSION_NAME});

    // level-0 texture: four quadrant colors
    constexpr uint32_t TW = 256, TH = 256;
    std::vector<uint8_t> px(TW * TH * 4);
    for (uint32_t y = 0; y < TH; ++y)
        for (uint32_t x = 0; x < TW; ++x) {
            uint8_t* p = &px[(y * TW + x) * 4];
            bool left = x < TW / 2, top = y < TH / 2;
            p[0] = left ? 255 : 0;             // r
            p[1] = top ? 0 : 255;              // g
            p[2] = !left && !top ? 255 : 0;    // b
            p[3] = 255;
        }
    auto tex = makeTexture(vk, TW, TH);
    uploadAndGenerateMips(vk, tex, TW, TH, px.data());

    if (headless) {
        vkmini::Headless hl;
        hl.init(vk, 512, 256);
        Pipe p = makePipeline(vk, vk::Format::eR8G8B8A8Unorm,
                              tex.view.get());
        hl.render(vk::ImageLayout::eColorAttachmentOptimal,
                  [&](vk::CommandBuffer c) {
                      record(c, hl.color.view.get(), hl.extent, p);
                  });
        hl.savePpm(ppmOut);
        printf("wrote %s\n", ppmOut);
        return 0;
    }

    vkmini::Swapchain sc;
    sc.create(vk, vk.phys, surface.get(), 512, 256);
    Pipe p = makePipeline(vk, sc.format, tex.view.get());
    auto cmd = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::ePrimary, 1});
    auto acqSem = vk.device->createSemaphoreUnique({});
    auto doneSem = vk.device->createSemaphoreUnique({});
    auto fence = vk.device->createFenceUnique({});

    for (int f = 0; f < 240 && !glfwWindowShouldClose(win); ++f) {
        glfwPollEvents();
        vk.device->resetFences(fence.get());
        auto acq = vk.device->acquireNextImageKHR(
            sc.sc.get(), UINT64_MAX, acqSem.get(), {});
        vk::Image img = sc.images[acq.value];
        cmd[0]->reset();
        cmd[0]->begin(
            {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        imgBarrier(cmd[0].get(), img, vk::ImageAspectFlagBits::eColor,
                   vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eColorAttachmentOptimal,
                   vk::AccessFlags{},
                   vk::AccessFlagBits::eColorAttachmentWrite,
                   vk::PipelineStageFlagBits::eTopOfPipe,
                   vk::PipelineStageFlagBits::eColorAttachmentOutput);
        record(cmd[0].get(), sc.views[acq.value].get(), sc.extent, p);
        imgBarrier(cmd[0].get(), img, vk::ImageAspectFlagBits::eColor,
                   vk::ImageLayout::eColorAttachmentOptimal,
                   vk::ImageLayout::ePresentSrcKHR,
                   vk::AccessFlagBits::eColorAttachmentWrite,
                   vk::AccessFlags{},
                   vk::PipelineStageFlagBits::eColorAttachmentOutput,
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
    sc.views.clear();
    sc.sc.reset();
    surface.reset();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
