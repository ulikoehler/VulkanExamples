// SPDX-License-Identifier: CC0-1.0
//
// A 3x3 video wall in ~300 lines: nine source "streams" live as
// layers of ONE VkImage array; a single pipeline draws nine quads,
// each parameterized by push constants (dst rect, src crop, layer,
// fade, filter). Click a tile to zoom it fullscreen with a cubic
// ease-out; click again (or ESC) to return.
//
//   ./app --headless o.ppm        overview 3x3 grid -> PPM
//   ./app --headless o.ppm --zoom 5   tile 5 zoomed -> PPM
//   ./app                         windowed, click to zoom
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc
//
// Tile sources here are procedural patterns standing in for the
// zero-copy imported frames of posts 48/49 — the draw path is what
// a camera wall actually renders.

#include "vkmini.hpp"
#include <GLFW/glfw3.h>
#include <chrono>
#include <cmath>

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) out vec2 vUV;
layout(location = 1) flat out float vLayer;
layout(push_constant) uniform PC {
    vec4 rect;  // x,y,w,h in 0..1 window space
    vec4 uv;    // src crop x,y,w,h
    vec4 misc;  // layer, fade, saturation, hueShift(rad)
} pc;
void main() {
    // two triangles per tile, indexed 0..5
    const vec2 QUAD[6] = vec2[](vec2(0,0), vec2(1,0), vec2(0,1),
                                vec2(0,1), vec2(1,0), vec2(1,1));
    vec2 q = QUAD[gl_VertexIndex];
    vec2 p = pc.rect.xy + q * pc.rect.zw;      // 0..1
    gl_Position = vec4(p * 2.0 - 1.0, 0, 1);   // -> NDC
    gl_Position.y = -gl_Position.y;            // UV origin top-left
    vUV = pc.uv.xy + q * pc.uv.zw;
    vLayer = pc.misc.x;
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(binding = 0) uniform sampler2DArray tex;
layout(location = 0) in vec2 vUV;
layout(location = 1) flat in float vLayer;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform PC {
    vec4 rect; vec4 uv;
    vec4 misc;  // layer, fade, saturation, hueShift(rad)
} pc;
void main() {
    vec3 c = texture(tex, vec3(vUV, vLayer)).rgb;
    // same filter order a camera wall uses: saturation -> hue -> fade
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(l), c, pc.misc.z);
    float co = cos(pc.misc.w), si = sin(pc.misc.w);
    c = clamp(mat3( .299+.701*co+.168*si,  .587-.587*co+.330*si,  .114-.114*co-.497*si,
                    .299-.299*co-.328*si,  .587+.413*co+.035*si,  .114-.114*co+.292*si,
                    .299-.300*co+1.25*si,  .587-.588*co-1.05*si,  .114+.886*co-.203*si) * c,
              0.0, 1.0);
    outColor = vec4(c * pc.misc.y, 1);
}
)GLSL";

struct Push {
    float rect[4], uv[4], misc[4]; // 48 bytes
};

static constexpr int kTiles = 9;
static constexpr uint32_t kSrc = 512; // source texture size per layer

/// Fill the 9 array layers with distinct procedural patterns.
static void fillSources(vkmini::Vk& vk, vk::Image img) {
    std::vector<uint8_t> px(kSrc * kSrc * 4 * kTiles);
    for (int t = 0; t < kTiles; ++t)
        for (uint32_t y = 0; y < kSrc; ++y)
            for (uint32_t x = 0; x < kSrc; ++x) {
                uint8_t* p = &px[(t * kSrc * kSrc + y * kSrc + x) * 4];
                // per-tile hue + gradients/checker so each tile differs
                p[0] = uint8_t((x * 255 / kSrc) ^ (t * 40));
                p[1] = uint8_t((y * 255 / kSrc) ^ (t * 80));
                p[2] = uint8_t(((x / 32 + y / 32 + t) & 1) ? 230 : 60);
                p[3] = 255;
            }
    auto st = vk.createBuffer(px.size(),
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);
    memcpy(st.mapped, px.data(), px.size());
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::ImageMemoryBarrier b{};
        b.setOldLayout(vk::ImageLayout::eUndefined)
         .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
         .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
         .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
         .setImage(img)
         .setSubresourceRange(
             {vk::ImageAspectFlagBits::eColor, 0, 1, 0, kTiles})
         .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::eTransfer,
                          {}, {}, {}, b);
        std::vector<vk::BufferImageCopy> rs(kTiles);
        for (int t = 0; t < kTiles; ++t)
            rs[t].setBufferOffset(t * kSrc * kSrc * 4)
                .setBufferRowLength(0).setBufferImageHeight(0)
                .setImageSubresource(
                    {vk::ImageAspectFlagBits::eColor, 0, uint32_t(t), 1})
                .setImageExtent({kSrc, kSrc, 1});
        c.copyBufferToImage(st.buf.get(), img,
                            vk::ImageLayout::eTransferDstOptimal, rs);
        b.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
         .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
         .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
         .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                          vk::PipelineStageFlagBits::eFragmentShader,
                          {}, {}, {}, b);
    });
}

/// Per-tile rect for cell `i` (0..8) at zoom `z` (0=grid,1=full),
/// with `fade` for the other tiles.
static Push pushFor(int i, float z, float W, float H) {
    int gx = i % 3, gy = i / 3;
    float x0 = gx / 3.f, y0 = gy / 3.f, s = 1.f / 3.f;
    // ease-out cubic on the normalized cell rect
    float e = 1.f - std::pow(1.f - z, 3.f);
    Push p{};
    p.rect[0] = x0 + (0.f - x0) * e;
    p.rect[1] = y0 + (0.f - y0) * e;
    p.rect[2] = s + (1.f - s) * e;
    p.rect[3] = s + (1.f - s) * e;
    p.uv[2] = p.uv[3] = 1.f;
    p.misc[0] = float(i);
    p.misc[1] = 1.f;
    p.misc[2] = 1.f;
    p.misc[3] = 0.f;
    (void)W; (void)H;
    return p;
}

int main(int argc, char** argv) {
    const char* ppm = nullptr;
    int zoomTile = -1;
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--headless") && i + 1 < argc)
            ppm = argv[++i];
        else if (!strcmp(argv[i], "--zoom") && i + 1 < argc)
            zoomTile = atoi(argv[++i]);

    vkmini::Vk vk;
    std::vector<const char*> exts;
    if (!ppm) {                       // windowed: GLFW's surface exts
        glfwInit();
        uint32_t n;
        const char** e = glfwGetRequiredInstanceExtensions(&n);
        exts.assign(e, e + n);
    }
    vk.createInstance(exts);
    vk.pickPhysicalDevice();
    vk.createDevice();

    // ---- one texture array holding all nine sources -------------
    vk::ImageCreateInfo ici{};
    ici.setImageType(vk::ImageType::e2D)
        .setFormat(vk::Format::eR8G8B8A8Unorm)
        .setExtent({kSrc, kSrc, 1})
        .setMipLevels(1).setArrayLayers(kTiles)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage(vk::ImageUsageFlagBits::eSampled |
                  vk::ImageUsageFlagBits::eTransferDst);
    auto srcImg = vk.device->createImageUnique(ici);
    auto req = vk.device->getImageMemoryRequirements(srcImg.get());
    auto srcMem = vk.device->allocateMemoryUnique(
        {req.size, vk.memoryType(req.memoryTypeBits,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal)});
    vk.device->bindImageMemory(srcImg.get(), srcMem.get(), 0);
    fillSources(vk, srcImg.get());
    vk::ImageViewCreateInfo vi{};
    vi.setImage(srcImg.get()).setViewType(vk::ImageViewType::e2DArray)
      .setFormat(vk::Format::eR8G8B8A8Unorm)
      .setSubresourceRange(
          {vk::ImageAspectFlagBits::eColor, 0, 1, 0, kTiles});
    auto srcView = vk.device->createImageViewUnique(vi);

    // ---- descriptors --------------------------------------------
    auto sampler = vk.device->createSamplerUnique(
        {{}, vk::Filter::eLinear, vk::Filter::eLinear,
         vk::SamplerMipmapMode::eNearest,
         vk::SamplerAddressMode::eClampToEdge,
         vk::SamplerAddressMode::eClampToEdge,
         vk::SamplerAddressMode::eClampToEdge});
    vk::DescriptorSetLayoutBinding b0{
        0, vk::DescriptorType::eCombinedImageSampler, 1,
        vk::ShaderStageFlagBits::eFragment};
    vk::DescriptorSetLayoutCreateInfo dslci{};
    dslci.setBindings(b0);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dslci);
    vk::DescriptorPoolSize ps{vk::DescriptorType::eCombinedImageSampler, 1};
    auto dpool = vk.device->createDescriptorPoolUnique({{}, 1, ps});
    vk::DescriptorSetLayout dslH = dsl.get();
    auto dsets = vk.device->allocateDescriptorSetsUnique(
        vk::DescriptorSetAllocateInfo{}.setDescriptorPool(dpool.get())
            .setSetLayouts(dslH));
    vk::DescriptorImageInfo dii{sampler.get(), srcView.get(),
                                vk::ImageLayout::eShaderReadOnlyOptimal};
    vk.device->updateDescriptorSets(
        vk::WriteDescriptorSet{dsets[0].get(), 0, 0,
            vk::DescriptorType::eCombinedImageSampler, dii}, {});

    // ---- pipeline ------------------------------------------------
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "w.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "w.frag");
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
    std::array dyn{vk::DynamicState::eViewport,
                   vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dsi{};
    dsi.setDynamicStates(dyn);
    vk::PipelineRasterizationStateCreateInfo rs{};
    rs.setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eNone)
        .setFrontFace(vk::FrontFace::eCounterClockwise)
        .setLineWidth(1.f);
    vk::PipelineMultisampleStateCreateInfo ms{};
    ms.setRasterizationSamples(vk::SampleCountFlagBits::e1);
    vk::PipelineColorBlendAttachmentState cba{};
    cba.setColorWriteMask(vk::ColorComponentFlagBits::eR |
                          vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB |
                          vk::ColorComponentFlagBits::eA);
    vk::PipelineColorBlendStateCreateInfo blend{};
    blend.setAttachments(cba);
    vk::PushConstantRange pcr{vk::ShaderStageFlagBits::eVertex |
                              vk::ShaderStageFlagBits::eFragment,
                              0, sizeof(Push)};
    vk::PipelineLayoutCreateInfo plci{};
    plci.setSetLayouts(dslH).setPushConstantRanges(pcr);
    auto layout = vk.device->createPipelineLayoutUnique(plci);
    vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
    vk::PipelineRenderingCreateInfo pr{};
    pr.setColorAttachmentFormats(cfmt);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(stages).setPVertexInputState(&vin)
      .setPInputAssemblyState(&ia).setPViewportState(&vps)
      .setPDynamicState(&dsi).setPRasterizationState(&rs)
      .setPMultisampleState(&ms).setPColorBlendState(&blend)
      .setLayout(layout.get()).setPNext(&pr);
    auto pipe = std::move(
        vk.device->createGraphicsPipelineUnique({}, gi).value);

    // ---- the wall: one draw per tile -----------------------------
    auto drawWall = [&](vk::CommandBuffer c, uint32_t W, uint32_t H,
                        int zoom, float zt) {
        c.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe.get());
        c.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                             layout.get(), 0, dsets[0].get(), {});
        vk::Viewport vp{0, 0, float(W), float(H), 0, 1};
        vk::Rect2D sc{{0, 0}, {W, H}};
        c.setViewport(0, vp); c.setScissor(0, sc);
        for (int i = 0; i < kTiles; ++i) {
            if (zoom >= 0 && i != zoom) continue; // grid fades out
            Push p = pushFor(i, zoom >= 0 ? zt : 0.f,
                             float(W), float(H));
            p.misc[3] = i * 0.35f; // demo: per-tile hue shift
            c.pushConstants<Push>(layout.get(),
                vk::ShaderStageFlagBits::eVertex |
                vk::ShaderStageFlagBits::eFragment, 0, p);
            c.draw(6, 1, 0, 0);
        }
    };

    if (ppm) {
        vkmini::Headless hl;
        uint32_t W = 960, H = 540;
        hl.init(vk, W, H);
        hl.render(vk::ImageLayout::eColorAttachmentOptimal,
                  [&](vk::CommandBuffer c) {
            vk::RenderingAttachmentInfo att{};
            att.setImageView(hl.color.view.get())
               .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
               .setLoadOp(vk::AttachmentLoadOp::eClear)
               .setClearValue(vk::ClearColorValue{
                   std::array{0.05f, 0.05f, 0.08f, 1.f}})
               .setStoreOp(vk::AttachmentStoreOp::eStore);
            vk::RenderingInfo ri{};
            ri.setRenderArea({{0, 0}, {W, H}}).setLayerCount(1)
              .setColorAttachments(att);
            c.beginRendering(ri);
            drawWall(c, W, H, zoomTile, 1.f);
            c.endRendering();
        });
        hl.savePpm(ppm);
        printf("wrote %s\n", ppm);
        return 0;
    }

    // ---- windowed: swapchain loop, click = zoom -------------------
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* win = glfwCreateWindow(960, 540, "vk wall", 0, 0);
    VkSurfaceKHR surf;
    glfwCreateWindowSurface(vk.instance.get(), win, 0, &surf);
    vkmini::Swapchain sc;
    sc.create(vk, vk.phys, surf, 960, 540);
    auto cbs = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::ePrimary,
         uint32_t(sc.images.size())});
    auto imageAv = vk.device->createSemaphoreUnique({});
    auto renderDone = vk.device->createSemaphoreUnique({});

    int zoom = -1; float zt = 0.f;
    auto last = std::chrono::steady_clock::now();
    glfwSetMouseButtonCallback(win,
        [](GLFWwindow* w, int btn, int act, int) {
            if (btn != GLFW_MOUSE_BUTTON_LEFT || act != GLFW_PRESS) return;
            auto* z = (int*)glfwGetWindowUserPointer(w);
            double mx, my; int W, H;
            glfwGetCursorPos(w, &mx, &my);
            glfwGetWindowSize(w, &W, &H);
            int cell = int(my / H * 3) * 3 + int(mx / W * 3);
            *z = (*z == cell) ? -1 : cell;
        });
    glfwSetWindowUserPointer(win, &zoom);

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        zt = std::clamp(zt + (zoom >= 0 ? dt : -dt) * 4.f, 0.f, 1.f);
        uint32_t idx = vk.device->acquireNextImageKHR(
            sc.sc.get(), UINT64_MAX, imageAv.get(), {}).value;
        auto& c = cbs[idx];
        c->reset();
        c->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::ImageMemoryBarrier b{};
        b.setOldLayout(vk::ImageLayout::eUndefined)
         .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
         .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
         .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
         .setImage(sc.images[idx])
         .setSubresourceRange(
             {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})
         .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
        c->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            {}, {}, {}, b);
        vk::RenderingAttachmentInfo att{};
        att.setImageView(sc.views[idx].get())
           .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
           .setLoadOp(vk::AttachmentLoadOp::eClear)
           .setClearValue(vk::ClearColorValue{
               std::array{0.05f, 0.05f, 0.08f, 1.f}})
           .setStoreOp(vk::AttachmentStoreOp::eStore);
        vk::RenderingInfo ri{};
        ri.setRenderArea({{0, 0}, sc.extent}).setLayerCount(1)
          .setColorAttachments(att);
        c->beginRendering(ri);
        drawWall(c.get(), sc.extent.width, sc.extent.height,
                 zoom, zoom < 0 ? 1.f - zt : zt);
        c->endRendering();
        b.setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
         .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
         .setSrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
        c->pipelineBarrier(
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, b);
        c->end();
        vk::PipelineStageFlags ws =
            vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::SubmitInfo si{};
        si.setWaitSemaphores(imageAv.get()).setWaitDstStageMask(ws)
          .setCommandBuffers(c.get())
          .setSignalSemaphores(renderDone.get());
        vk.queue.submit(si);
        vk::PresentInfoKHR pi{};
        pi.setWaitSemaphores(renderDone.get())
          .setSwapchains(sc.sc.get()).setImageIndices(idx);
        (void)vk.queue.presentKHR(pi);
        vk.queue.waitIdle(); // demo-grade sync: one frame in flight
    }
    vk.device->waitIdle();
    return 0;
}
