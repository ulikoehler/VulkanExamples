// SPDX-License-Identifier: CC0-1.0
//
// Aspect-correct image placement: the same 4:3 source drawn into
// three cells as FIT (letterbox, whole image visible), FILL
// (stretch, distorts) and COVER (crop, fills the cell). The math is
// two divisions — but every broken video wall gets it wrong, so
// here's the verified version.
//
//   ./app --headless out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc

#include "vkmini.hpp"
#include <cmath>

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) out vec2 vUV;
layout(push_constant) uniform PC {
    vec4 dst;   // x,y,w,h — target rect in 0..1 window space
    vec4 src;   // x,y,w,h — source crop in 0..1 texture space
} pc;
void main() {
    const vec2 QUAD[6] = vec2[](vec2(0,0), vec2(1,0), vec2(0,1),
                                vec2(0,1), vec2(1,0), vec2(1,1));
    vec2 q = QUAD[gl_VertexIndex];
    vec2 p = pc.dst.xy + q * pc.dst.zw;
    gl_Position = vec4(p * 2.0 - 1.0, 0, 1);  // Vulkan NDC: +y down
    vUV = pc.src.xy + q * pc.src.zw;
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(texture(tex, vUV).rgb, 1); }
)GLSL";

struct Push { float dst[4], src[4]; };

/// FIT: whole image visible, aspect kept -> pillar/letterbox bands.
/// Returns the dst rect to draw into (sub-rect of `cell`).
static void fitRect(float* dst, float cx, float cy, float cw, float ch,
                    float srcAspect) {
    float cellAspect = cw / ch;
    if (cellAspect > srcAspect) {         // cell wider -> pillars
        float w = ch * srcAspect;
        dst[0] = cx + (cw - w) * 0.5f; dst[1] = cy;
        dst[2] = w;                      dst[3] = ch;
    } else {                              // cell taller -> letterbox
        float h = cw / srcAspect;
        dst[0] = cx;                      dst[1] = cy + (ch - h) * 0.5f;
        dst[2] = cw;                      dst[3] = h;
    }
}

/// COVER: fill the whole cell, crop the overflow. Returns the src
/// rect (a center crop of the source in 0..1 UV space).
static void coverSrc(float* src, float cw, float ch, float srcAspect) {
    float cellAspect = cw / ch;
    src[0] = 0; src[1] = 0; src[2] = 1; src[3] = 1;
    if (cellAspect > srcAspect) {         // cell wider -> crop top/bot
        float keep = srcAspect / cellAspect;
        src[1] = (1.f - keep) * 0.5f; src[3] = keep;
    } else {                              // crop left/right
        float keep = cellAspect / srcAspect;
        src[0] = (1.f - keep) * 0.5f; src[2] = keep;
    }
}

int main(int argc, char** argv) {
    const char* ppm = "out.ppm";
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--headless") && i + 1 < argc)
            ppm = argv[++i];

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice();

    // ---- source: 4:3 with distinctive border + center marks -------
    constexpr uint32_t SW = 320, SH = 240;
    auto src = vk.createImage(SW, SH, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst);
    std::vector<uint8_t> px(SW * SH * 4);
    for (uint32_t y = 0; y < SH; ++y)
        for (uint32_t x = 0; x < SW; ++x) {
            auto* p = &px[(y * SW + x) * 4];
            bool border = x < 8 || y < 8 || x >= SW - 8 || y >= SH - 8;
            p[0] = border ? 255 : uint8_t(x * 255 / SW);
            p[1] = border ? 255 : uint8_t(y * 255 / SH);
            p[2] = border ? 0 : 64;
            p[3] = 255;
        }
    auto st = vk.createBuffer(px.size(), vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);
    memcpy(st.mapped, px.data(), px.size());
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::ImageMemoryBarrier b{};
        b.setOldLayout(vk::ImageLayout::eUndefined)
         .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
         .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
         .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
         .setImage(src.img.get())
         .setSubresourceRange({vk::ImageAspectFlagBits::eColor,0,1,0,1})
         .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, b);
        vk::BufferImageCopy r{};
        r.setImageSubresource({vk::ImageAspectFlagBits::eColor,0,0,1})
         .setImageExtent({SW, SH, 1});
        c.copyBufferToImage(st.buf.get(), src.img.get(),
            vk::ImageLayout::eTransferDstOptimal, r);
        b.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
         .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
         .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
         .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, b);
    });

    auto sampler = vk.device->createSamplerUnique(
        {{}, vk::Filter::eNearest, vk::Filter::eNearest});
    vk::DescriptorSetLayoutBinding b0{0,
        vk::DescriptorType::eCombinedImageSampler, 1,
        vk::ShaderStageFlagBits::eFragment};
    vk::DescriptorSetLayoutCreateInfo dslci{};
    dslci.setBindings(b0);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dslci);
    vk::DescriptorPoolSize dps{vk::DescriptorType::eCombinedImageSampler, 1};
    auto dpool = vk.device->createDescriptorPoolUnique({{}, 1, dps});
    auto dsets = vk.device->allocateDescriptorSetsUnique(
        vk::DescriptorSetAllocateInfo{}.setDescriptorPool(dpool.get())
            .setSetLayouts(dsl.get()));
    vk::DescriptorImageInfo di{sampler.get(), src.view.get(),
                               vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::WriteDescriptorSet w{};
    w.setDstSet(dsets[0].get()).setDstBinding(0)
     .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
     .setImageInfo(di);
    vk.device->updateDescriptorSets(w, {});

    auto vs = vk.shader(kVert, shaderc_vertex_shader, "l.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "l.frag");
    vk::PipelineShaderStageCreateInfo st2[2];
    st2[0].setStage(vk::ShaderStageFlagBits::eVertex).setModule(vs.get())
        .setPName("main");
    st2[1].setStage(vk::ShaderStageFlagBits::eFragment).setModule(fs.get())
        .setPName("main");
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
      .setCullMode(vk::CullModeFlagBits::eNone).setLineWidth(1.f);
    vk::PipelineMultisampleStateCreateInfo ms{};
    ms.setRasterizationSamples(vk::SampleCountFlagBits::e1);
    vk::PipelineColorBlendAttachmentState cba{};
    cba.setColorWriteMask(vk::ColorComponentFlagBits::eR |
                          vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB |
                          vk::ColorComponentFlagBits::eA);
    vk::PipelineColorBlendStateCreateInfo blend{};
    blend.setAttachments(cba);
    vk::PushConstantRange pcr{vk::ShaderStageFlagBits::eVertex,
                            0, sizeof(Push)};
    vk::PipelineLayoutCreateInfo plci{};
    plci.setSetLayouts(dsl.get()).setPushConstantRanges(pcr);
    auto layout = vk.device->createPipelineLayoutUnique(plci);
    vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
    vk::PipelineRenderingCreateInfo pr{};
    pr.setColorAttachmentFormats(cfmt);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(st2).setPVertexInputState(&vin)
      .setPInputAssemblyState(&ia).setPViewportState(&vps)
      .setPDynamicState(&dsi).setPRasterizationState(&rs)
      .setPMultisampleState(&ms).setPColorBlendState(&blend)
      .setLayout(layout.get()).setPNext(&pr);
    auto pipe = std::move(
        vk.device->createGraphicsPipelineUnique({}, gi).value);

    // ---- three cells: FIT | FILL | COVER ---------------------------
    constexpr uint32_t W = 960, H = 400;
    const float srcAspect = float(SW) / SH;   // 4:3 = 1.333
    vkmini::Headless hl;
    hl.init(vk, W, H);
    hl.render(vk::ImageLayout::eColorAttachmentOptimal,
              [&](vk::CommandBuffer c) {
        vk::RenderingAttachmentInfo att{};
        att.setImageView(hl.color.view.get())
           .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
           .setLoadOp(vk::AttachmentLoadOp::eClear)
           .setClearValue(vk::ClearColorValue{
               std::array{0.02f, 0.02f, 0.35f, 1.f}})  // navy bands
           .setStoreOp(vk::AttachmentStoreOp::eStore);
        vk::RenderingInfo ri{};
        ri.setRenderArea({{0,0},{W,H}}).setLayerCount(1)
          .setColorAttachments(att);
        c.beginRendering(ri);
        c.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe.get());
        vk::Viewport vp{0,0,float(W),float(H),0,1};
        vk::Rect2D sc{{0,0},{W,H}};
        c.setViewport(0, vp); c.setScissor(0, sc);
        c.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                             layout.get(), 0, dsets[0].get(), {});
        // cells are 16:9-ish wide — a 4:3 source must pillarbox in FIT
        for (int i = 0; i < 3; ++i) {
            float cx = i / 3.f + 0.01f, cy = 0.08f,
                  cw = 1.f / 3.f - 0.02f, ch = 0.84f;
            Push p{};
            p.src[2] = p.src[3] = 1.f;
            if (i == 0) fitRect(p.dst, cx, cy, cw, ch, srcAspect);
            else if (i == 1) {           // FILL: whole cell, distorted
                p.dst[0] = cx; p.dst[1] = cy;
                p.dst[2] = cw; p.dst[3] = ch;
            } else                       // COVER: crop, fills cell
                { p.dst[0] = cx; p.dst[1] = cy; p.dst[2] = cw;
                  p.dst[3] = ch; coverSrc(p.src, cw, ch, srcAspect); }
            c.pushConstants<Push>(layout.get(),
                vk::ShaderStageFlagBits::eVertex, 0, p);
            c.draw(6, 1, 0, 0);
        }
        c.endRendering();
    });
    hl.savePpm(ppm);
    printf("wrote %s\n", ppm);
    return 0;
}
