// SPDX-License-Identifier: CC0-1.0
//
// Text in Vulkan without pulling in a UI toolkit:
//   HarfBuzz shapes the string (positions + glyph ids),
//   FreeType rasterizes each glyph's coverage,
//   a 2-pass chamfer turns coverage into a signed distance field,
//   packed into one R8 atlas texture,
//   one instanced-style quad draw per glyph, smoothstep in the shader.
//
// SDF means the SAME atlas renders tiny labels and huge titles
// without resampling artifacts — and outline/glow is a free shader
// branch on the distance value.
//
//   ./app --headless out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc \
//             $(pkg-config --cflags --libs freetype2 harfbuzz)

#include "vkmini.hpp"
#include <ft2build.h>
#include FT_FREETYPE_H
#include <hb.h>
#include <hb-ft.h>
#include <cmath>
#include <map>

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) in vec2 inPos;   // pixels
layout(location = 1) in vec2 inUV;    // atlas 0..1
layout(push_constant) uniform PC { vec4 xf; } pc; // scale.xy, ofs.xy
layout(location = 0) out vec2 vUV;
void main() {
    gl_Position = vec4(inPos * pc.xf.xy + pc.xf.zw, 0, 1);
    vUV = inUV;
}
)GLSL";

// SDF value 0.5 = edge; >0.5 inside. fwidth-scaled smoothstep gives
// resolution-independent antialiasing.
static const char* kFrag = R"GLSL(
#version 460
layout(binding = 0) uniform sampler2D atlas;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() {
    float d = texture(atlas, vUV).r;
    float w = fwidth(d) * 1.2;
    float a = smoothstep(0.5 - w, 0.5 + w, d);
    float glow = (smoothstep(0.40, 0.5, d) - a) * 0.8;
    vec3 rgb = vec3(1.0, 0.85, 0.2) * a + vec3(0.5, 0.35, 1.0) * glow;
    outColor = vec4(rgb, a + glow);  // alpha drives blending
}
)GLSL";

// ------------------------------------------------------------------
// atlas: per-glyph SDF, chamfer distance transform
// ------------------------------------------------------------------
static constexpr int kCell = 48;   // atlas cell px
static constexpr int kCols = 16;   // 16x16 cells = 256 glyph slots
static constexpr float kSpread = 6.f;

struct Glyph { float u0, v0, u1, v1; float bx, by, w, h; };

/// Chamfer distance transform: returns dist (px) to nearest `true`
/// pixel of mask (w*h), using 3-4 / 1-4 weights approximated by ints.
static std::vector<float> distTo(const uint8_t* m, int w, int h,
                                 uint8_t what) {
    const float INF = 1e9f;
    std::vector<float> d(w * h, INF);
    for (int i = 0; i < w * h; ++i) if ((m[i] > 127) == what) d[i] = 0;
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        int i = y * w + x;
        if (x > 0)     d[i] = std::min(d[i], d[i-1] + 1.f);
        if (y > 0) {
            d[i] = std::min(d[i], d[i-w] + 1.f);
            if (x > 0)     d[i] = std::min(d[i], d[i-w-1] + 1.414f);
            if (x < w - 1) d[i] = std::min(d[i], d[i-w+1] + 1.414f);
        }
    }
    for (int y = h - 1; y >= 0; --y) for (int x = w - 1; x >= 0; --x) {
        int i = y * w + x;
        if (x < w - 1) d[i] = std::min(d[i], d[i+1] + 1.f);
        if (y < h - 1) {
            d[i] = std::min(d[i], d[i+w] + 1.f);
            if (x < w - 1) d[i] = std::min(d[i], d[i+w+1] + 1.414f);
            if (x > 0)     d[i] = std::min(d[i], d[i+w-1] + 1.414f);
        }
    }
    return d;
}

int main(int argc, char** argv) {
    const char* ppm = "out.ppm";
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--headless") && i + 1 < argc)
            ppm = argv[++i];

    const char* fontPath =
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf";
    const char* text = "Vulkan <3 SDF text — CAM 1";

    // ---- FreeType + HarfBuzz -------------------------------------
    FT_Library ft; FT_Init_FreeType(&ft);
    FT_Face face;
    if (FT_New_Face(ft, fontPath, 0, &face))
        { fprintf(stderr, "no font at %s\n", fontPath); return 1; }
    constexpr int kPx = 40; // render size (atlas target size ~48 cell)
    FT_Set_Pixel_Sizes(face, 0, kPx);

    hb_font_t* hbFont = hb_ft_font_create(face, nullptr);
    hb_buffer_t* buf = hb_buffer_create();
    hb_buffer_add_utf8(buf, text, -1, 0, -1);
    hb_buffer_guess_segment_properties(buf);
    hb_shape(hbFont, buf, nullptr, 0);
    unsigned ng; auto* info = hb_buffer_get_glyph_infos(buf, &ng);
    auto* pos = hb_buffer_get_glyph_positions(buf, &ng);

    // ---- rasterize each used glyph -> SDF cell --------------------
    const int A = kCols * kCell;             // atlas px (768)
    std::vector<uint8_t> atlas(A * A, 0);
    std::map<uint32_t, Glyph> glyphs;
    int next = 0;
    for (unsigned i = 0; i < ng; ++i) {
        uint32_t g = info[i].codepoint;
        if (glyphs.count(g)) continue;
        if (FT_Load_Glyph(face, g, FT_LOAD_DEFAULT) ||
            FT_Render_Glyph(face->glyph, FT_RENDER_MODE_NORMAL))
            continue;
        auto& bm = face->glyph->bitmap;
        int gw = bm.width, gh = bm.rows;
        if (gw > kCell - 4 || gh > kCell - 4) { next++; continue; }
        int cx = (next % kCols) * kCell, cy = (next / kCols) * kCell;
        ++next;
        // coverage -> inside field centered in the cell
        std::vector<uint8_t> cov(kCell * kCell, 0);
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x)
                cov[(y + 2) * kCell + x + 2] = bm.buffer[y * gw + x];
        auto dIn = distTo(cov.data(), kCell, kCell, true);   // dist to inside
        auto dOut = distTo(cov.data(), kCell, kCell, false); // dist to outside
        for (int y = 0; y < kCell; ++y)
            for (int x = 0; x < kCell; ++x) {
                int j = y * kCell + x;
                // signed dist: inside pixels measure to outside edge
                float sd = (cov[j] > 127)
                    ? dOut[j]          // inside: distance to outside
                    : -dIn[j];         // outside: negative
                float v = 0.5f + sd / (2.f * kSpread);
                atlas[(cy + y) * A + cx + x] =
                    uint8_t(std::clamp(v, 0.f, 1.f) * 255);
            }
        Glyph gl{};
        gl.u0 = float(cx) / A;  gl.v0 = float(cy) / A;
        gl.u1 = float(cx + kCell) / A; gl.v1 = float(cy + kCell) / A;
        gl.bx = float(face->glyph->bitmap_left - 2);
        gl.by = float(face->glyph->bitmap_top + 2);
        gl.w = float(kCell); gl.h = float(kCell);
        // scale cell coords back to em space: coverage was padded by 2
        glyphs[g] = gl;
    }

    // ---- build quads ----------------------------------------------
    struct V { float x, y, u, v; };
    std::vector<V> quads;
    float penX = 30.f, penY = 240.f;
    for (unsigned i = 0; i < ng; ++i) {
        auto it = glyphs.find(info[i].codepoint);
        if (it == glyphs.end()) { penX += pos[i].x_advance / 64.f;
                                  penY -= pos[i].y_advance / 64.f;
                                  continue; }
        auto& g = it->second;
        float gx = penX + pos[i].x_offset / 64.f + g.bx;
        float gy = penY - pos[i].y_offset / 64.f - g.by;
        float x0 = gx, y0 = gy, x1 = gx + g.w, y1 = gy + g.h;
        V quad[6] = {
            {x0,y0,g.u0,g.v0},{x1,y0,g.u1,g.v0},{x0,y1,g.u0,g.v1},
            {x0,y1,g.u0,g.v1},{x1,y0,g.u1,g.v0},{x1,y1,g.u1,g.v1}};
        for (auto& v : quad) quads.push_back(v);
        penX += pos[i].x_advance / 64.f;
        penY -= pos[i].y_advance / 64.f;
    }
    printf("shaped %u glyphs, %zu vertices\n", ng, quads.size());

    // ---- Vulkan ---------------------------------------------------
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice();

    auto atlasImg = vk.createImage(A, A, vk::Format::eR8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst);
    auto st = vk.createBuffer(atlas.size(),
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);
    memcpy(st.mapped, atlas.data(), atlas.size());
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::ImageMemoryBarrier b{};
        b.setOldLayout(vk::ImageLayout::eUndefined)
         .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
         .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
         .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
         .setImage(atlasImg.img.get())
         .setSubresourceRange({vk::ImageAspectFlagBits::eColor,0,1,0,1})
         .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, b);
        vk::BufferImageCopy r{};
        r.setImageSubresource({vk::ImageAspectFlagBits::eColor,0,0,1})
         .setImageExtent({uint32_t(A), uint32_t(A), 1});
        c.copyBufferToImage(st.buf.get(), atlasImg.img.get(),
            vk::ImageLayout::eTransferDstOptimal, r);
        b.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
         .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
         .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
         .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, b);
    });

    auto vbuf = vk.createBuffer(quads.size() * sizeof(V),
        vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);
    memcpy(vbuf.mapped, quads.data(), quads.size() * sizeof(V));

    auto sampler = vk.device->createSamplerUnique(
        {{}, vk::Filter::eLinear, vk::Filter::eLinear,
         vk::SamplerMipmapMode::eNearest,
         vk::SamplerAddressMode::eClampToEdge,
         vk::SamplerAddressMode::eClampToEdge,
         vk::SamplerAddressMode::eClampToEdge});
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
    vk::DescriptorImageInfo di{sampler.get(), atlasImg.view.get(),
                               vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::WriteDescriptorSet w{};
    w.setDstSet(dsets[0].get()).setDstBinding(0)
     .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
     .setImageInfo(di);
    vk.device->updateDescriptorSets(w, {});

    auto vs = vk.shader(kVert, shaderc_vertex_shader, "t.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "t.frag");
    vk::PipelineShaderStageCreateInfo st2[2];
    st2[0].setStage(vk::ShaderStageFlagBits::eVertex).setModule(vs.get())
        .setPName("main");
    st2[1].setStage(vk::ShaderStageFlagBits::eFragment).setModule(fs.get())
        .setPName("main");
    vk::VertexInputBindingDescription vib{0, sizeof(V)};
    std::array via{
        vk::VertexInputAttributeDescription{0, 0,
            vk::Format::eR32G32Sfloat, 0},
        vk::VertexInputAttributeDescription{1, 0,
            vk::Format::eR32G32Sfloat, 8}};
    vk::PipelineVertexInputStateCreateInfo vin{};
    vin.setVertexBindingDescriptions(vib)
       .setVertexAttributeDescriptions(via);
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
      .setLineWidth(1.f);
    vk::PipelineMultisampleStateCreateInfo ms{};
    ms.setRasterizationSamples(vk::SampleCountFlagBits::e1);
    vk::PipelineColorBlendAttachmentState cba{};
    cba.setColorWriteMask(vk::ColorComponentFlagBits::eR |
                          vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB |
                          vk::ColorComponentFlagBits::eA)
       .setBlendEnable(VK_TRUE)
       .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
       .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
       .setColorBlendOp(vk::BlendOp::eAdd)
       .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
       .setDstAlphaBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
       .setAlphaBlendOp(vk::BlendOp::eAdd);
    vk::PipelineColorBlendStateCreateInfo blend{};
    blend.setAttachments(cba);
    vk::PushConstantRange pcr{vk::ShaderStageFlagBits::eVertex, 0, 16};
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

    constexpr uint32_t W = 960, H = 480;
    vkmini::Headless hl;
    hl.init(vk, W, H);
    hl.render(vk::ImageLayout::eColorAttachmentOptimal,
              [&](vk::CommandBuffer c) {
        vk::RenderingAttachmentInfo att{};
        att.setImageView(hl.color.view.get())
           .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
           .setLoadOp(vk::AttachmentLoadOp::eClear)
           .setClearValue(vk::ClearColorValue{
               std::array{0.06f, 0.06f, 0.10f, 1.f}})
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
        c.bindVertexBuffers(0, vbuf.buf.get(), vk::DeviceSize{0});
        float xf[4] = {2.f / W, 2.f / H, -1.f, -1.f}; // px -> NDC
        c.pushConstants(layout.get(), vk::ShaderStageFlagBits::eVertex,
                        0, 16, xf);
        c.draw(uint32_t(quads.size()), 1, 0, 0);
        c.endRendering();
    });
    hl.savePpm(ppm);
    printf("wrote %s\n", ppm);
    return 0;
}
