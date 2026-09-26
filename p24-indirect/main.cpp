// SPDX-License-Identifier: CC0-1.0
//
// Indirect drawing: vkCmdDrawIndirect reads its VkDrawIndirectCommand
// {vertexCount, instanceCount, firstVertex, firstInstance} records
// from a DEVICE-SIDE buffer instead of call arguments. One API call
// replays N draws — and because the buffer is device memory, a
// compute shader can WRITE the draw list (GPU-driven culling /
// count compaction) without a CPU round-trip.
//
// Here: a vertex buffer holds 4 colored quads back-to-back; a draw
// buffer holds 4 VkDrawIndirectCommand records; one
// vkCmdDrawIndirect renders all four.
//
//   ./app --headless o.ppm     one frame -> PPM for check.py
//   ./app                      windowed
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec3 inCol;
layout(location = 0) out vec3 vColor;
void main() {
    gl_Position = vec4(inPos, 0, 1);
    vColor = inCol;
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(vColor, 1.0); }
)GLSL";

struct Vert {
    float pos[2];
    float col[3];
};

int main(int argc, char** argv) {
    bool headless = argc > 1 && std::string(argv[1]) == "--headless";
    const char* ppmOut = argc > 2 ? argv[2] : "out.ppm";
    (void)headless;

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice({});

    // ---- vertex data: 4 quads, 6 verts each, RGB+white ----------
    auto quad = [](float cx, float cy, float s, float r, float g,
                   float b) {
        std::array<Vert, 6> q{};
        std::array p{std::array{-0.5f, -0.5f}, std::array{0.5f, -0.5f},
                     std::array{0.5f, 0.5f},   std::array{-0.5f, -0.5f},
                     std::array{0.5f, 0.5f},   std::array{-0.5f, 0.5f}};
        for (int i = 0; i < 6; ++i) {
            q[i].pos[0] = cx + p[i][0] * s;
            q[i].pos[1] = cy + p[i][1] * s;
            q[i].col[0] = r;
            q[i].col[1] = g;
            q[i].col[2] = b;
        }
        return q;
    };
    std::vector<Vert> verts;
    float cols[4][3] = {{1, .2f, .2f}, {.2f, 1, .2f},
                        {.2f, .2f, 1}, {1, 1, 1}};
    // 2x2 grid of quads
    for (int i = 0; i < 4; ++i) {
        float cx = (i % 2 == 0) ? -0.45f : 0.45f;
        float cy = (i < 2) ? -0.45f : 0.45f;
        auto q = quad(cx, cy, 0.6f, cols[i][0], cols[i][1],
                      cols[i][2]);
        verts.insert(verts.end(), q.begin(), q.end());
    }

    auto vstage = vk.createBuffer(
        verts.size() * sizeof(Vert),
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(vstage.mapped, verts.data(), verts.size() * sizeof(Vert));
    auto vbuf = vk.createBuffer(
        verts.size() * sizeof(Vert),
        vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::BufferCopy cp{0, 0, verts.size() * sizeof(Vert)};
        c.copyBuffer(vstage.buf.get(), vbuf.buf.get(), cp);
    });

    // ---- the INDIRECT draw buffer -------------------------------
    // Each record = one draw: 6 vertices starting at quad i*6.
    // Same CPU-writable buffer a compute shader could have filled.
    std::array<vk::DrawIndirectCommand, 4> draws{};
    for (int i = 0; i < 4; ++i)
        draws[i] = vk::DrawIndirectCommand{6, 1, uint32_t(i * 6), 0};
    auto istage = vk.createBuffer(
        sizeof(draws), vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(istage.mapped, draws.data(), sizeof(draws));
    auto ibuf = vk.createBuffer(
        sizeof(draws),
        vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eIndirectBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::BufferCopy cp{0, 0, sizeof(draws)};
        c.copyBuffer(istage.buf.get(), ibuf.buf.get(), cp);
    });

    // ---- pipeline ------------------------------------------------
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "s.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "s.frag");
    vk::PipelineShaderStageCreateInfo stages[2];
    stages[0].setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(vs.get())
        .setPName("main");
    stages[1].setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(fs.get())
        .setPName("main");
    vk::VertexInputBindingDescription bind{0, sizeof(Vert),
                                           vk::VertexInputRate::eVertex};
    std::array attribs{
        vk::VertexInputAttributeDescription{0, 0,
            vk::Format::eR32G32Sfloat, 0},
        vk::VertexInputAttributeDescription{1, 0,
            vk::Format::eR32G32B32Sfloat, 8}};
    vk::PipelineVertexInputStateCreateInfo vin{};
    vin.setVertexBindingDescriptions(bind)
        .setVertexAttributeDescriptions(attribs);
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
                      .setStoreOp(vk::AttachmentStoreOp::eStore)
                      .setClearValue(vk::ClearValue{
                          vk::ClearColorValue{
                              std::array{0.05f, 0.05f, 0.08f,
                                         1.f}}});
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
                  vk::DeviceSize off = 0;
                  c.bindVertexBuffers(0, vbuf.buf.get(), off);
                  // ONE call -> 4 draws, parameters read from ibuf
                  c.drawIndirect(ibuf.buf.get(), 0, 4,
                                 sizeof(vk::DrawIndirectCommand));
                  c.endRendering();
              });
    hl.savePpm(ppmOut);
    printf("wrote %s\n", ppmOut);
    return 0;
}
