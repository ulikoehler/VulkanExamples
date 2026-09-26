// SPDX-License-Identifier: CC0-1.0
//
// Vertex pulling: throw away VkVertexInputBindingDescription
// entirely — the vertex shader reads gl_VertexIndex and FETCHES its
// own vertex from an SSBO. No fixed-function vertex fetch, no
// format table, no binding granularity — arbitrary vertex layouts
// (AoS, SoA, compressed, quantized) decided in shader code.
//
// Here: an SSBO holds 24 Vert{x,y,r,g,b}; the shader indexes it
// with gl_VertexIndex. Same 2x2 grid as the indirect post — to
// prove the output is identical without ANY vertex input state.
//
//   ./app --headless o.ppm     one frame -> PPM for check.py
//   ./app                      windowed
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* kVert = R"GLSL(
#version 460
struct Vert { float x, y; float r, g, b; };
layout(std430, binding = 0) readonly buffer Verts { Vert v[]; };
layout(location = 0) out vec3 vColor;
void main() {
    Vert vert = v[gl_VertexIndex];   // THE pull
    gl_Position = vec4(vert.x, vert.y, 0, 1);
    vColor = vec3(vert.r, vert.g, vert.b);
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

    // ---- vertices as a plain STORAGE buffer ---------------------
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
    for (int i = 0; i < 4; ++i) {
        float cx = (i % 2 == 0) ? -0.45f : 0.45f;
        float cy = (i < 2) ? -0.45f : 0.45f;
        auto q = quad(cx, cy, 0.6f, cols[i][0], cols[i][1],
                      cols[i][2]);
        verts.insert(verts.end(), q.begin(), q.end());
    }

    auto stage = vk.createBuffer(
        verts.size() * sizeof(Vert),
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(stage.mapped, verts.data(), verts.size() * sizeof(Vert));
    // eStorageBuffer — NOT eVertexBuffer. There's no vertex buffer.
    auto ssbo = vk.createBuffer(
        verts.size() * sizeof(Vert),
        vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::BufferCopy cp{0, 0, verts.size() * sizeof(Vert)};
        c.copyBuffer(stage.buf.get(), ssbo.buf.get(), cp);
    });

    // ---- descriptor: binding 0 = the SSBO ------------------------
    vk::DescriptorSetLayoutBinding binding{
        0, vk::DescriptorType::eStorageBuffer, 1,
        vk::ShaderStageFlagBits::eVertex};
    vk::DescriptorSetLayoutCreateInfo dslci{};
    dslci.setBindings(binding);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dslci);
    vk::DescriptorPoolSize ps{vk::DescriptorType::eStorageBuffer, 1};
    auto dpool = vk.device->createDescriptorPoolUnique(
        {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1,
         ps});
    vk::DescriptorSetLayout dslH = dsl.get();
    vk::DescriptorSetAllocateInfo dsai{};
    dsai.setDescriptorPool(dpool.get())
        .setSetLayouts(dslH);
    auto dsets = vk.device->allocateDescriptorSetsUnique(dsai);
    vk::DescriptorBufferInfo dbi{ssbo.buf.get(), 0,
                                 verts.size() * sizeof(Vert)};
    vk::WriteDescriptorSet w{};
    w.setDstSet(dsets[0].get())
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setBufferInfo(dbi);
    vk.device->updateDescriptorSets(w, {});

    // ---- pipeline — note: EMPTY vertex input state ---------------
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "s.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "s.frag");
    vk::PipelineShaderStageCreateInfo stages[2];
    stages[0].setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(vs.get())
        .setPName("main");
    stages[1].setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(fs.get())
        .setPName("main");
    vk::PipelineVertexInputStateCreateInfo vin{};   // zero bindings!
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
    plci.setSetLayouts(dslH);
    auto layout = vk.device->createPipelineLayoutUnique(plci);
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
                  c.bindDescriptorSets(
                      vk::PipelineBindPoint::eGraphics, layout.get(),
                      0, dsets[0].get(), {});
                  // draw(24) — vertex shader pulls all data itself
                  c.draw(24, 1, 0, 0);
                  c.endRendering();
              });
    hl.savePpm(ppmOut);
    printf("wrote %s\n", ppmOut);
    return 0;
}
