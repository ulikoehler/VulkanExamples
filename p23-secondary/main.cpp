// SPDX-License-Identifier: CC0-1.0
//
// Secondary command buffers: only PRIMARY buffers can be submitted
// to a queue — but a primary can CALL secondaries via
// vkCmdExecuteCommands. That's the mechanism for multi-threaded
// recording: N threads each record a secondary, one thread's
// primary replays them inside its rendering.
//
// The demo records 3 secondaries (one per column of the grid) —
// trivially single-threaded here, but each could come from its own
// thread without any locking.
//
//   ./app --headless o.ppm     one frame -> PPM for check.py
//   ./app                      windowed
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) out vec3 vColor;
layout(push_constant) uniform Push { vec3 col; vec2 ofs; } pc;
void main() {
    vec2 p[6] = vec2[](vec2(-0.5,-0.5), vec2(0.5,-0.5), vec2(0.5,0.5),
                       vec2(-0.5,-0.5), vec2(0.5,0.5),  vec2(-0.5,0.5));
    gl_Position = vec4(p[gl_VertexIndex] * 0.25 + pc.ofs, 0, 1);
    vColor = pc.col;
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(vColor, 1.0); }
)GLSL";

struct Push {
    float col[3]; // vec3 at offset 0 (16-byte aligned)
    float pad;    // vec3 occupies 12 B but aligns next member to 16
    float ofs[2]; // vec2 at offset 16
};

/// The KEY restriction of a secondary that will run inside
/// rendering: it inherits the render state — declared via
/// VkCommandBufferInheritanceInfo (rendering variant here: the
/// dynamic-rendering version of VkCommandBufferInheritance-
/// RenderingInfo). The secondary contains NO beginRendering.
static void recordSecondary(vk::CommandBuffer c, vk::Format fmt,
                            vk::Extent2D extent,
                            vk::PipelineLayout layout,
                            vk::Pipeline pipe, float ox,
                            float r, float g, float b) {
    vk::CommandBufferInheritanceRenderingInfo inhr{};
    vk::Format f = fmt;
    inhr.setColorAttachmentFormats(f);
    vk::CommandBufferInheritanceInfo inh{};
    inh.setPNext(&inhr);
    vk::CommandBufferBeginInfo bi{};
    bi.setFlags(
            vk::CommandBufferUsageFlagBits::eRenderPassContinue)
        .setPInheritanceInfo(&inh);
    c.begin(bi);
    // secondaries do NOT inherit dynamic state from the primary —
    // with dynamic viewport/scissor we must set them HERE, or the
    // rasterizer clips everything to an undefined rect.
    vk::Viewport vp{0, 0, float(extent.width),
                    float(extent.height), 0.f, 1.f};
    c.setViewport(0, vp);
    vk::Rect2D sc{{0, 0}, extent};
    c.setScissor(0, sc);
    c.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe);
    Push p{{r, g, b}, 0.f, {ox, 0.f}};
    c.pushConstants(layout, vk::ShaderStageFlagBits::eVertex, 0,
                    sizeof(Push), &p);
    c.draw(6, 1, 0, 0);
    c.end();
}

int main(int argc, char** argv) {
    bool headless = argc > 1 && std::string(argv[1]) == "--headless";
    const char* ppmOut = argc > 2 ? argv[2] : "out.ppm";

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice({});
    (void)headless;
    (void)ppmOut;

    vk::Format fmt = vk::Format::eR8G8B8A8Unorm;
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "s.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "s.frag");
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
    vk::PushConstantRange pcr{};
    pcr.setStageFlags(vk::ShaderStageFlagBits::eVertex)
        .setOffset(0)
        .setSize(sizeof(Push));
    auto layout =
        vk.device->createPipelineLayoutUnique({{}, {}, pcr});
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
        .setLayout(layout.get())
        .setPNext(&rendering);
    auto pres = vk.device->createGraphicsPipelineUnique({}, gi);
    if (pres.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    auto pipe = std::move(pres.value);

    vkmini::Headless hl;
    hl.init(vk, 384, 384);

    // ---- record 3 SECONDARY buffers (could be 3 threads) --------
    auto secs = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::eSecondary, 3});
    recordSecondary(secs[0].get(), fmt, hl.extent, layout.get(),
                    pipe.get(), -0.6f, 1.f, 0.2f, 0.2f);
    recordSecondary(secs[1].get(), fmt, hl.extent, layout.get(),
                    pipe.get(), 0.f, 0.2f, 1.f, 0.2f);
    recordSecondary(secs[2].get(), fmt, hl.extent, layout.get(),
                    pipe.get(), 0.6f, 0.2f, 0.2f, 1.f);
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
                  vk::Viewport vp{0, 0, 384, 384, 0.f, 1.f};
                  c.setViewport(0, vp);
                  vk::Rect2D sc{{0, 0}, hl.extent};
                  c.setScissor(0, sc);
                  // replay all 3 secondaries inside OUR rendering
                  for (auto& s : secs)
                      c.executeCommands(s.get());
                  c.endRendering();
              });
    hl.savePpm(ppmOut);
    printf("wrote %s\n", ppmOut);
    return 0;
}
