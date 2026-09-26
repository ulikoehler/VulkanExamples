// SPDX-License-Identifier: CC0-1.0
//
// GPU timestamps: vkCmdWriteTimestamp drops a hardware timestamp
// into a VkQueryPool at a chosen pipeline stage; after the frame
// completes, vkGetQueryPoolResults returns ticks. Multiply by
// VkPhysicalDeviceLimits::timestampPeriod to get nanoseconds.
// This measures GPU-side time — including async queue work that
// CPU-side chrono never sees.
//
// Demo: timestamp before + after a fullscreen draw, diff reported.
//
//   ./app --headless o.ppm     one frame -> PPM + timing print
//   ./app                      windowed
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* kVert = R"GLSL(
#version 460
void main() {
    vec2 p = vec2((gl_VertexIndex << 1 & 2), gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0, 1);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) out vec4 outColor;
void main() {
    // enough work to produce a measurable time on any GPU
    float acc = 0.0;
    for (int i = 0; i < 64; ++i)
        acc += sin(float(i) * gl_FragCoord.x * 0.001);
    outColor = vec4(abs(acc), 0.3, 0.5, 1.0);
}
)GLSL";

int main(int argc, char** argv) {
    bool headless = argc > 1 && std::string(argv[1]) == "--headless";
    const char* ppmOut = argc > 2 ? argv[2] : "out.ppm";
    (void)headless;

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice({});

    auto props = vk.phys.getProperties();
    float nsPerTick = props.limits.timestampPeriod;
    printf("timestampPeriod: %f ns/tick\n", nsPerTick);
    if (!props.limits.timestampComputeAndGraphics)
        throw std::runtime_error("queue cannot timestamp");

    // query pool of 2 timestamps
    auto qp = vk.device->createQueryPoolUnique(
        {{}, vk::QueryType::eTimestamp, 2});

    // minimal graphics pipeline
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
    hl.render(
        vk::ImageLayout::eColorAttachmentOptimal,
        [&](vk::CommandBuffer c) {
            // reset BEFORE recording the writes (query pools are
            // reset in a command buffer or via vkResetQueryPool)
            c.resetQueryPool(qp.get(), 0, 2);
            // timestamp at the TOP of the graphics pipe
            c.writeTimestamp(
                vk::PipelineStageFlagBits::eTopOfPipe, qp.get(),
                0);
            vk::RenderingAttachmentInfo att{};
            att.setImageView(hl.color.view.get())
                .setImageLayout(
                    vk::ImageLayout::eColorAttachmentOptimal)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setClearValue(vk::ClearValue{vk::ClearColorValue{
                    std::array{0.f, 0.f, 0.f, 1.f}}});
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
            c.draw(3, 1, 0, 0);
            c.endRendering();
            // bottom of pipe = after ALL the work above retired
            c.writeTimestamp(
                vk::PipelineStageFlagBits::eBottomOfPipe,
                qp.get(), 1);
        });
    // results are valid once the command buffer's fence waited —
    // hl.render already waited; getQueryPoolResults is then
    // instant.
    uint64_t ts[2];
    auto r = vk.device->getQueryPoolResults(
        qp.get(), 0, 2, sizeof(ts), ts, sizeof(uint64_t),
        vk::QueryResultFlagBits::e64);
    if (r != vk::Result::eSuccess)
        throw std::runtime_error("query pool read failed");
    double ns = double(ts[1] - ts[0]) * double(nsPerTick);
    printf("gpu time: %.1f us (%llu ticks @ %.0f ns)\n", ns / 1000.0,
           (unsigned long long)(ts[1] - ts[0]), double(nsPerTick));
    hl.savePpm(ppmOut);
    printf("wrote %s\n", ppmOut);
    return 0;
}
