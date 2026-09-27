// SPDX-License-Identifier: CC0-1.0
//
// VK_KHR_shader_clock — read the GPU clock INSIDE the shader.
// Where timestamp queries (p56) measure whole command spans,
// clockRealtimeEXT/clockARB tells you which PIXEL cost how many
// cycles. This renders a cycle-count heatmap of a raymarched-ish
// fragment workload.
//
//   ./app --headless out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

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

// Left half: cheap shader. Right half: expensive loop. The clock
// reads bracket the work and become the output pixel.
static const char* kFrag = R"GLSL(
#version 460
#extension GL_EXT_shader_realtime_clock : enable
#extension GL_ARB_gpu_shader_int64 : enable
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() {
    uvec2 t0 = clockRealtime2x32EXT();   // or clock2x32ARB()
    // deliberately expensive: nested trig, iteration count varies
    // with position so the heatmap isn't flat
    float acc = 0.0;
    int iters = (vUV.x < 0.5) ? 4 : int(300 + 3000 * vUV.y);
    for (int i = 0; i < iters; ++i)
        acc += sin(float(i) * vUV.x * 91.7 + vUV.y * 33.3)
             * cos(float(i) * 0.37);
    uvec2 t1 = clockRealtime2x32EXT();
    uint64_t cycles = (uint64_t(t1.y) << 32 | t1.x) -
                      (uint64_t(t0.y) << 32 | t0.x);
    // heatmap: blue = cheap, red = expensive. ~40k cycles saturates.
    float heat = clamp(float(cycles) / 60000.0, 0.0, 1.0);
    outColor = vec4(heat, 0.1, 1.0 - heat, 1.0) +
               vec4(0.02) * acc; // keep acc alive
}
)GLSL";

int main(int argc, char** argv) {
    const char* ppm = "out.ppm";
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--headless") && i + 1 < argc)
            ppm = argv[++i];

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    // feature: shaderDeviceClock for clockRealtime, subgroup clock
    // for clockARB in subgroups
    vk::PhysicalDeviceShaderClockFeaturesKHR clockFeat{};
    clockFeat.setShaderDeviceClock(true).setShaderSubgroupClock(true);
    vk.createDevice({VK_KHR_SHADER_CLOCK_EXTENSION_NAME}, &clockFeat);

    constexpr uint32_t W = 640, Hh = 360;
    vkmini::Headless hl;
    hl.init(vk, W, Hh);

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
    vk::PipelineViewportStateCreateInfo vps{};
    vps.setViewportCount(1).setScissorCount(1);
    std::array dynSt{vk::DynamicState::eViewport,
                     vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dyn{};
    dyn.setDynamicStates(dynSt);
    vk::PipelineRasterizationStateCreateInfo rs{};
    rs.setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eNone)
        .setFrontFace(vk::FrontFace::eCounterClockwise)
        .setLineWidth(1.0f);
    vk::PipelineMultisampleStateCreateInfo ms{};
    ms.setRasterizationSamples(vk::SampleCountFlagBits::e1);
    vk::PipelineColorBlendAttachmentState cb{};
    cb.setColorWriteMask(vk::ColorComponentFlagBits::eR |
                         vk::ColorComponentFlagBits::eG |
                         vk::ColorComponentFlagBits::eB |
                         vk::ColorComponentFlagBits::eA);
    vk::PipelineColorBlendStateCreateInfo blend{};
    blend.setAttachments(cb);
    vk::PipelineLayoutCreateInfo plci{};
    auto layout = vk.device->createPipelineLayoutUnique(plci);
    vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(cfmt);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(stages)
        .setPVertexInputState(&vin).setPInputAssemblyState(&ia)
        .setPViewportState(&vps).setPDynamicState(&dyn)
        .setPRasterizationState(&rs).setPMultisampleState(&ms)
        .setPColorBlendState(&blend).setLayout(layout.get())
        .setPNext(&rendering);
    auto pres = vk.device->createGraphicsPipelineUnique({}, gi);
    if (pres.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    auto pipe = std::move(pres.value);

    hl.render(vk::ImageLayout::eColorAttachmentOptimal,
              [&](vk::CommandBuffer c) {
                  vk::RenderingAttachmentInfo att{};
                  att.setImageView(hl.color.view.get())
                      .setImageLayout(
                          vk::ImageLayout::eColorAttachmentOptimal)
                      .setLoadOp(vk::AttachmentLoadOp::eClear)
                      .setStoreOp(vk::AttachmentStoreOp::eStore);
                  vk::RenderingInfo ri{};
                  ri.setRenderArea({{0, 0}, hl.extent})
                      .setLayerCount(1).setColorAttachments(att);
                  c.beginRendering(ri);
                  vk::Viewport v{0, 0, float(W), float(Hh), 0.f, 1.f};
                  c.setViewport(0, v);
                  vk::Rect2D sc{{0,0}, hl.extent}; c.setScissor(0, sc);
                  c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 pipe.get());
                  c.draw(3, 1, 0, 0);
                  c.endRendering();
              });
    hl.savePpm(ppm);
    printf("wrote %s\n", ppm);
    return 0;
}
