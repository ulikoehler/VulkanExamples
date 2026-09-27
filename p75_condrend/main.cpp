// SPDX-License-Identifier: CC0-1.0
//
// VK_EXT_conditional_rendering — the GPU decides whether a recorded
// draw executes at all. vkCmdBeginConditionalRenderingEXT points at a
// uint32 predicate in a buffer: 0 -> the enclosed commands are
// skipped by the GPU (predicated rendering, like D3D's
// SetPredication). The CPU records everything; the GPU culls.
//
// Demo: same quad drawn twice — once inside a predicate=1 region
// (visible) and once inside predicate=0 (skipped).
//
//   ./app --headless out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

static const char* kVert = R"GLSL(
#version 460
layout(push_constant) uniform PC { float xoff; int r, g, b; };
const vec2 p[3] = vec2[](vec2(0, -0.8), vec2(0.8, 0.8),
                         vec2(-0.8, 0.8));
void main() {
    vec2 v = p[gl_VertexIndex];
    gl_Position = vec4(v.x * 0.5 + xoff, v.y, 0, 1);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(push_constant) uniform PC { float xoff; int r, g, b; };
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(r, g, b, 1); }
)GLSL";

int main(int argc, char** argv) {
    const char* ppm = "out.ppm";
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--headless") && i + 1 < argc)
            ppm = argv[++i];

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    vk::PhysicalDeviceConditionalRenderingFeaturesEXT cr{};
    cr.setConditionalRendering(true);
    vk::PhysicalDeviceDynamicRenderingFeatures dyn{};
    cr.setPNext(&dyn);
    vk.createDevice(
        {VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME}, &cr);
    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr, vk.device.get());

    // predicate buffer: two words, [1, 0]
    auto pred = vk.createBuffer(
        8, vk::BufferUsageFlagBits::eConditionalRenderingEXT,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    static_cast<uint32_t*>(pred.mapped)[0] = 1;
    static_cast<uint32_t*>(pred.mapped)[1] = 0;

    auto vs = vk.shader(kVert, shaderc_vertex_shader, "v.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "f.frag");
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
    vk::PipelineDynamicStateCreateInfo dyn2{};
    dyn2.setDynamicStates(dynStates);
    vk::PipelineRasterizationStateCreateInfo rs{};
    rs.setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eNone)
        .setFrontFace(vk::FrontFace::eCounterClockwise)
        .setLineWidth(1.0f);
    vk::PipelineMultisampleStateCreateInfo ms{};
    ms.setRasterizationSamples(vk::SampleCountFlagBits::e1);
    vk::PipelineColorBlendAttachmentState cbAtt{};
    cbAtt.setColorWriteMask(
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    vk::PipelineColorBlendStateCreateInfo blend{};
    blend.setAttachments(cbAtt);
    vk::PushConstantRange pcr{vk::ShaderStageFlagBits::eVertex |
                                  vk::ShaderStageFlagBits::eFragment,
                              0, 16};
    vk::PipelineLayoutCreateInfo li{};
    li.setPushConstantRanges(pcr);
    auto layout = vk.device->createPipelineLayoutUnique(li);
    vk::PipelineRenderingCreateInfo rendering{};
    std::array cf{vk::Format::eR8G8B8A8Unorm};
    rendering.setColorAttachmentFormats(cf);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(stages)
        .setPVertexInputState(&vin)
        .setPInputAssemblyState(&ia)
        .setPViewportState(&vp)
        .setPDynamicState(&dyn2)
        .setPRasterizationState(&rs)
        .setPMultisampleState(&ms)
        .setPColorBlendState(&blend)
        .setLayout(layout.get())
        .setPNext(&rendering);
    auto pres = vk.device->createGraphicsPipelineUnique({}, gi);
    if (pres.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    auto pipe = std::move(pres.value);

    vkmini::Headless h;
    h.init(vk, 320, 240);
    h.render(vk::ImageLayout::eColorAttachmentOptimal,
             [&](vk::CommandBuffer c) {
        vk::RenderingAttachmentInfo att{};
        att.setImageView(h.color.view.get())
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setClearValue(vk::ClearValue{vk::ClearColorValue{
                std::array{0.f, 0.f, 0.f, 1.f}}});
        vk::RenderingInfo ri{};
        ri.setRenderArea({{0, 0}, h.extent})
            .setLayerCount(1)
            .setColorAttachments(att);
        c.beginRendering(ri);
        c.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe.get());
        vk::Viewport vpv{0, 0, float(h.extent.width),
                         float(h.extent.height), 0, 1};
        c.setViewport(0, vpv);
        c.setScissor(0, vk::Rect2D{{0, 0}, h.extent});

        // left triangle, predicate word 0 (=1) -> executes
        struct PC { float x; int r, g, b; } pc{-0.5f, 1, 0, 0};
        c.pushConstants(layout.get(),
                        vk::ShaderStageFlagBits::eVertex |
                            vk::ShaderStageFlagBits::eFragment,
                        0, sizeof pc, &pc);
        vk::ConditionalRenderingBeginInfoEXT bi{};
        bi.setBuffer(pred.buf.get()).setOffset(0);
        c.beginConditionalRenderingEXT(bi, dldi);
        c.draw(3, 1, 0, 0);
        c.endConditionalRenderingEXT(dldi);

        // right triangle, predicate word 1 (=0) -> GPU skips it
        pc = {0.5f, 0, 0, 1};
        c.pushConstants(layout.get(),
                        vk::ShaderStageFlagBits::eVertex |
                            vk::ShaderStageFlagBits::eFragment,
                        0, sizeof pc, &pc);
        bi.setOffset(4);
        c.beginConditionalRenderingEXT(bi, dldi);
        c.draw(3, 1, 0, 0);
        c.endConditionalRenderingEXT(dldi);
        c.endRendering();
    });
    h.savePpm(ppm);

    auto* px = static_cast<const uint8_t*>(h.readback.mapped);
    auto pxAt = [&](uint32_t x, uint32_t y) {
        const uint8_t* p = px + (y * h.extent.width + x) * 4;
        return std::tuple(p[0], p[1], p[2]);
    };
    auto [lr, lg, lb] = pxAt(80, 120);   // inside left triangle
    auto [rr, rg, rb] = pxAt(240, 120);  // inside right triangle
    printf("left=(%u,%u,%u) right=(%u,%u,%u)\n", lr, lg, lb, rr, rg,
           rb);
    if (!(lr > 200 && rr == 0 && rg == 0 && rb == 0))
        throw std::runtime_error("predicate ignored");
    printf("done: draw #2 skipped entirely by the GPU predicate\n");
    return 0;
}
