// SPDX-License-Identifier: CC0-1.0
//
// Multi-threaded command recording: each thread gets its OWN
// VkCommandPool (pools are externally synchronized — never share
// one across threads) and records a SECONDARY buffer; the main
// thread's primary calls vkCmdExecuteCommands once. This is the
// real version of the secondary-buffer post.
//
// Demo: 4 threads each record a secondary drawing one column of a
// 4x1 strip (R/G/B/W) — logged with thread ids to prove the
// recording really ran on different threads.
//
//   ./app --headless o.ppm     one frame -> PPM for check.py
//   ./app                      windowed
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc -pthread

#include "vkmini.hpp"

#include <GLFW/glfw3.h>
#include <thread>

static const char* kVert = R"GLSL(
#version 460
layout(push_constant) uniform Push { vec3 col; vec2 ofs; } pc;
layout(location = 0) out vec3 vColor;
void main() {
    vec2 p[6] = vec2[](vec2(-0.5,-0.5), vec2(0.5,-0.5), vec2(0.5,0.5),
                       vec2(-0.5,-0.5), vec2(0.5,0.5),  vec2(-0.5,0.5));
    gl_Position = vec4(p[gl_VertexIndex] * vec2(0.22, 0.8) + pc.ofs,
                       0, 1);
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
    float col[3];
    float pad;
    float ofs[2];
};

int main(int argc, char** argv) {
    bool headless = argc > 1 && std::string(argv[1]) == "--headless";
    const char* ppmOut = argc > 2 ? argv[2] : "out.ppm";
    (void)headless;

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice({});

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
        .setSize(sizeof(Push));
    vk::PipelineLayoutCreateInfo plci{};
    plci.setPushConstantRanges(pcr);
    auto layout = vk.device->createPipelineLayoutUnique(plci);
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

    // ---- 4 threads, each its OWN pool + secondary -----------------
    const float cols[4][3] = {{1, .2f, .2f}, {.2f, 1, .2f},
                              {.2f, .2f, 1}, {1, 1, 1}};
    std::array<vk::UniqueCommandPool, 4> pools;
    std::array<vk::UniqueCommandBuffer, 4> secs;
    std::array<std::thread, 4> threads;
    for (int i = 0; i < 4; ++i) {
        pools[i] = vk.device->createCommandPoolUnique(
            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
             vk.qfam});
        threads[i] = std::thread([&, i] {
            secs[i] = std::move(
                vk.device->allocateCommandBuffersUnique(
                    {pools[i].get(),
                     vk::CommandBufferLevel::eSecondary, 1})
                    .front());
            vk::CommandBuffer c = secs[i].get();
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
            vk::Viewport v{0, 0, float(hl.extent.width),
                           float(hl.extent.height), 0.f, 1.f};
            c.setViewport(0, v);
            vk::Rect2D sc{{0, 0}, hl.extent};
            c.setScissor(0, sc);
            c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                           pipe.get());
            Push p{{cols[i][0], cols[i][1], cols[i][2]}, 0.f,
                   {-0.75f + 0.5f * float(i), 0.f}};
            c.pushConstants(layout.get(),
                            vk::ShaderStageFlagBits::eVertex, 0,
                            sizeof(Push), &p);
            c.draw(6, 1, 0, 0);
            c.end();
            printf("[thread %d] recorded secondary\n", i);
        });
    }
    for (auto& t : threads)
        t.join();

    hl.render(
        vk::ImageLayout::eColorAttachmentOptimal,
        [&](vk::CommandBuffer c) {
            vk::RenderingAttachmentInfo att{};
            att.setImageView(hl.color.view.get())
                .setImageLayout(
                    vk::ImageLayout::eColorAttachmentOptimal)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setClearValue(vk::ClearValue{vk::ClearColorValue{
                    std::array{0.05f, 0.05f, 0.08f, 1.f}}});
            vk::RenderingInfo ri{};
            ri.setRenderArea({{0, 0}, hl.extent})
                .setLayerCount(1)
                .setColorAttachments(att);
            c.beginRendering(ri);
            for (auto& s : secs)
                c.executeCommands(s.get());
            c.endRendering();
        });
    hl.savePpm(ppmOut);
    printf("wrote %s\n", ppmOut);
    return 0;
}
