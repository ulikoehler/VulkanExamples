// SPDX-License-Identifier: CC0-1.0
//
// GPU-side profiling with core Vulkan timestamp queries:
// vkCmdWriteTimestamp drops a GPU-clock sample into a query pool at
// command-buffer granularity; vkGetQueryPoolResults reads them back.
// Bracket a draw/dispatch and you know its GPU time — no external
// profiler needed. VK_KHR_calibrated_timestamps maps the GPU clock
// onto the CPU clock for end-to-end latency.
//
//   ./app --headless out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc

#include "vkmini.hpp"
#include <chrono>

static const char* kVert = R"GLSL(
#version 460
void main() {
    vec2 p = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0,
                  (gl_VertexIndex == 2) ? 3.0 : -1.0);
    gl_Position = vec4(p, 0, 1);
}
)GLSL";
// deliberately not-trivial fragment work so the delta is measurable
static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) out vec4 outColor;
void main() {
    vec2 uv = gl_FragCoord.xy / 1024.0;
    float acc = 0.0;
    for (int i = 0; i < 400; ++i)
        acc += sin(uv.x * i * 0.13) * cos(uv.y * i * 0.07);
    outColor = vec4(fract(abs(acc)), uv, 1);
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
    vk.createDevice();

    // timestamp support lives in queue-family props, not a feature
    auto props = vk.phys.getQueueFamilyProperties();
    printf("queue family %u: timestampValidBits=%u\n", vk.qfam,
           props[vk.qfam].timestampValidBits);
    if (!props[vk.qfam].timestampValidBits) {
        printf("SKIP: queue family cannot timestamp\n");
        return 0;
    }
    float period =
        vk.phys.getProperties().limits.timestampPeriod; // ns/tick
    printf("timestampPeriod: %.1f ns\n", period);

    auto qp = vk.device->createQueryPoolUnique(
        {{}, vk::QueryType::eTimestamp, 2});

    // minimal pipeline
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "t.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "t.frag");
    vk::PipelineShaderStageCreateInfo stg[2];
    stg[0].setStage(vk::ShaderStageFlagBits::eVertex).setModule(vs.get())
        .setPName("main");
    stg[1].setStage(vk::ShaderStageFlagBits::eFragment).setModule(fs.get())
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
    vk::PipelineLayoutCreateInfo plci{};
    auto layout = vk.device->createPipelineLayoutUnique(plci);
    vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
    vk::PipelineRenderingCreateInfo pr{};
    pr.setColorAttachmentFormats(cfmt);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(stg).setPVertexInputState(&vin)
      .setPInputAssemblyState(&ia).setPViewportState(&vps)
      .setPDynamicState(&dsi).setPRasterizationState(&rs)
      .setPMultisampleState(&ms).setPColorBlendState(&blend)
      .setLayout(layout.get()).setPNext(&pr);
    auto pipe = std::move(
        vk.device->createGraphicsPipelineUnique({}, gi).value);

    vkmini::Headless hl;
    hl.init(vk, 1024, 1024);

    // warm up once (pipeline/jit noise must not land in the measure)
    hl.render(vk::ImageLayout::eColorAttachmentOptimal,
              [&](vk::CommandBuffer c) {
        vk::RenderingAttachmentInfo att{};
        att.setImageView(hl.color.view.get())
           .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
           .setLoadOp(vk::AttachmentLoadOp::eClear)
           .setStoreOp(vk::AttachmentStoreOp::eStore);
        vk::RenderingInfo ri{};
        ri.setRenderArea({{0,0},hl.extent}).setLayerCount(1)
          .setColorAttachments(att);
        c.beginRendering(ri);
        vk::Viewport vp{0,0,float(hl.extent.width),
                        float(hl.extent.height),0,1};
        vk::Rect2D sc{{0,0},hl.extent};
        c.setViewport(0, vp); c.setScissor(0, sc);
        c.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe.get());
        c.draw(3, 1, 0, 0);
        c.endRendering();
    });

    // ---- measured frame: timestamps bracket the draw --------------
    auto cmds = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::ePrimary, 1});
    auto& c = cmds[0];
    c->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    c->resetQueryPool(qp.get(), 0, 2);
    vk::ImageMemoryBarrier b{};
    b.setOldLayout(vk::ImageLayout::eUndefined)
     .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
     .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
     .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
     .setImage(hl.color.img.get())
     .setSubresourceRange({vk::ImageAspectFlagBits::eColor,0,1,0,1})
     .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
    c->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, {}, {}, b);

    c->writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe,
                      qp.get(), 0);

    vk::RenderingAttachmentInfo att{};
    att.setImageView(hl.color.view.get())
       .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
       .setLoadOp(vk::AttachmentLoadOp::eClear)
       .setStoreOp(vk::AttachmentStoreOp::eStore);
    vk::RenderingInfo ri{};
    ri.setRenderArea({{0,0},hl.extent}).setLayerCount(1)
      .setColorAttachments(att);
    c->beginRendering(ri);
    vk::Viewport vp{0,0,float(hl.extent.width),
                    float(hl.extent.height),0,1};
    vk::Rect2D sc{{0,0},hl.extent};
    c->setViewport(0, vp); c->setScissor(0, sc);
    c->bindPipeline(vk::PipelineBindPoint::eGraphics, pipe.get());
    c->draw(3, 1, 0, 0);
    c->endRendering();

    c->writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
                      qp.get(), 1);
    c->end();

    auto t0 = std::chrono::steady_clock::now();
    vk::SubmitInfo si{}; si.setCommandBuffers(c.get());
    vk.queue.submit(si);
    vk.queue.waitIdle();
    auto t1 = std::chrono::steady_clock::now();

    uint64_t ts[2];
    auto qr = vk.device->getQueryPoolResults(
        qp.get(), 0, 2, sizeof(ts), ts, 8,
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (qr != vk::Result::eSuccess) {
        printf("SKIP: timestamp query failed\n"); return 0;
    }

    double gpuUs = (ts[1] - ts[0]) * period / 1000.0;
    double cpuUs = std::chrono::duration<double, std::micro>(
                       t1 - t0).count();
    printf("draw: %.1f us GPU  (%.1f us submit->done incl. "
           "CPU overhead)\n", gpuUs, cpuUs);

    // GPU<->CPU clock correspondence, where the driver supports it
    for (auto& e : vk.phys.enumerateDeviceExtensionProperties())
        if (std::string(e.extensionName.data()) ==
            "VK_KHR_calibrated_timestamps")
            printf("VK_KHR_calibrated_timestamps present — map GPU "
                   "ticks to CLOCK_MONOTONIC for latency work\n");

    hl.render(vk::ImageLayout::eColorAttachmentOptimal,
              [](vk::CommandBuffer){}); // satisfy savePpm's layout
    hl.savePpm(ppm);
    printf("wrote %s\n", ppm);
    return 0;
}
