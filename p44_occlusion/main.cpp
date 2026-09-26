// SPDX-License-Identifier: CC0-1.0
//
// Occlusion queries: vkCmdBeginQuery(OCCLUSION) counts how many
// fragments actually PASS the depth test — the GPU-side answer to
// "did anyone see this object?". One query pool, one query per
// object, results read back on the host.
//
// Demo: blocker quad (near, depth-writes) + occluded quad (far,
// same pixels, fails depth) + a visible quad in the corner.
//   query[0] over the occluded draw -> 0 samples
//   query[1] over the visible draw  -> >0 samples
//
//   ./app --headless o.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* kVert = R"GLSL(
#version 460
layout(push_constant) uniform Push { vec2 ofs; float scl; float z; }
    pc;
void main() {
    vec2 p[6] = vec2[](vec2(-0.5,-0.5), vec2(0.5,-0.5),
                       vec2(0.5,0.5), vec2(-0.5,-0.5),
                       vec2(0.5,0.5),  vec2(-0.5,0.5));
    gl_Position = vec4(p[gl_VertexIndex] * pc.scl + pc.ofs,
                       pc.z, 1);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(push_constant) uniform Push { vec2 ofs; float scl; float z; }
    pc;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(pc.z, 0.5, 0.5, 1.0); }
)GLSL";

int main() {
    setbuf(stdout, nullptr);
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice({});

    // depth attachment for the headless target
    auto depth = vk.createImage(
        384, 384, vk::Format::eD32Sfloat,
        vk::ImageUsageFlagBits::eDepthStencilAttachment);
    vk::ImageSubresourceRange dsr{vk::ImageAspectFlagBits::eDepth,
                                  0, 1, 0, 1};

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
    vk::PipelineDepthStencilStateCreateInfo ds{};
    ds.setDepthTestEnable(VK_TRUE)
        .setDepthWriteEnable(VK_TRUE)
        .setDepthCompareOp(vk::CompareOp::eLess);
    vk::PipelineColorBlendAttachmentState cb{};
    cb.setColorWriteMask(
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    vk::PipelineColorBlendStateCreateInfo blend{};
    blend.setAttachments(cb);
    struct Push { float ofs[2]; float scl; float z; };
    vk::PushConstantRange pcr{};
    pcr.setStageFlags(vk::ShaderStageFlagBits::eVertex |
                      vk::ShaderStageFlagBits::eFragment)
        .setSize(sizeof(Push));
    vk::PipelineLayoutCreateInfo plci{};
    plci.setPushConstantRanges(pcr);
    auto layout = vk.device->createPipelineLayoutUnique(plci);
    vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
    vk::Format dfmt = vk::Format::eD32Sfloat;
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(cfmt)
        .setDepthAttachmentFormat(dfmt);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(stages)
        .setPVertexInputState(&vin)
        .setPInputAssemblyState(&ia)
        .setPViewportState(&vp)
        .setPDynamicState(&dyn)
        .setPRasterizationState(&rs)
        .setPMultisampleState(&ms)
        .setPDepthStencilState(&ds)
        .setPColorBlendState(&blend)
        .setLayout(layout.get())
        .setPNext(&rendering);
    auto pres = vk.device->createGraphicsPipelineUnique({}, gi);
    auto pipe = std::move(pres.value);

    // occlusion query pool: 2 slots
    vk::QueryPoolCreateInfo qpi{};
    qpi.setQueryType(vk::QueryType::eOcclusion).setQueryCount(2);
    auto qpool = vk.device->createQueryPoolUnique(qpi);

    vkmini::Headless hl;
    hl.init(vk, 384, 384);
    // hl.render transitions color to renderLayout; depth needs the
    // same treatment inside the lambda's first barrier:
    hl.render(vk::ImageLayout::eColorAttachmentOptimal,
              [&](vk::CommandBuffer c) {
                  vk::ImageMemoryBarrier db{};
                  db.setOldLayout(vk::ImageLayout::eUndefined)
                      .setNewLayout(
                          vk::ImageLayout::eDepthAttachmentOptimal)
                      .setSrcQueueFamilyIndex(
                          VK_QUEUE_FAMILY_IGNORED)
                      .setDstQueueFamilyIndex(
                          VK_QUEUE_FAMILY_IGNORED)
                      .setImage(depth.img.get())
                      .setSubresourceRange(dsr)
                      .setSrcAccessMask({})
                      .setDstAccessMask(
                          vk::AccessFlagBits::eDepthStencilAttachmentWrite);
                  c.pipelineBarrier(
                      vk::PipelineStageFlagBits::eTopOfPipe,
                      vk::PipelineStageFlagBits::
                          eEarlyFragmentTests,
                      {}, {}, {}, db);
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
                  vk::RenderingAttachmentInfo datt{};
                  datt.setImageView(depth.view.get())
                      .setImageLayout(
                          vk::ImageLayout::eDepthAttachmentOptimal)
                      .setLoadOp(vk::AttachmentLoadOp::eClear)
                      .setStoreOp(
                          vk::AttachmentStoreOp::eDontCare)
                      .setClearValue(vk::ClearValue{
                          vk::ClearDepthStencilValue{1.0f, 0}});
                  vk::RenderingInfo ri{};
                  ri.setRenderArea({{0, 0}, hl.extent})
                      .setLayerCount(1)
                      .setColorAttachments(att)
                      .setPDepthAttachment(&datt);
                  c.beginRendering(ri);
                  vk::Viewport v{0, 0, 384, 384, 0.f, 1.f};
                  c.setViewport(0, v);
                  vk::Rect2D sc{{0, 0}, hl.extent};
                  c.setScissor(0, sc);
                  c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 pipe.get());
                  c.resetQueryPool(qpool.get(), 0, 2);
                  auto quad = [&](float ox, float oy, float s,
                                  float z) {
                      Push p{{ox, oy}, s, z};
                      c.pushConstants(
                          layout.get(),
                          vk::ShaderStageFlagBits::eVertex |
                              vk::ShaderStageFlagBits::eFragment,
                          0, sizeof(Push), &p);
                      c.draw(6, 1, 0, 0);
                  };
                  // 1. blocker at z=0.2 covering the center
                  quad(0, 0, 0.6f, 0.2f);
                  // 2. query 0: same area, z=0.8 -> fully occluded
                  c.beginQuery(qpool.get(), 0, {});
                  quad(0, 0, 0.6f, 0.8f);
                  c.endQuery(qpool.get(), 0);
                  // 3. query 1: corner, z=0.1 -> visible
                  c.beginQuery(qpool.get(), 1, {});
                  quad(-0.7f, -0.7f, 0.2f, 0.1f);
                  c.endQuery(qpool.get(), 1);
                  c.endRendering();
              });
    uint64_t res[2];
    auto qr = vk.device->getQueryPoolResults(
        qpool.get(), 0, 2, sizeof(res), res, sizeof(uint64_t),
        vk::QueryResultFlagBits::e64);
    printf("query[0] occluded quad: %llu samples passed\n",
           (unsigned long long)res[0]);
    printf("query[1] visible quad : %llu samples passed\n",
           (unsigned long long)res[1]);
    hl.savePpm("out.ppm");
    return !(res[0] == 0 && res[1] > 0);
}
