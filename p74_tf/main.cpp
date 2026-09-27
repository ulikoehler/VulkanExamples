// SPDX-License-Identifier: CC0-1.0
//
// VK_EXT_transform_feedback — capture vertex shader outputs into a
// buffer while drawing. The GPU-loop pattern for particle systems
// that predate compute shaders: rasterizerDiscard=true means nothing
// is rasterized; the vertex shader is a pure per-vertex kernel and
// vkCmdBeginTransformFeedbackEXT writes the varyings out.
//
// Demo: N particles, pos += vel*dt entirely in the vertex stage.
// CPU verifies the exact result after the draw.
//
//   ./app
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"
#include <cmath>

constexpr uint32_t kN = 4096;
constexpr float kDt = 0.25f;

// The xfb_buffer/xfb_offset/xfb_stride layout qualifiers ARE the
// capture stream description — glslang turns them into SPIR-V
// XfbBuffer/XfbStride/Offset decorations, no pNext on the pipeline.
static const char* kVert = R"GLSL(
#version 460

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 vel;

layout(location = 0, xfb_buffer = 0, xfb_offset = 0,
       xfb_stride = 16) out vec2 outPos;
layout(location = 1, xfb_buffer = 0, xfb_offset = 8,
       xfb_stride = 16) out vec2 outVel;

void main() {
    outPos = pos + vel * 0.25;   // dt baked in, exact in fp
    outVel = vel;
    gl_Position = vec4(outPos, 0, 1);   // unused: rasterizerDiscard
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) out vec4 o;
void main() { o = vec4(1); }
)GLSL";

int main() {
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    vk::PhysicalDeviceTransformFeedbackFeaturesEXT tf{};
    tf.setTransformFeedback(true);
    vk::PhysicalDeviceDynamicRenderingFeatures dyn{};
    tf.setPNext(&dyn);
    vk.createDevice({VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME}, &tf);
    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr, vk.device.get());

    // ---- input: pos.xy, vel.xy, 16B stride ------------------------
    auto inBuf = vk.createBuffer(
        vk::DeviceSize(kN) * 16,
        vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    auto* in = static_cast<float*>(inBuf.mapped);
    for (uint32_t i = 0; i < kN; ++i) {
        in[i * 4 + 0] = float(i) * 0.5f;
        in[i * 4 + 1] = float(i) * 0.25f;
        in[i * 4 + 2] = 1.0f;             // vx
        in[i * 4 + 3] = -2.0f;            // vy
    }
    // ---- output: same layout, transform-feedback target -----------
    auto outBuf = vk.createBuffer(
        vk::DeviceSize(kN) * 16,
        vk::BufferUsageFlagBits::eTransformFeedbackBufferEXT |
            vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);

    auto vs = vk.shader(kVert, shaderc_vertex_shader, "tf.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "tf.frag");

    vk::PipelineShaderStageCreateInfo stages[2];
    stages[0].setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(vs.get())
        .setPName("main");
    stages[1].setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(fs.get())
        .setPName("main");

    // vertex input: 2 x vec2 at stride 16
    vk::VertexInputBindingDescription bind{0, 16,
                                           vk::VertexInputRate::
                                               eVertex};
    vk::VertexInputAttributeDescription attrs[2];
    attrs[0] = {0, 0, vk::Format::eR32G32Sfloat, 0};
    attrs[1] = {1, 0, vk::Format::eR32G32Sfloat, 8};
    vk::PipelineVertexInputStateCreateInfo vin{};
    vin.setVertexBindingDescriptions(bind)
        .setVertexAttributeDescriptions(attrs);
    vk::PipelineInputAssemblyStateCreateInfo ia{};
    ia.setTopology(vk::PrimitiveTopology::ePointList);
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
        .setLineWidth(1.0f)
        .setRasterizerDiscardEnable(true); // <- pure vertex kernel
    vk::PipelineMultisampleStateCreateInfo ms{};
    ms.setRasterizationSamples(vk::SampleCountFlagBits::e1);
    vk::PipelineColorBlendStateCreateInfo blend{};
    vk::PipelineLayoutCreateInfo li{};
    auto layout = vk.device->createPipelineLayoutUnique(li);
    vk::PipelineRenderingCreateInfo rendering{}; // 0 attachments
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

    vk.oneTime([&](vk::CommandBuffer c) {
        vk::RenderingInfo ri{};
        ri.setRenderArea({{0, 0}, {64, 64}}).setLayerCount(1);
        c.beginRendering(ri);
        c.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe.get());
        vk::DeviceSize ofs = 0;
        c.bindVertexBuffers(0, inBuf.buf.get(), ofs);
        // THE extension calls: bind capture buffer + bracket the draw
        vk::DeviceSize sz = vk::DeviceSize(kN) * 16;
        c.bindTransformFeedbackBuffersEXT(0, outBuf.buf.get(), ofs,
                                          sz, dldi);
        c.beginTransformFeedbackEXT(0, {}, {}, dldi);
        c.draw(kN, 1, 0, 0);
        c.endTransformFeedbackEXT(0, {}, {}, dldi);
        c.endRendering();
    });

    auto* o = static_cast<const float*>(outBuf.mapped);
    size_t bad = 0;
    for (uint32_t i = 0; i < kN; ++i) {
        float ex = in[i * 4 + 0] + in[i * 4 + 2] * kDt;
        float ey = in[i * 4 + 1] + in[i * 4 + 3] * kDt;
        if (o[i * 4 + 0] != ex || o[i * 4 + 1] != ey ||
            o[i * 4 + 2] != in[i * 4 + 2] || o[i * 4 + 3] != in[i * 4 + 3])
            ++bad;
    }
    printf("%u particles stepped on vertex stage -> TF buffer, "
           "bad=%zu\n", kN, bad);
    if (bad)
        throw std::runtime_error("transform feedback mismatch");
    printf("done: pos' = pos + vel*dt captured without rasterization\n");
    return 0;
}
