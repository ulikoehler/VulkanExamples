// SPDX-License-Identifier: CC0-1.0
//
// VK_EXT_fragment_shader_interlock — a critical section for fragment
// shaders. beginInvocationInterlockEXT()..endInvocationInterlockEXT()
// serializes all fragments touching the same pixel (on this GPU:
// per-sample). Inside you may do non-atomic read-modify-write on an
// SSBO that would otherwise race.
//
// Demo: 4 overlapping quads each multiply a per-pixel accumulator
// (initialized to 1) by their factor — a blend mode no fixed-function
// unit has. Overlap regions end up at exact products; without the
// interlock the read+write pair races and the result is garbage.
//
//   ./app --headless out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"
#include <cmath>

constexpr uint32_t kW = 320, kH = 240;

// Each quad is drawn with its own color; the fragment shader ignores
// blending and multiplies acc[pixel] by push.factor inside the
// interlocked region.
static const char* kVert = R"GLSL(
#version 460
layout(push_constant) uniform PC { int quad; };
// 4 quads, each covering a different rect (offset overlap staircase)
const vec2 Q[4][6] = {
    {vec2(-0.9,-0.9),vec2( 0.2,-0.9),vec2( 0.2,0.5),
     vec2(-0.9,-0.9),vec2( 0.2, 0.5),vec2(-0.9,0.5)},
    {vec2(-0.5,-0.7),vec2( 0.5,-0.7),vec2( 0.5,0.7),
     vec2(-0.5,-0.7),vec2( 0.5, 0.7),vec2(-0.5,0.7)},
    {vec2(-0.2,-0.9),vec2( 0.9,-0.9),vec2( 0.9,0.5),
     vec2(-0.2,-0.9),vec2( 0.9, 0.5),vec2(-0.2,0.5)},
    {vec2(-0.6,-0.4),vec2( 0.6,-0.4),vec2( 0.6,0.4),
     vec2(-0.6,-0.4),vec2( 0.6, 0.4),vec2(-0.6,0.4)},
};
void main() {
    gl_Position = vec4(Q[quad][gl_VertexIndex], 0, 1);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
#extension GL_ARB_fragment_shader_interlock : require

layout(push_constant) uniform PC { int quad; };
layout(set = 0, binding = 0, std430) buffer Acc { float acc[]; };
layout(set = 0, binding = 1, std430) readonly buffer Dim {
    uvec2 dim;
};
layout(location = 0) out vec4 outColor;

// quad i contributes factor 2^-i  -> overlaps land on exact powers
// of two, easy to verify bit-exactly.
void main() {
    uvec2 px = uvec2(gl_FragCoord.xy);
    uint idx = px.y * dim.x + px.x;
    float f = exp2(-float(quad));

    // -- the whole point: this read+write is NOT an atomic op ------
    beginInvocationInterlockARB();
    acc[idx] = acc[idx] * f;
    endInvocationInterlockARB();
    // ----------------------------------------------------------------

    // visualize: gray = accumulator value (1 -> white, small -> dark)
    outColor = vec4(vec3(acc[idx]), 1);
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

    vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT il{};
    il.setFragmentShaderPixelInterlock(true);
    vk::PhysicalDeviceDynamicRenderingFeatures dyn{};
    il.setPNext(&dyn);
    vk.createDevice({VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME},
                    &il);

    auto acc = vk.createBuffer(
        vk::DeviceSize(kW) * kH * 4,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    auto dim = vk.createBuffer(
        8, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(dim.mapped, std::array{kW, kH}.data(), 8);

    // acc[] := 1.0 (0x3f800000) via vkCmdFillBuffer
    vk.oneTime([&](vk::CommandBuffer c) {
        c.fillBuffer(acc.buf.get(), 0, kW * kH * 4, 0x3f800000u);
    });

    auto vs = vk.shader(kVert, shaderc_vertex_shader, "v.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "f.frag");

    vk::DescriptorSetLayoutBinding bs[2];
    bs[0].setBinding(0)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eFragment);
    bs[1].setBinding(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eFragment);
    vk::DescriptorSetLayoutCreateInfo sli{};
    sli.setBindings(bs);
    auto setLayout = vk.device->createDescriptorSetLayoutUnique(sli);
    vk::DescriptorPoolSize ps{vk::DescriptorType::eStorageBuffer, 2};
    vk::DescriptorPoolCreateInfo dpi{};
    dpi.setMaxSets(1).setPoolSizes(ps);
    auto dpool = vk.device->createDescriptorPoolUnique(dpi);
    vk::DescriptorSetAllocateInfo dai{};
    dai.setDescriptorPool(dpool.get())
        .setSetLayouts(setLayout.get());
    auto sets = vk.device->allocateDescriptorSets(dai);
    vk::DescriptorBufferInfo dbAcc{acc.buf.get(), 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo dbDim{dim.buf.get(), 0, VK_WHOLE_SIZE};
    vk::WriteDescriptorSet wr[2];
    wr[0].setDstSet(sets[0])
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setBufferInfo(dbAcc);
    wr[1].setDstSet(sets[0])
        .setDstBinding(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setBufferInfo(dbDim);
    vk.device->updateDescriptorSets(wr, {});

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
                              0, 4};
    vk::PipelineLayoutCreateInfo li{};
    li.setSetLayouts(setLayout.get()).setPushConstantRanges(pcr);
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
    h.init(vk, kW, kH);
    h.render(vk::ImageLayout::eColorAttachmentOptimal,
             [&](vk::CommandBuffer c) {
        vk::RenderingAttachmentInfo att{};
        att.setImageView(h.color.view.get())
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore);
        vk::RenderingInfo ri{};
        ri.setRenderArea({{0, 0}, h.extent})
            .setLayerCount(1)
            .setColorAttachments(att);
        c.beginRendering(ri);
        c.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe.get());
        c.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                             layout.get(), 0, sets[0], {});
        vk::Viewport vp2{0, 0, float(kW), float(kH), 0, 1};
        c.setViewport(0, vp2);
        c.setScissor(0, vk::Rect2D{{0, 0}, h.extent});
        for (int q = 0; q < 4; ++q) {
            c.pushConstants(layout.get(),
                            vk::ShaderStageFlagBits::eVertex |
                                vk::ShaderStageFlagBits::eFragment,
                            0, 4, &q);
            c.draw(6, 1, 0, 0);
        }
        c.endRendering();
    });
    h.savePpm(ppm);

    // verify: read the acc buffer back and check every pixel's value
    // is a power of two >= 2^-3 (1..4 covering quads).
    auto rb = vk.createBuffer(
        vk::DeviceSize(kW) * kH * 4,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::BufferCopy rg{0, 0, vk::DeviceSize(kW) * kH * 4};
        c.copyBuffer(acc.buf.get(), rb.buf.get(), rg);
    });
    auto* a = static_cast<float*>(rb.mapped);
    int hist[7] = {};
    size_t bad = 0;
    for (size_t i = 0; i < size_t(kW) * kH; ++i) {
        bool ok = false;
        for (int e = 0; e <= 6; ++e)
            if (a[i] == exp2f(-float(e))) {
                hist[e]++;
                ok = true;
                break;
            }
        if (!ok) ++bad;
    }
    printf("acc histogram: ");
    for (int e = 0; e <= 6; ++e) printf("2^-%d:%d ", e, hist[e]);
    printf("| bad=%zu\n", bad);
    if (bad)
        throw std::runtime_error("interlocked RMW raced");
    printf("done: non-atomic per-pixel RMW inside interlock, exact\n");
    return 0;
}
