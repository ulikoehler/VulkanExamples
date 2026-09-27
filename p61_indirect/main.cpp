// SPDX-License-Identifier: CC0-1.0
//
// GPU-driven rendering: compute culls 256 candidate quads and writes
// BOTH the indirect command buffer AND the draw count — the CPU never
// knows how many draws happen. One vkCmdDrawIndirectCount consumes it.
//
//   ./app --headless out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

// ---------------- compute: cull + compact ------------------------
// Each workgroup item decides keep/drop by a simple hash (stable
// pattern ~ 3/8 of candidates) and appends a VkDrawIndirectCommand.
static const char* kComp = R"GLSL(
#version 460
layout(local_size_x = 64) in;

struct Cmd { uint vertexCount, instanceCount, firstVertex,
                  firstInstance; };
layout(std430, binding = 0) readonly buffer Src {
    uvec4 cand[];         // xy pos16.16, z size, w color id
};
layout(std430, binding = 1) buffer Dst {
    uint drawCount;       // atomic counter — GPU writes it!
    Cmd cmds[];
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= cand.length()) return;
    uvec4 c = cand[i];
    // "cull": keep candidates whose id hashes to ~3/8 of values
    uint h = (i * 2654435761u) >> 29;         // 0..7
    if (h > 2) return;                        // drop
    uint slot = atomicAdd(drawCount, 1u);
    cmds[slot] = Cmd(6, 1, 0, i);            // quad via gl_VertexIndex
}
)GLSL";

// ---------------- graphics ---------------------------------------
static const char* kVert = R"GLSL(
#version 460
layout(std430, binding = 0) readonly buffer Src { uvec4 cand[]; };
layout(location = 0) out vec3 vCol;
const vec2 kQuad[6] = vec2[](
    vec2(-1,-1), vec2(1,-1), vec2(1,1),
    vec2(-1,-1), vec2(1,1),  vec2(-1,1));
void main() {
    uvec4 c = cand[gl_InstanceIndex];
    vec2 pos = vec2(c.xy) / 65536.0 * 2.0 - 1.0;
    float s = float(c.z) / 65536.0;
    gl_Position = vec4(pos + kQuad[gl_VertexIndex] * s, 0, 1);
    uint id = c.w;
    vCol = vec3((id&1), (id>>1&1), (id>>2&1)) * 0.75 + 0.25;
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec3 vCol;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(vCol, 1); }
)GLSL";

int main(int argc, char** argv) {
    const char* ppm = "out.ppm";
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--headless") && i + 1 < argc)
            ppm = argv[++i];

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    // KHR ext not strictly needed — drawIndirectCount is core 1.2.
    vk.createDevice({VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME});

    constexpr uint32_t W = 640, Hh = 360, N = 256;

    // candidate data on CPU — positions/sizes/colors
    std::vector<uint32_t> cand(N * 4);
    for (uint32_t i = 0; i < N; ++i) {
        // deterministic pseudo-random grid scatter
        uint32_t gx = i % 16, gy = i / 16;
        cand[i*4+0] = uint32_t((gx + 0.5f) / 16.f * 65536.f);
        cand[i*4+1] = uint32_t((gy + 0.5f) / 16.f * 65536.f);
        cand[i*4+2] = uint32_t(0.045f * 65536.f);
        cand[i*4+3] = i % 8;
    }
    auto srcBuf = vk.createBuffer(
        cand.size() * 4,
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);
    memcpy(srcBuf.mapped, cand.data(), cand.size() * 4);

    // indirect buffer: count(4B) + N VkDrawIndirectCommand (16B each)
    auto indBuf = vk.createBuffer(
        4 + N * 16,
        vk::BufferUsageFlagBits::eIndirectBuffer |
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);
    memset(indBuf.mapped, 0, 4);

    // descriptor set shared by compute + vertex shader (binding 0
    // = candidates SSBO, binding 1 = indirect buffer as SSBO)
    std::array bindings{
        vk::DescriptorSetLayoutBinding{
            0, vk::DescriptorType::eStorageBuffer, 1,
            vk::ShaderStageFlagBits::eCompute |
            vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{
            1, vk::DescriptorType::eStorageBuffer, 1,
            vk::ShaderStageFlagBits::eCompute}};
    vk::DescriptorSetLayoutCreateInfo dslci{};
    dslci.setBindings(bindings);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dslci);
    vk::DescriptorPoolSize ps{vk::DescriptorType::eStorageBuffer, 2};
    auto dpool = vk.device->createDescriptorPoolUnique(
        {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1, ps});
    vk::DescriptorSetLayout dslH = dsl.get();
    vk::DescriptorSetAllocateInfo dsai{};
    dsai.setDescriptorPool(dpool.get()).setSetLayouts(dslH);
    auto dsets = vk.device->allocateDescriptorSetsUnique(dsai);
    std::array dbi{vk::DescriptorBufferInfo{srcBuf.buf.get(), 0,
                                           cand.size() * 4},
                   vk::DescriptorBufferInfo{indBuf.buf.get(), 0,
                                            4 + N * 16}};
    std::array wr{
        vk::WriteDescriptorSet{dsets[0].get(), 0, 0,
                               vk::DescriptorType::eStorageBuffer,
                               {}, dbi[0]},
        vk::WriteDescriptorSet{dsets[0].get(), 1, 0,
                               vk::DescriptorType::eStorageBuffer,
                               {}, dbi[1]}};
    vk.device->updateDescriptorSets(wr, {});

    // -------- compute pipeline ------------------------------------
    auto cs = vk.shader(kComp, shaderc_compute_shader, "cull.comp");
    vk::ComputePipelineCreateInfo cpci{};
    vk::PipelineLayoutCreateInfo plci{};
    plci.setSetLayouts(dslH);
    auto layout = vk.device->createPipelineLayoutUnique(plci);
    vk::PipelineShaderStageCreateInfo cstage{};
    cstage.setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(cs.get()).setPName("main");
    cpci.setStage(cstage).setLayout(layout.get());
    auto cres = vk.device->createComputePipelineUnique({}, cpci);
    if (cres.result != vk::Result::eSuccess)
        throw std::runtime_error("compute pipeline failed");
    auto cpipe = std::move(cres.value);

    // -------- graphics pipeline -----------------------------------
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "s.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "s.frag");
    vk::PipelineShaderStageCreateInfo gst[2];
    gst[0].setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(vs.get()).setPName("main");
    gst[1].setStage(vk::ShaderStageFlagBits::eFragment)
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
    vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(cfmt);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(gst)
        .setPVertexInputState(&vin).setPInputAssemblyState(&ia)
        .setPViewportState(&vps).setPDynamicState(&dyn)
        .setPRasterizationState(&rs).setPMultisampleState(&ms)
        .setPColorBlendState(&blend).setLayout(layout.get())
        .setPNext(&rendering);
    auto gres = vk.device->createGraphicsPipelineUnique({}, gi);
    if (gres.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    auto gpipe = std::move(gres.value);

    vkmini::Headless hl;
    hl.init(vk, W, Hh);

    // phase 1: compute culling — writes drawCount + command array.
    // oneTime submits + waitIdle => full sync before the draw pass.
    vk.oneTime([&](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, cpipe.get());
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             layout.get(), 0, dsets[0].get(), {});
        c.dispatch(N / 64, 1, 1);
    });
    printf("GPU wrote drawCount=%u (of %u candidates)\n",
           *static_cast<uint32_t*>(indBuf.mapped), N);

    // phase 2: render — CPU doesn't decide how many draws happen
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
                                 gpipe.get());
                  c.bindDescriptorSets(
                      vk::PipelineBindPoint::eGraphics, layout.get(),
                      0, dsets[0].get(), {});
                  // indirect buf@4 = VkDrawIndirectCommand array,
                  // count buf@0 = uint32 written by the shader
                  c.drawIndirectCount(indBuf.buf.get(), 4,
                                      indBuf.buf.get(), 0, N, 16);
                  c.endRendering();
              });
    hl.savePpm(ppm);
    printf("wrote %s\n", ppm);
    return 0;
}
