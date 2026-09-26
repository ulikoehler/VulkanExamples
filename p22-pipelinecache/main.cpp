// SPDX-License-Identifier: CC0-1.0
//
// Pipeline cache: vkCreateGraphicsPipelines is expensive — driver
// compiles SPIR-V to native ISA and merges fixed-function state.
// VkPipelineCache lets you persist that work across runs:
// vkGetPipelineCacheData -> file -> VkPipelineCacheCreateInfo.
//
// First run:  builds pipelines cold, writes pipeline.cache
// Second run: loads the blob, pipelines hit the cache
//
//   ./app --headless o.ppm     one frame -> PPM
//   python3 check.py           runs it twice, verifies cache reuse
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) out vec3 vColor;
void main() {
    vec2 p[3] = vec2[](vec2(-0.8, 0.7), vec2(0.8, 0.7),
                       vec2(0.0, -0.8));
    vec3 c[3] = vec3[](vec3(1, 0, 0), vec3(0, 1, 0),
                       vec3(0, 0, 1));
    gl_Position = vec4(p[gl_VertexIndex], 0.0, 1.0);
    vColor = c[gl_VertexIndex];
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(vColor, 1.0); }
)GLSL";

/// Load an existing cache blob if present, else start empty.
static vk::UniquePipelineCache makeCache(vkmini::Vk& vk,
                                         const char* path) {
    std::vector<char> blob;
    if (std::ifstream f{path, std::ios::binary | std::ios::ate}) {
        auto n = f.tellg();
        blob.resize(size_t(n));
        f.seekg(0);
        f.read(blob.data(), n);
    }
    vk::PipelineCacheCreateInfo ci{};
    if (!blob.empty()) {
        ci.setInitialDataSize(blob.size())
            .setPInitialData(blob.data());
        printf("[cache] loaded %zu bytes from %s\n", blob.size(),
               path);
    } else {
        printf("[cache] starting cold\n");
    }
    return vk.device->createPipelineCacheUnique(ci);
}

/// Persist whatever the driver accumulated in the cache.
static void saveCache(vkmini::Vk& vk, vk::PipelineCache cache,
                      const char* path) {
    auto data = vk.device->getPipelineCacheData(cache);
    std::ofstream f{path, std::ios::binary};
    f.write(reinterpret_cast<const char*>(data.data()),
            std::streamsize(data.size()));
    printf("[cache] saved %zu bytes to %s\n", data.size(), path);
}

int main() {
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice({});

    auto cache = makeCache(vk, "pipeline.cache");

    auto vs = vk.shader(kVert, shaderc_vertex_shader, "t.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "t.frag");
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
    vk::PipelineRenderingCreateInfo rendering{};
    vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
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
    // createGraphicsPipelineUnique takes the cache as first arg —
    // hits avoid the ISA compile inside the driver
    auto res = vk.device->createGraphicsPipelineUnique(
        cache.get(), gi);
    if (res.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    auto pipe = std::move(res.value);

    // render one frame headless to prove the cached pipeline works
    vkmini::Headless hl;
    hl.init(vk, 256, 256);
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
                  c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 pipe.get());
                  vk::Viewport vp{0, 0, 256, 256, 0.f, 1.f};
                  c.setViewport(0, vp);
                  vk::Rect2D sc{{0, 0}, hl.extent};
                  c.setScissor(0, sc);
                  c.draw(3, 1, 0, 0);
                  c.endRendering();
              });
    hl.savePpm("out.ppm");
    printf("wrote out.ppm\n");

    saveCache(vk, cache.get(), "pipeline.cache");
    return 0;
}
