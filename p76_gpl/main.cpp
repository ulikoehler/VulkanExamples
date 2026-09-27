// SPDX-License-Identifier: CC0-1.0
//
// VK_EXT_graphics_pipeline_library — a graphics pipeline compiled in
// four pieces. Each "library" pipeline bakes in only its slice of
// state; vkCreateGraphicsPipelines with VkPipelineLibraryCreateInfoKHR
// links them into one executable pipeline. Shader variants multiply
// cheaply: swap the fragment-shader library, keep the other three.
//
// Layout of the split (per the spec):
//   VertexInputInterface   vertex input + input assembly
//   PreRasterization       vertex(+tess+geom) shaders, viewport,
//                          rasterizer state
//   FragmentShader         fragment shader
//   FragmentOutput         blend state + attachment formats
//
// Demo: two full pipelines linked from the same three libraries +
// two different fragment-shader libraries (red / blue).
//
//   ./app --headless out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"
#include <chrono>
#include <functional>

static const char* kVert = R"GLSL(
#version 460
const vec2 p[6] = vec2[](vec2(-0.9,-0.9), vec2(0.9,-0.9),
                         vec2(0.9, 0.9), vec2(-0.9,-0.9),
                         vec2(0.9, 0.9), vec2(-0.9,0.9));
void main() { gl_Position = vec4(p[gl_VertexIndex], 0, 1); }
)GLSL";

static const char* kFragRed = R"GLSL(
#version 460
layout(location = 0) out vec4 o;
void main() { o = vec4(1, 0, 0, 1); }
)GLSL";

static const char* kFragBlue = R"GLSL(
#version 460
layout(location = 0) out vec4 o;
void main() { o = vec4(0, 0.2, 1, 1); }
)GLSL";

int main(int argc, char** argv) {
    const char* ppm = "out.ppm";
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--headless") && i + 1 < argc)
            ppm = argv[++i];

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl{};
    gpl.setGraphicsPipelineLibrary(true);
    vk::PhysicalDeviceDynamicRenderingFeatures dyn{};
    gpl.setPNext(&dyn);
    vk.createDevice(
        {VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME}, &gpl);

    auto layout = vk.device->createPipelineLayoutUnique(
        vk::PipelineLayoutCreateInfo{});
    vk::PipelineRenderingCreateInfo rendering{};
    std::array cf{vk::Format::eR8G8B8A8Unorm};
    rendering.setColorAttachmentFormats(cf);

    // helper: create a library pipeline for one slice
    auto mkLib = [&](vk::GraphicsPipelineLibraryFlagsEXT flags,
                     std::function<void(vk::GraphicsPipelineCreateInfo&)>
                         fill) {
        vk::GraphicsPipelineLibraryCreateInfoEXT lib{};
        lib.setFlags(flags);
        vk::GraphicsPipelineCreateInfo gi{};
        gi.setFlags(vk::PipelineCreateFlagBits::eLibraryKHR |
                    vk::PipelineCreateFlagBits::
                        eRetainLinkTimeOptimizationInfoEXT)
            .setLayout(layout.get())
            .setPNext(&lib);
        lib.setPNext(&rendering);
        fill(gi);
        auto r = vk.device->createGraphicsPipelineUnique({}, gi);
        if (r.result != vk::Result::eSuccess)
            throw std::runtime_error("library pipeline failed");
        return std::move(r.value);
    };

    // --- 1. vertex input interface ---------------------------------
    vk::PipelineVertexInputStateCreateInfo vin{};
    vk::PipelineInputAssemblyStateCreateInfo ia{};
    ia.setTopology(vk::PrimitiveTopology::eTriangleList);
    auto libVin = mkLib(vk::GraphicsPipelineLibraryFlagBitsEXT::
                            eVertexInputInterface,
                        [&](vk::GraphicsPipelineCreateInfo& g) {
                            g.setPVertexInputState(&vin)
                                .setPInputAssemblyState(&ia);
                        });

    // --- 2. pre-rasterization (vs module + raster + viewport) ------
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "v.vert");
    vk::PipelineShaderStageCreateInfo vsStage{};
    vsStage.setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(vs.get())
        .setPName("main");
    vk::PipelineViewportStateCreateInfo vp{};
    vp.setViewportCount(1).setScissorCount(1);
    std::array dynStates{vk::DynamicState::eViewport,
                         vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynS{};
    dynS.setDynamicStates(dynStates);
    vk::PipelineRasterizationStateCreateInfo rs{};
    rs.setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eNone)
        .setFrontFace(vk::FrontFace::eCounterClockwise)
        .setLineWidth(1.0f);
    auto libPre = mkLib(vk::GraphicsPipelineLibraryFlagBitsEXT::
                            ePreRasterizationShaders,
                        [&](vk::GraphicsPipelineCreateInfo& g) {
                            g.setStages(vsStage)
                                .setPViewportState(&vp)
                                .setPDynamicState(&dynS)
                                .setPRasterizationState(&rs);
                        });

    // --- 3. fragment shader libraries (two variants) ---------------
    auto mkFragLib = [&](const char* src, const char* name) {
        auto fs = vk.shader(src, shaderc_fragment_shader, name);
        vk::PipelineShaderStageCreateInfo st{};
        st.setStage(vk::ShaderStageFlagBits::eFragment)
            .setModule(fs.get())
            .setPName("main");
        vk::PipelineMultisampleStateCreateInfo ms{};
        ms.setRasterizationSamples(vk::SampleCountFlagBits::e1);
        return mkLib(vk::GraphicsPipelineLibraryFlagBitsEXT::
                         eFragmentShader,
                     [&](vk::GraphicsPipelineCreateInfo& g) {
                         g.setStages(st).setPMultisampleState(&ms);
                     });
    };
    auto libFragR = mkFragLib(kFragRed, "red.frag");
    auto libFragB = mkFragLib(kFragBlue, "blue.frag");

    // --- 4. fragment output interface (blending + formats) ---------
    vk::PipelineColorBlendAttachmentState cbAtt{};
    cbAtt.setColorWriteMask(
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    vk::PipelineColorBlendStateCreateInfo blend{};
    blend.setAttachments(cbAtt);
    vk::PipelineMultisampleStateCreateInfo msO{};
    msO.setRasterizationSamples(vk::SampleCountFlagBits::e1);
    auto libFO = mkLib(vk::GraphicsPipelineLibraryFlagBitsEXT::
                           eFragmentOutputInterface,
                       [&](vk::GraphicsPipelineCreateInfo& g) {
                           g.setPColorBlendState(&blend)
                               .setPMultisampleState(&msO);
                       });

    // --- link: full pipeline = libVin + libPre + libFrag? + libFO --
    auto link = [&](vk::Pipeline fragLib) {
        std::array<vk::Pipeline, 4> libs{libVin.get(), libPre.get(),
                                         fragLib, libFO.get()};
        vk::PipelineLibraryCreateInfoKHR lib{};
        lib.setLibraries(libs);
        vk::GraphicsPipelineCreateInfo gi{};
        gi.setLayout(layout.get())
            .setFlags(vk::PipelineCreateFlagBits::
                          eLinkTimeOptimizationEXT)
            .setPNext(&lib);
        lib.setPNext(&rendering);
        auto t0 = std::chrono::steady_clock::now();
        auto r = vk.device->createGraphicsPipelineUnique({}, gi);
        double us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
        printf("link time: %.0f us\n", us);
        if (r.result != vk::Result::eSuccess)
            throw std::runtime_error("link failed");
        return std::move(r.value);
    };
    auto pipeRed = link(libFragR.get());
    auto pipeBlue = link(libFragB.get());

    // draw: red quad left, blue quad right (two scissored draws)
    vkmini::Headless h;
    h.init(vk, 320, 240);
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
        vk::Viewport vpv{0, 0, float(h.extent.width),
                         float(h.extent.height), 0, 1};
        c.setViewport(0, vpv);
        c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                       pipeRed.get());
        c.setScissor(0, vk::Rect2D{{0, 0}, {160, 240}});
        c.draw(6, 1, 0, 0);
        c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                       pipeBlue.get());
        c.setScissor(0, vk::Rect2D{{160, 0}, {160, 240}});
        c.draw(6, 1, 0, 0);
        c.endRendering();
    });
    h.savePpm(ppm);

    auto* px = static_cast<const uint8_t*>(h.readback.mapped);
    const uint8_t* L = px + (120 * 320 + 80) * 4;
    const uint8_t* R = px + (120 * 320 + 240) * 4;
    printf("left=(%u,%u,%u) right=(%u,%u,%u)\n", L[0], L[1], L[2],
           R[0], R[1], R[2]);
    if (!(L[0] > 200 && R[2] > 200))
        throw std::runtime_error("linked pipelines produced wrong px");
    printf("done: 4 libraries -> 2 linked pipelines, both render\n");
    return 0;
}
