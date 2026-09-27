// SPDX-License-Identifier: CC0-1.0
//
// VK_KHR_pipeline_executable_properties — ask the driver what it
// REALLY compiled: per-stage instruction stats and (on RADV/AMDVLK)
// the actual ISA disassembly. This is the API RenderDoc/NSight use
// for their "shader stats" panels.
//
//   ./app        prints executable stats + first ISA bytes
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() {
    float a = sin(vUV.x * 12.9) * cos(vUV.y * 78.2);
    outColor = vec4(a * 0.5 + 0.5, vUV, 1);
}
)GLSL";

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

int main() {
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR pef{};
    pef.setPipelineExecutableInfo(true);
    vk.createDevice(
        {VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME},
        &pef);
    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr, vk.device.get());

    // a minimal graphics pipeline to dissect
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
    auto layout = vk.device->createPipelineLayoutUnique({});
    vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(cfmt);
    vk::GraphicsPipelineCreateInfo gi{};
    // without this flag the driver may not retain IR for the query
    gi.setFlags(vk::PipelineCreateFlagBits::eCaptureInternalRepresentationsKHR)
        .setStages(stages)
        .setPVertexInputState(&vin).setPInputAssemblyState(&ia)
        .setPViewportState(&vps).setPDynamicState(&dyn)
        .setPRasterizationState(&rs).setPMultisampleState(&ms)
        .setPColorBlendState(&blend).setLayout(layout.get())
        .setPNext(&rendering);
    auto pres = vk.device->createGraphicsPipelineUnique({}, gi);
    if (pres.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    auto pipe = std::move(pres.value);

    // ---- the interesting part ------------------------------------
    vk::PipelineInfoKHR pi{pipe.get()};
    auto execs = vk.device->getPipelineExecutablePropertiesKHR(
        pi, dldi);
    printf("%zu executables (compiled shader stages)\n", execs.size());
    for (auto& e : execs) {
        printf("  stage %s: '%s' — %s\n",
               vk::to_string(e.stages).c_str(), e.name.data(),
               e.description.data());
    }

    for (uint32_t i = 0; i < execs.size(); ++i) {
        vk::PipelineExecutableInfoKHR pei{pipe.get(), i};
        auto stats =
            vk.device->getPipelineExecutableStatisticsKHR(pei, dldi);
        for (auto& s : stats) {
            printf("  %-24s ", s.name.data());
            if (s.format ==
                vk::PipelineExecutableStatisticFormatKHR::eBool32)
                printf("%s\n", s.value.b32 ? "true" : "false");
            else if (s.format ==
                     vk::PipelineExecutableStatisticFormatKHR::eInt64)
                printf("%ld\n", (long)s.value.i64);
            else if (s.format ==
                     vk::PipelineExecutableStatisticFormatKHR::eFloat64)
                printf("%.2f\n", s.value.f64);
            else
                printf("%lu\n", (unsigned long)s.value.u64);
        }
        // internal representations need a manual 3-pass dance:
        // (1) count, (2) sizes, (3) fill caller-provided buffers
        // 2-pass: query fills names + dataSize with pData=null;
        // then allocate and re-query for the payload
        uint32_t repCount = 0;
        VkPipelineExecutableInfoKHR cInfo{};
        cInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
        cInfo.pipeline = pipe.get();
        cInfo.executableIndex = i;
        dldi.vkGetPipelineExecutableInternalRepresentationsKHR(
            vk.device.get(), &cInfo, &repCount, nullptr);
        std::vector<VkPipelineExecutableInternalRepresentationKHR>
            reps(repCount);
        std::vector<std::vector<char>> blobs(repCount);
        for (uint32_t r = 0; r < repCount; ++r) {
            auto& e = reps[r];
            e.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
            e.pNext = nullptr;
            blobs[r].resize(8 << 20);        // 8MB per IR is plenty
            e.pData = blobs[r].data();
            e.dataSize = blobs[r].size();
        }
        VkResult r1 = dldi.vkGetPipelineExecutableInternalRepresentationsKHR(
            vk.device.get(), &cInfo, &repCount, reps.data());
        for (auto& r : reps) {
            // driver may not shrink dataSize — for text IRs the real
            // length is strlen(pData)
            size_t len = r.isText
                ? strnlen((char*)r.pData, r.dataSize) : r.dataSize;
            printf("  IR '%s' (%s): %zu bytes", r.name,
                   r.description, len);
            if (r.isText && len) {
                printf("\n---\n%.*s\n---",
                       (int)std::min<size_t>(len, 400),
                       (char*)r.pData);
            }
            printf("\n");
        }
    }
    printf("OK\n");
    return 0;
}
