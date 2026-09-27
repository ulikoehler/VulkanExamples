// SPDX-License-Identifier: CC0-1.0
//
// VK_EXT_mesh_shader — replace the entire fixed-function vertex
// pipeline with compute-like shader stages:
//   task shader (optional, like a workgroup that emits mesh workgroups)
//   mesh shader (emits vertices + triangles directly from code)
//
// No vertex buffer, no input assembler, no index buffer — the mesh
// shader *generates* geometry procedurally.
//
//   ./app --headless out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

// task shader: one workgroup emits 4 mesh workgroups (a 2x2 tile grid)
static const char* kTask = R"GLSL(
#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1) in;
taskPayloadSharedEXT uint tileId;
void main() {
    tileId = 0;
    EmitMeshTasksEXT(2, 2, 1);   // 4 mesh workgroups
}
)GLSL";

// mesh shader: each workgroup emits one quad (4 verts, 2 tris) whose
// position comes from the workgroup ID — a 2x2 grid of colored quads
static const char* kMesh = R"GLSL(
#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 4) in;
layout(triangles, max_vertices = 4, max_primitives = 2) out;
layout(location = 0) out vec3 vCol[];

void main() {
    // workgroup id -> quad position in a 2x2 layout
    uint gx = gl_WorkGroupID.x, gy = gl_WorkGroupID.y;
    vec2 base = vec2(-0.9 + float(gx) * 0.95,
                     -0.9 + float(gy) * 0.95);
    vec2 size = vec2(0.8, 0.8);
    vec3 col = vec3(gx, gy, 0.5) * 0.8 + 0.2;

    SetMeshOutputsEXT(4, 2);   // declare BEFORE writing outputs

    if (gl_LocalInvocationID.x < 4) {
        uint i = gl_LocalInvocationID.x;
        vec2 o = vec2((i & 1), (i >> 1));          // quad corners
        gl_MeshVerticesEXT[i].gl_Position =
            vec4(base + o * size, 0, 1);
        vCol[i] = col;
    }
    if (gl_LocalInvocationID.x < 2) {
        uint p = gl_LocalInvocationID.x;
        gl_PrimitiveTriangleIndicesEXT[p] =
            uvec3(0, 1, 2) + (p == 0 ? uvec3(0) : uvec3(1, 3, 0) - uvec3(0));
    }
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
    vk::PhysicalDeviceMeshShaderFeaturesEXT mf{};
    mf.setMeshShader(true).setTaskShader(true);
    vk.createDevice({VK_EXT_MESH_SHADER_EXTENSION_NAME}, &mf);
    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr, vk.device.get());

    // mesh shaders need SPIR-V 1.4/GLSL targets — compile via options
    auto mkShader = [&](const char* src, shaderc_shader_kind kind,
                        const char* name) {
        shaderc::CompileOptions opts;
        opts.SetTargetEnvironment(shaderc_target_env_vulkan,
                                  shaderc_env_version_vulkan_1_2);
        shaderc::Compiler cc;
        auto res = cc.CompileGlslToSpv(src, kind, name, opts);
        if (res.GetCompilationStatus() !=
            shaderc_compilation_status_success)
            throw std::runtime_error(name + std::string(": ") +
                                     res.GetErrorMessage());
        std::vector<uint32_t> spv(res.cbegin(), res.cend());
        vk::ShaderModuleCreateInfo si{};
        si.setCode(spv);
        return vk.device->createShaderModuleUnique(si);
    };

    auto ts = mkShader(kTask, shaderc_task_shader, "t.task");
    auto ms = mkShader(kMesh, shaderc_mesh_shader, "m.mesh");
    auto fs = mkShader(kFrag, shaderc_fragment_shader, "s.frag");

    vk::PipelineShaderStageCreateInfo stages[3];
    stages[0].setStage(vk::ShaderStageFlagBits::eTaskEXT)
        .setModule(ts.get()).setPName("main");
    stages[1].setStage(vk::ShaderStageFlagBits::eMeshEXT)
        .setModule(ms.get()).setPName("main");
    stages[2].setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(fs.get()).setPName("main");

    // NO vertex input state, NO input assembly needed — but the
    // structs must still be passed (empty)
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
    vk::PipelineMultisampleStateCreateInfo msaa{};
    msaa.setRasterizationSamples(vk::SampleCountFlagBits::e1);
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
    gi.setStages(stages)
        .setPViewportState(&vps).setPDynamicState(&dyn)
        .setPRasterizationState(&rs).setPMultisampleState(&msaa)
        .setPColorBlendState(&blend).setLayout(layout.get())
        .setPNext(&rendering);
    auto pres = vk.device->createGraphicsPipelineUnique({}, gi);
    if (pres.result != vk::Result::eSuccess)
        throw std::runtime_error("mesh pipeline failed");
    auto pipe = std::move(pres.value);

    constexpr uint32_t W = 640, Hh = 360;
    vkmini::Headless hl;
    hl.init(vk, W, Hh);
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
                                 pipe.get());
                  // drawMeshTasks replaces vkCmdDraw entirely —
                  // EXT function, fetched via dynamic dispatch
                  dldi.vkCmdDrawMeshTasksEXT(c, 1, 1, 1);
                  c.endRendering();
              });
    hl.savePpm(ppm);
    printf("wrote %s\n", ppm);
    return 0;
}
