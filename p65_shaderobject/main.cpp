// SPDX-License-Identifier: CC0-1.0
//
// VK_EXT_shader_object — the end of VkPipeline. Compile shaders,
// create VkShaderEXT objects, and at draw time bind the shader and
// set EVERY piece of pipeline state through vkCmdSet* calls. No
// pipeline objects, no pipeline cache, no state baking.
//
// This is the model D3D12 chose and Vulkan finally got.
//
//   ./app --headless out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) out vec3 vCol;
const vec2 p[3] = vec2[](vec2(-0.8,-0.8), vec2(0.8,-0.8),
                         vec2(0.0, 0.8));
const vec3 c[3] = vec3[](vec3(1,0.3,0.2), vec3(0.2,1,0.4),
                         vec3(0.3,0.5,1));
void main() {
    vCol = c[gl_VertexIndex];
    gl_Position = vec4(p[gl_VertexIndex], 0, 1);
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

    vk::PhysicalDeviceShaderObjectFeaturesEXT so{};
    so.setShaderObject(true);
    vk::PhysicalDeviceDynamicRenderingFeatures dyn{};
    dyn.setDynamicRendering(true);
    so.setPNext(&dyn);
    vk.createDevice({VK_EXT_SHADER_OBJECT_EXTENSION_NAME}, &so);
    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr, vk.device.get());

    // -------- create shader objects (no pipeline!) ---------------
    auto mkShader = [&](const char* src, shaderc_shader_kind kind,
                        vk::ShaderStageFlagBits stage,
                        const char* name) {
        shaderc::Compiler cc;
        auto res = cc.CompileGlslToSpv(src, kind, name);
        if (res.GetCompilationStatus() !=
            shaderc_compilation_status_success)
            throw std::runtime_error(res.GetErrorMessage());
        std::vector<uint32_t> spv(res.cbegin(), res.cend());
        VkShaderCreateInfoEXT sci{};
        sci.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
        sci.stage = (VkShaderStageFlagBits)stage;
        sci.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
        sci.codeSize = spv.size() * 4;
        sci.pCode = spv.data();
        sci.pName = "main";
        VkShaderEXT sh;
        vk::Result r = vk::Result(dldi.vkCreateShadersEXT(
            vk.device.get(), 1, &sci, nullptr, &sh));
        if (r != vk::Result::eSuccess)
            throw std::runtime_error("shader object failed");
        return sh;
    };
    VkShaderEXT vs = mkShader(kVert, shaderc_vertex_shader,
                              vk::ShaderStageFlagBits::eVertex, "v");
    VkShaderEXT fs = mkShader(kFrag, shaderc_fragment_shader,
                              vk::ShaderStageFlagBits::eFragment, "f");

    // ---------------- headless target ----------------------------
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

                  // EVERY state that used to live in VkPipeline is
                  // now dynamic — the command stream carries it all
                  vk::Viewport vp{0, 0, float(W), float(Hh), 0.f, 1.f};
                  c.setViewportWithCount(vp);
                  vk::Rect2D sc{{0,0}, hl.extent};
                  c.setScissorWithCount(sc);
                  dldi.vkCmdSetCullModeEXT(c, VK_CULL_MODE_NONE);
                  dldi.vkCmdSetFrontFaceEXT(c, VK_FRONT_FACE_COUNTER_CLOCKWISE);
                  dldi.vkCmdSetPrimitiveTopologyEXT(
                      c, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
                  dldi.vkCmdSetPrimitiveRestartEnableEXT(c, VK_FALSE);
                  dldi.vkCmdSetRasterizerDiscardEnableEXT(c, VK_FALSE);
                  dldi.vkCmdSetPolygonModeEXT(c, VK_POLYGON_MODE_FILL);
                  dldi.vkCmdSetRasterizationSamplesEXT(
                      c, VK_SAMPLE_COUNT_1_BIT);
                  dldi.vkCmdSetSampleMaskEXT(
                      c, VK_SAMPLE_COUNT_1_BIT,
                      (VkSampleMask[]){0xFFFFFFFF});
                  dldi.vkCmdSetAlphaToCoverageEnableEXT(c, VK_FALSE);
                  dldi.vkCmdSetDepthTestEnableEXT(c, VK_FALSE);
                  dldi.vkCmdSetDepthWriteEnableEXT(c, VK_FALSE);
                  dldi.vkCmdSetDepthCompareOpEXT(c, VK_COMPARE_OP_NEVER);
                  dldi.vkCmdSetDepthBoundsTestEnableEXT(c, VK_FALSE);
                  dldi.vkCmdSetStencilTestEnableEXT(c, VK_FALSE);
                  dldi.vkCmdSetStencilOpEXT(
                      c, VK_STENCIL_FACE_FRONT_BIT,
                      VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP,
                      VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
                  dldi.vkCmdSetStencilCompareMask(
                      c, VK_STENCIL_FACE_FRONT_BIT, 0xFF);
                  dldi.vkCmdSetStencilWriteMask(
                      c, VK_STENCIL_FACE_FRONT_BIT, 0xFF);
                  dldi.vkCmdSetStencilReference(
                      c, VK_STENCIL_FACE_FRONT_BIT, 0);
                  VkBool32 cbOff = VK_FALSE;
                  dldi.vkCmdSetColorBlendEnableEXT(c, 0, 1, &cbOff);
                  VkColorComponentFlags cwm =
                      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
                  dldi.vkCmdSetColorWriteMaskEXT(c, 0, 1, &cwm);
                  // bind shaders — no pipeline bind at all
                  VkShaderStageFlagBits st[2] = {
                      VK_SHADER_STAGE_VERTEX_BIT,
                      VK_SHADER_STAGE_FRAGMENT_BIT};
                  VkShaderEXT shs[2] = {vs, fs};
                  dldi.vkCmdBindShadersEXT(c, 2, st, shs);
                  c.draw(3, 1, 0, 0);
                  c.endRendering();
              });
    hl.savePpm(ppm);
    printf("wrote %s (no VkPipeline was created)\n", ppm);
    dldi.vkDestroyShaderEXT(vk.device.get(), vs, nullptr);
    dldi.vkDestroyShaderEXT(vk.device.get(), fs, nullptr);
    return 0;
}
