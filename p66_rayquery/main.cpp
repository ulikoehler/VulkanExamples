// SPDX-License-Identifier: CC0-1.0
//
// VK_KHR_ray_query — hardware ray tracing in a plain fragment shader.
// We build a BLAS holding one triangle, bind it to the fragment
// shader, and cast a shadow ray per pixel toward a point light.
// Where the triangle occludes the light a hard shadow appears on the
// ground grid.
//
//   ./app --headless out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

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

// One shadow ray per pixel: world-space point on the y=0 ground
// plane -> point light. The BLAS triangle blocks part of the view.
static const char* kFrag = R"GLSL(
#version 460
#extension GL_EXT_ray_query : require
layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
const vec3 LIGHT = vec3(-1.5, 3.0, 0.5);
void main() {
    vec2 xz = (vUV - 0.5) * 8.0;
    vec3 P = vec3(xz.x, 0.0, xz.y);
    vec3 dir = LIGHT - P;
    float tmax = length(dir);
    dir /= tmax;
    rayQueryEXT q;
    rayQueryInitializeEXT(q, scene,
        gl_RayFlagsTerminateOnFirstHitEXT, 0xFF,
        P + dir * 0.02, 0.0, dir, tmax - 0.04);
    while (rayQueryProceedEXT(q)) {}
    bool shadow = rayQueryGetIntersectionTypeEXT(q, true)
                != gl_RayQueryCommittedIntersectionNoneEXT;
    // ground plane shading: checker + diffuse falloff
    ivec2 cell = ivec2(floor(xz));
    float checker = ((cell.x + cell.y) & 1) == 0 ? 0.85 : 0.55;
    vec3 base = vec3(0.9, 0.75, 0.4) * checker
              * max(dot(dir, vec3(0, 1, 0)), 0.0);
    outColor = vec4(base * (shadow ? 0.15 : 1.0), 1);
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
    // feature chain: BDA -> AS -> ray query (all required)
    vk::PhysicalDeviceBufferDeviceAddressFeatures bda{};
    bda.setBufferDeviceAddress(true);
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR asf{};
    asf.setAccelerationStructure(true).setPNext(&bda);
    vk::PhysicalDeviceRayQueryFeaturesKHR rqf{};
    rqf.setRayQuery(true).setPNext(&asf);
    vk.createDevice({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                     VK_KHR_RAY_QUERY_EXTENSION_NAME,
                     VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},
                    &rqf);
    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr, vk.device.get());

    // ---- BLAS: a single triangle floating above the ground --------
    const float verts[9] = {-1.6f, 1.2f, -1.0f,
                             1.8f, 1.6f, -0.4f,
                             0.2f, 1.0f,  1.6f};
    const vk::BufferUsageFlags rt =
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
        vk::BufferUsageFlagBits::eShaderDeviceAddress;
    auto staging = vk.createBuffer(sizeof(verts),
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);
    memcpy(staging.mapped, verts, sizeof(verts));
    auto vb = vk.createBuffer(sizeof(verts),
        rt | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::BufferCopy cp{};
        cp.setSize(sizeof(verts));
        c.copyBuffer(staging.buf.get(), vb.buf.get(), cp);
    });
    auto vbAddr = vk.device->getBufferAddress({vb.buf.get()});

    vk::AccelerationStructureGeometryTrianglesDataKHR tris{};
    tris.setVertexFormat(vk::Format::eR32G32B32Sfloat)
        .setVertexData(vbAddr)
        .setVertexStride(sizeof(float) * 3)
        .setMaxVertex(2) // HIGHEST index — 3 verts = indices 0..2
                         // (3 reads a garbage 4th vertex: noisy AS)
        .setIndexType(vk::IndexType::eNoneKHR);
    vk::AccelerationStructureGeometryKHR geo{};
    geo.setGeometryType(vk::GeometryTypeKHR::eTriangles)
        .setGeometry({tris})
        .setFlags(vk::GeometryFlagBitsKHR::eOpaque);
    vk::AccelerationStructureBuildGeometryInfoKHR bi{};
    bi.setType(vk::AccelerationStructureTypeKHR::eBottomLevel)
        .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace)
        .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
        .setGeometries(geo);
    uint32_t primCount = 1;
    auto sizes = vk.device->getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, bi, primCount,
        dldi);
    auto asBuf = vk.createBuffer(sizes.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
        vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::AccelerationStructureCreateInfoKHR aci{};
    aci.setBuffer(asBuf.buf.get())
        .setSize(sizes.accelerationStructureSize)
        .setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
    VkAccelerationStructureKHR blas;
    if (dldi.vkCreateAccelerationStructureKHR(
            vk.device.get(),
            reinterpret_cast<const VkAccelerationStructureCreateInfoKHR*>(
                &aci), nullptr, &blas) != VK_SUCCESS)
        throw std::runtime_error("BLAS create failed");
    auto scratch = vk.createBuffer(sizes.buildScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    bi.setDstAccelerationStructure(blas)
        .setScratchData(vk.device->getBufferAddress({scratch.buf.get()}));
    VkAccelerationStructureBuildGeometryInfoKHR cbi = bi;
    const VkAccelerationStructureBuildRangeInfoKHR range{1, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;
    vk.oneTime([&](vk::CommandBuffer c) {
        dldi.vkCmdBuildAccelerationStructuresKHR(c, 1, &cbi, &ranges);
    });

    // ---- TLAS: one instance pointing at the BLAS ------------------
    // (spec-conformant path — binding a raw BLAS to ray queries is
    // allowed but some drivers only handle the instanced form)
    VkAccelerationStructureInstanceKHR inst{};
    inst.transform.matrix[0][0] = 1.f;
    inst.transform.matrix[1][1] = 1.f;
    inst.transform.matrix[2][2] = 1.f;
    inst.instanceCustomIndex = 0;
    inst.mask = 0xFF;
    inst.instanceShaderBindingTableRecordOffset = 0;
    inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    VkAccelerationStructureDeviceAddressInfoKHR asai{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    asai.accelerationStructure = blas;
    inst.accelerationStructureReference =
        dldi.vkGetAccelerationStructureDeviceAddressKHR(vk.device.get(),
                                                        &asai);
    auto instBuf = vk.createBuffer(sizeof(inst),
        vk::BufferUsageFlagBits::
            eAccelerationStructureBuildInputReadOnlyKHR |
        vk::BufferUsageFlagBits::eShaderDeviceAddress |
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    auto instStage = vk.createBuffer(sizeof(inst),
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);
    memcpy(instStage.mapped, &inst, sizeof(inst));
    vk::AccelerationStructureGeometryKHR instGeo{};
    vk::AccelerationStructureGeometryInstancesDataKHR insts{};
    insts.setArrayOfPointers(false)
        .setData(vk.device->getBufferAddress({instBuf.buf.get()}));
    instGeo.setGeometryType(vk::GeometryTypeKHR::eInstances)
        .setGeometry({insts});
    vk::AccelerationStructureBuildGeometryInfoKHR tbi{};
    tbi.setType(vk::AccelerationStructureTypeKHR::eTopLevel)
        .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::
                      ePreferFastTrace)
        .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
        .setGeometries(instGeo);
    auto tsizes = vk.device->getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, tbi, 1, dldi);
    auto tlasBuf = vk.createBuffer(tsizes.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
        vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::AccelerationStructureCreateInfoKHR tci{};
    tci.setBuffer(tlasBuf.buf.get())
        .setSize(tsizes.accelerationStructureSize)
        .setType(vk::AccelerationStructureTypeKHR::eTopLevel);
    VkAccelerationStructureKHR tlas;
    if (dldi.vkCreateAccelerationStructureKHR(
            vk.device.get(),
            reinterpret_cast<const VkAccelerationStructureCreateInfoKHR*>(
                &tci), nullptr, &tlas) != VK_SUCCESS)
        throw std::runtime_error("TLAS create failed");
    auto tscratch = vk.createBuffer(tsizes.buildScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    tbi.setDstAccelerationStructure(tlas)
        .setScratchData(vk.device->getBufferAddress({tscratch.buf.get()}));
    VkAccelerationStructureBuildGeometryInfoKHR tcbi = tbi;
    const VkAccelerationStructureBuildRangeInfoKHR trange{1, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR* tranges = &trange;
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::BufferCopy icp{};
        icp.setSize(sizeof(inst));
        c.copyBuffer(instStage.buf.get(), instBuf.buf.get(), icp);
        // instance upload must land before the TLAS build reads it
        vk::MemoryBarrier mb{};
        mb.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(
                vk::AccessFlagBits::eAccelerationStructureReadKHR);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                          vk::PipelineStageFlagBits::
                              eAccelerationStructureBuildKHR,
                          {}, mb, {}, {});
        dldi.vkCmdBuildAccelerationStructuresKHR(c, 1, &tcbi, &tranges);
    });

    // ---- pipeline ------------------------------------------------
    constexpr uint32_t W = 640, H = 360;
    vkmini::Headless hl;
    hl.init(vk, W, H);
    vk::DescriptorSetLayoutBinding bind0{};
    bind0.setBinding(0)
        .setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eFragment);
    vk::DescriptorSetLayoutCreateInfo dsli{};
    dsli.setBindings(bind0);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dsli);
    vk::DescriptorPoolSize psz{
        vk::DescriptorType::eAccelerationStructureKHR, 1};
    vk::DescriptorPoolCreateInfo dpi{};
    dpi.setMaxSets(1).setPoolSizes(psz);
    auto pool = vk.device->createDescriptorPoolUnique(dpi);
    vk::DescriptorSetAllocateInfo dai{};
    dai.setDescriptorPool(pool.get())
        .setSetLayouts(dsl.get());
    auto dsets = vk.device->allocateDescriptorSetsUnique(dai);
    auto& dset = dsets.front();
    vk::WriteDescriptorSetAccelerationStructureKHR asw{};
    vk::AccelerationStructureKHR tlasVk{tlas};
    asw.setAccelerationStructures(tlasVk);
    vk::WriteDescriptorSet wr{};
    wr.setDstSet(dset.get()).setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR)
        .setDescriptorCount(1)
        .setPNext(&asw);
    vk.device->updateDescriptorSets(wr, {});

    auto vs = vk.shader(kVert, shaderc_vertex_shader, "s.vert");
    // GL_EXT_ray_query emits SPV_KHR_ray_query -> needs SPIR-V 1.4+
    shaderc::CompileOptions opts;
    opts.SetTargetEnvironment(shaderc_target_env_vulkan,
                              shaderc_env_version_vulkan_1_3);
    shaderc::Compiler cc;
    auto fspv = cc.CompileGlslToSpv(kFrag, shaderc_fragment_shader,
                                  "s.frag", opts);
    if (fspv.GetCompilationStatus() !=
        shaderc_compilation_status_success)
        throw std::runtime_error(std::string("frag: ") +
                                 fspv.GetErrorMessage());
    std::vector<uint32_t> code(fspv.cbegin(), fspv.cend());
    auto fs = vk.device->createShaderModuleUnique({{}, code});

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
    vk::PipelineLayoutCreateInfo pli{};
    pli.setSetLayouts(dsl.get());
    auto layout = vk.device->createPipelineLayoutUnique(pli);
    vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(cfmt);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(stages)
        .setPVertexInputState(&vin).setPInputAssemblyState(&ia)
        .setPViewportState(&vps).setPDynamicState(&dyn)
        .setPRasterizationState(&rs).setPMultisampleState(&ms)
        .setPColorBlendState(&blend).setLayout(layout.get())
        .setPNext(&rendering);
    auto pres = vk.device->createGraphicsPipelineUnique({}, gi);
    if (pres.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    auto pipe = std::move(pres.value);

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
                  c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 pipe.get());
                  c.bindDescriptorSets(
                      vk::PipelineBindPoint::eGraphics, layout.get(),
                      0, dset.get(), {});
                  c.setViewport(0, vk::Viewport{0, 0, float(W),
                                              float(H), 0.f, 1.f});
                  c.setScissor(0, vk::Rect2D{{0, 0}, {W, H}});
                  c.draw(3, 1, 0, 0);
                  c.endRendering();
              });
    hl.savePpm(ppm);
    dldi.vkDestroyAccelerationStructureKHR(vk.device.get(), tlas,
                                           nullptr);
    dldi.vkDestroyAccelerationStructureKHR(vk.device.get(), blas,
                                           nullptr);
    printf("wrote %s\n", ppm);
    return 0;
}
