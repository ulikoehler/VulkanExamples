// SPDX-License-Identifier: CC0-1.0
//
// VK_KHR_ray_tracing_pipeline — the full RT pipeline, not ray query:
// vkCmdTraceRaysKHR runs a *ray generation* shader; traceRayEXT()
// inside it walks the TLAS and calls miss or closest-hit shaders.
// Results land in a storage image — the classic path-tracer shape.
//
// Three shader stages + a shader binding table (SBT) telling the
// hardware where each callable shader lives.
//
//   ./app --headless out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

// ---------- shaders (GL_EXT_ray_tracing) --------------------------
// raygen: one ray per pixel through a pinhole camera
static const char* kRgen = R"GLSL(
#version 460
#extension GL_EXT_ray_tracing : require
layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;
layout(set = 0, binding = 1, rgba8) uniform image2D img;
layout(location = 0) rayPayloadEXT vec3 pay;
void main() {
    vec2 uv = (vec2(gl_LaunchIDEXT.xy) + 0.5)
            / vec2(gl_LaunchSizeEXT.xy);
    vec2 ndc = uv * 2.0 - 1.0;
    vec3 origin = vec3(0, 0, -2);
    vec3 dir    = normalize(vec3(ndc.x, -ndc.y, 1.4));
    pay = vec3(0);
    traceRayEXT(scene, gl_RayFlagsOpaqueEXT, 0xFF,
                0 /*sbtOffset*/, 0 /*sbtStride*/, 0 /*missIdx*/,
                origin, 0.001, dir, 100.0, 0 /*payloadLoc*/);
    imageStore(img, ivec2(gl_LaunchIDEXT.xy), vec4(pay, 1));
}
)GLSL";

static const char* kMiss = R"GLSL(
#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) rayPayloadInEXT vec3 pay;
void main() {
    // sky gradient along ray direction
    vec3 d = normalize(gl_WorldRayDirectionEXT);
    pay = mix(vec3(0.5, 0.7, 1.0), vec3(0.05, 0.05, 0.15),
              clamp(d.y * 0.5 + 0.5, 0.0, 1.0));
}
)GLSL";

static const char* kHit = R"GLSL(
#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) rayPayloadInEXT vec3 pay;
void main() {
    // flat orange triangle
    pay = vec3(1.0, 0.45, 0.1);
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
    vk::PhysicalDeviceBufferDeviceAddressFeatures bda{};
    bda.setBufferDeviceAddress(true);
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR asf{};
    asf.setAccelerationStructure(true).setPNext(&bda);
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rtp{};
    rtp.setRayTracingPipeline(true).setPNext(&asf);
    vk.createDevice({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                     VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                     VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},
                    &rtp);
    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr, vk.device.get());

    // ================= BLAS: one triangle ==========================
    const float verts[9] = {-1.0f, -0.7f, 0.5f,
                             1.0f, -0.7f, 0.5f,
                             0.0f,  1.0f, 0.5f};
    const auto rtIn =
        vk::BufferUsageFlagBits::
            eAccelerationStructureBuildInputReadOnlyKHR |
        vk::BufferUsageFlagBits::eShaderDeviceAddress;
    auto vstage = vk.createBuffer(
        sizeof(verts), vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent, true);
    memcpy(vstage.mapped, verts, sizeof(verts));
    auto vb = vk.createBuffer(sizeof(verts),
        rtIn | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::BufferCopy cp{};
        cp.setSize(sizeof(verts));
        c.copyBuffer(vstage.buf.get(), vb.buf.get(), cp);
    });

    vk::AccelerationStructureGeometryTrianglesDataKHR tris{};
    tris.setVertexFormat(vk::Format::eR32G32B32Sfloat)
        .setVertexData(vk.device->getBufferAddress({vb.buf.get()}))
        .setVertexStride(12)
        .setMaxVertex(2)               // highest INDEX, not count!
        .setIndexType(vk::IndexType::eNoneKHR);
    vk::AccelerationStructureGeometryKHR geo{};
    geo.setGeometryType(vk::GeometryTypeKHR::eTriangles)
        .setGeometry({tris})
        .setFlags(vk::GeometryFlagBitsKHR::eOpaque);
    vk::AccelerationStructureBuildGeometryInfoKHR bi{};
    bi.setType(vk::AccelerationStructureTypeKHR::eBottomLevel)
        .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::
                      ePreferFastTrace)
        .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
        .setGeometries(geo);
    auto bsizes = vk.device->getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, bi, 1, dldi);
    auto bbuf = vk.createBuffer(bsizes.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::AccelerationStructureCreateInfoKHR aci{};
    aci.setBuffer(bbuf.buf.get())
        .setSize(bsizes.accelerationStructureSize)
        .setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
    VkAccelerationStructureKHR blas;
    if (dldi.vkCreateAccelerationStructureKHR(
            vk.device.get(),
            reinterpret_cast<const VkAccelerationStructureCreateInfoKHR*>(
                &aci), nullptr, &blas) != VK_SUCCESS)
        throw std::runtime_error("BLAS create failed");
    auto bscratch = vk.createBuffer(bsizes.buildScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    bi.setDstAccelerationStructure(blas)
        .setScratchData(
            vk.device->getBufferAddress({bscratch.buf.get()}));
    VkAccelerationStructureBuildGeometryInfoKHR cbi = bi;
    const VkAccelerationStructureBuildRangeInfoKHR br{1, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR* brs = &br;
    vk.oneTime([&](vk::CommandBuffer c) {
        dldi.vkCmdBuildAccelerationStructuresKHR(c, 1, &cbi, &brs);
    });

    // ================= TLAS: one instance ==========================
    VkAccelerationStructureInstanceKHR inst{};
    inst.transform.matrix[0][0] = 1.f;
    inst.transform.matrix[1][1] = 1.f;
    inst.transform.matrix[2][2] = 1.f;
    inst.mask = 0xFF;
    inst.flags =
        VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    VkAccelerationStructureDeviceAddressInfoKHR bai{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    bai.accelerationStructure = blas;
    inst.accelerationStructureReference =
        dldi.vkGetAccelerationStructureDeviceAddressKHR(
            vk.device.get(), &bai);
    auto ibuf = vk.createBuffer(sizeof(inst),
        rtIn | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    auto istage = vk.createBuffer(sizeof(inst),
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent, true);
    memcpy(istage.mapped, &inst, sizeof(inst));
    vk::AccelerationStructureGeometryInstancesDataKHR insts{};
    insts.setArrayOfPointers(false)
        .setData(vk.device->getBufferAddress({ibuf.buf.get()}));
    vk::AccelerationStructureGeometryKHR igeo{};
    igeo.setGeometryType(vk::GeometryTypeKHR::eInstances)
        .setGeometry({insts});
    vk::AccelerationStructureBuildGeometryInfoKHR tbi{};
    tbi.setType(vk::AccelerationStructureTypeKHR::eTopLevel)
        .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::
                      ePreferFastTrace)
        .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
        .setGeometries(igeo);
    auto tsizes = vk.device->getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, tbi, 1, dldi);
    auto tbuf = vk.createBuffer(tsizes.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::AccelerationStructureCreateInfoKHR tci{};
    tci.setBuffer(tbuf.buf.get())
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
        .setScratchData(
            vk.device->getBufferAddress({tscratch.buf.get()}));
    VkAccelerationStructureBuildGeometryInfoKHR tcbi = tbi;
    const VkAccelerationStructureBuildRangeInfoKHR tr{1, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR* trs = &tr;
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::BufferCopy icp{};
        icp.setSize(sizeof(inst));
        c.copyBuffer(istage.buf.get(), ibuf.buf.get(), icp);
        vk::MemoryBarrier mb{};
        mb.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(
                vk::AccessFlagBits::eAccelerationStructureReadKHR);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                          vk::PipelineStageFlagBits::
                              eAccelerationStructureBuildKHR,
                          {}, mb, {}, {});
        dldi.vkCmdBuildAccelerationStructuresKHR(c, 1, &tcbi, &trs);
    });

    // ================= shaders -> modules ==========================
    // RT shaders emit SPV_KHR_ray_tracing -> SPIR-V 1.4+ target
    shaderc::CompileOptions opts;
    opts.SetTargetEnvironment(shaderc_target_env_vulkan,
                              shaderc_env_version_vulkan_1_3);
    shaderc::Compiler cc;
    auto mkMod = [&](const char* src, shaderc_shader_kind kind,
                     const char* name) {
        auto r = cc.CompileGlslToSpv(src, kind, name, opts);
        if (r.GetCompilationStatus() !=
            shaderc_compilation_status_success)
            throw std::runtime_error(std::string(name) + ": " +
                                     r.GetErrorMessage());
        std::vector<uint32_t> code(r.cbegin(), r.cend());
        return vk.device->createShaderModuleUnique({{}, code});
    };
    auto mRgen = mkMod(kRgen, shaderc_raygen_shader, "rgen");
    auto mMiss = mkMod(kMiss, shaderc_miss_shader, "miss");
    auto mHit = mkMod(kHit, shaderc_closesthit_shader, "chit");

    // ================= RT pipeline + shader groups =================
    vk::RayTracingShaderGroupCreateInfoKHR groups[3]{};
    groups[0].setType(vk::RayTracingShaderGroupTypeKHR::eGeneral)
        .setGeneralShader(0);
    groups[1].setType(vk::RayTracingShaderGroupTypeKHR::eGeneral)
        .setGeneralShader(1);
    groups[2].setType(vk::RayTracingShaderGroupTypeKHR::
                          eTrianglesHitGroup)
        .setClosestHitShader(2);

    vk::PipelineShaderStageCreateInfo st[3];
    st[0].setStage(vk::ShaderStageFlagBits::eRaygenKHR)
        .setModule(mRgen.get()).setPName("main");
    st[1].setStage(vk::ShaderStageFlagBits::eMissKHR)
        .setModule(mMiss.get()).setPName("main");
    st[2].setStage(vk::ShaderStageFlagBits::eClosestHitKHR)
        .setModule(mHit.get()).setPName("main");

    // descriptor set: TLAS + storage image
    vk::DescriptorSetLayoutBinding db[2];
    db[0].setBinding(0)
        .setDescriptorType(
            vk::DescriptorType::eAccelerationStructureKHR)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);
    db[1].setBinding(1)
        .setDescriptorType(vk::DescriptorType::eStorageImage)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);
    vk::DescriptorSetLayoutCreateInfo dsli{};
    dsli.setBindings(db);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dsli);
    vk::DescriptorPoolSize psz[2] = {
        {vk::DescriptorType::eAccelerationStructureKHR, 1},
        {vk::DescriptorType::eStorageImage, 1}};
    vk::DescriptorPoolCreateInfo dpi{};
    dpi.setMaxSets(1).setPoolSizes(psz);
    auto dpool = vk.device->createDescriptorPoolUnique(dpi);
    vk::DescriptorSetAllocateInfo dai{};
    dai.setDescriptorPool(dpool.get()).setSetLayouts(dsl.get());
    auto dsets = vk.device->allocateDescriptorSets(dai);

    vk::PipelineLayoutCreateInfo pli{};
    pli.setSetLayouts(dsl.get());
    auto playout = vk.device->createPipelineLayoutUnique(pli);

    vk::RayTracingPipelineCreateInfoKHR rci{};
    rci.setStages(st)
        .setGroups(groups)
        .setMaxPipelineRayRecursionDepth(1)
        .setLayout(playout.get());
    VkPipeline rpipe;
    if (dldi.vkCreateRayTracingPipelinesKHR(
            vk.device.get(), {}, {}, 1,
            reinterpret_cast<const VkRayTracingPipelineCreateInfoKHR*>(
                &rci), nullptr, &rpipe) != VK_SUCCESS)
        throw std::runtime_error("RT pipeline failed");

    // ================= SBT =========================================
    auto rtp2 = vk.phys.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    auto& rp = rtp2.get<
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    const uint32_t hs = rp.shaderGroupHandleSize;
    const uint32_t ha = rp.shaderGroupHandleAlignment;
    const uint32_t base = rp.shaderGroupBaseAlignment;
    auto alignUp = [](uint32_t v, uint32_t a) {
        return (v + a - 1) / a * a;
    };
    const uint32_t recStride = alignUp(hs, ha);
    const uint32_t sbtSize = 3 * alignUp(recStride, base);
    auto sbt = vk.createBuffer(sbtSize,
        vk::BufferUsageFlagBits::eShaderBindingTableKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    // pull the 3 group handles out of the pipeline
    std::vector<uint8_t> handles(3 * hs);
    if (dldi.vkGetRayTracingShaderGroupHandlesKHR(
            vk.device.get(), rpipe, 0, 3, handles.size(),
            handles.data()) != VK_SUCCESS)
        throw std::runtime_error("group handles failed");
    auto sbtStage = vk.createBuffer(sbtSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent, true);
    // record i sits at i*alignUp(recStride, base) -- raygen/miss/hit
    // each get their own SBT region (raygen MUST be alone per spec
    // if >1 record; we keep the same layout for all three)
    for (int i = 0; i < 3; ++i)
        memcpy(static_cast<uint8_t*>(sbtStage.mapped) +
                   i * alignUp(recStride, base),
               handles.data() + i * hs, hs);
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::BufferCopy cp{};
        cp.setSize(sbtSize);
        c.copyBuffer(sbtStage.buf.get(), sbt.buf.get(), cp);
    });
    auto sbtAddr = vk.device->getBufferAddress({sbt.buf.get()});
    const uint32_t regionStride = alignUp(recStride, base);
    vk::StridedDeviceAddressRegionKHR rgn[3];
    for (int i = 0; i < 3; ++i)
        rgn[i]
            .setDeviceAddress(sbtAddr + i * regionStride)
            .setStride(recStride)
            .setSize(recStride);
    vk::StridedDeviceAddressRegionKHR callable{};

    // ================= output image + descriptors ==================
    constexpr uint32_t W = 640, H = 360;
    auto img = vk.createImage(W, H, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eTransferSrc);
    vk::WriteDescriptorSetAccelerationStructureKHR asw{};
    vk::AccelerationStructureKHR tlasVk{tlas};
    asw.setAccelerationStructures(tlasVk);
    vk::DescriptorImageInfo ii{};
    ii.setImageView(img.view.get())
        .setImageLayout(vk::ImageLayout::eGeneral);
    vk::WriteDescriptorSet wr[2];
    wr[0].setDstSet(dsets[0])
        .setDstBinding(0)
        .setDescriptorType(
            vk::DescriptorType::eAccelerationStructureKHR)
        .setDescriptorCount(1)
        .setPNext(&asw);
    wr[1].setDstSet(dsets[0])
        .setDstBinding(1)
        .setDescriptorType(vk::DescriptorType::eStorageImage)
        .setImageInfo(ii);
    vk.device->updateDescriptorSets(wr, {});

    auto rb = vk.createBuffer(vk::DeviceSize(W) * H * 4,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent, true);

    vk::ImageSubresourceRange sr{vk::ImageAspectFlagBits::eColor, 0,
                                 1, 0, 1};
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::ImageMemoryBarrier t0{};
        t0.setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eGeneral)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(img.img.get())
            .setSubresourceRange(sr)
            .setDstAccessMask(vk::AccessFlagBits::eShaderWrite);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::
                              eRayTracingShaderKHR,
                          {}, {}, {}, t0);

        c.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, rpipe);
        c.bindDescriptorSets(
            vk::PipelineBindPoint::eRayTracingKHR, playout.get(), 0,
            dsets[0], {});
        const auto* rgnC =
            reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(
                rgn);
        dldi.vkCmdTraceRaysKHR(c, &rgnC[0], &rgnC[1], &rgnC[2],
                               reinterpret_cast<
                                   const VkStridedDeviceAddressRegionKHR*>(
                                   &callable), W, H, 1);

        vk::ImageMemoryBarrier t1{};
        t1.setOldLayout(vk::ImageLayout::eGeneral)
            .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(img.img.get())
            .setSubresourceRange(sr)
            .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        c.pipelineBarrier(
            vk::PipelineStageFlagBits::eRayTracingShaderKHR,
            vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, t1);
        vk::BufferImageCopy rc{};
        rc.setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
            .setImageExtent({W, H, 1});
        c.copyImageToBuffer(img.img.get(),
                            vk::ImageLayout::eTransferSrcOptimal,
                            rb.buf.get(), rc);
    });

    // PPM + verify: count "triangle orange" pixels vs sky pixels
    std::ofstream f(ppm, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";
    auto* px = static_cast<const uint8_t*>(rb.mapped);
    size_t orange = 0, other = 0;
    for (uint32_t i = 0; i < W * H; ++i) {
        const uint8_t* p = px + i * 4;
        f.write(reinterpret_cast<const char*>(p), 3);
        if (p[0] > 200 && p[1] > 90 && p[1] < 140 && p[2] < 40)
            ++orange;
        else
            ++other;
    }
    printf("traceRays %ux%u: %zu orange hit pixels, %zu sky/miss\n",
           W, H, orange, other);
    if (orange < 1000 || orange > W * H / 2)
        throw std::runtime_error("RT output implausible");
    printf("done: raygen -> traceRayEXT -> miss/hit via SBT\n");
    dldi.vkDestroyAccelerationStructureKHR(vk.device.get(), blas,
                                           nullptr);
    dldi.vkDestroyAccelerationStructureKHR(vk.device.get(), tlas,
                                           nullptr);
    dldi.vkDestroyPipeline(vk.device.get(), rpipe, nullptr);
    return 0;
}
