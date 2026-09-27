// SPDX-License-Identifier: CC0-1.0
//
// VK_KHR_cooperative_matrix — the tensor-core style op in GLSL:
// coopmat<> is a matrix DISTRIBUTED over a whole subgroup; no lane
// owns it. coopMatLoad/coopMatStore scatter/gather it across the
// workgroup, coopMatMulAdd(A, B, C) computes A*B+C in one op.
//
// Demo: one 16x16x16 int8 x int8 -> int32 multiply-add, exactly
// verified against the CPU (int math: bit-exact).
//
//   ./app
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"
#include <random>

// The whole workgroup is ONE subgroup: the matrix lives inside the
// subgroup's registers, distributed across lanes in a layout the
// driver picks — coopMatLoad/Store is the only way in/out.
static const char* kComp = R"GLSL(
#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_8bit_storage : require

layout(local_size_x_id = 0) in;   // = subgroup size, set at runtime

layout(set = 0, binding = 0, std430) readonly buffer BA {
    int8_t a[];
};
layout(set = 0, binding = 1, std430) readonly buffer BB {
    int8_t b[];
};
layout(set = 0, binding = 2, std430) buffer BC {
    int32_t c[];
};

void main() {
    coopmat<int8_t,  gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> A;
    coopmat<int8_t,  gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> B;
    coopmat<int32_t, gl_ScopeSubgroup, 16, 16,
            gl_MatrixUseAccumulator> C;

    // C starts as the bias matrix; the op computes A*B + C
    coopMatLoad(A, a, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(B, b, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(C, c, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    C = coopMatMulAdd(A, B, C);
    coopMatStore(C, c, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
}
)GLSL";

int main() {
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    vk::PhysicalDeviceCooperativeMatrixFeaturesKHR cm{};
    cm.setCooperativeMatrix(true);
    vk::PhysicalDeviceShaderFloat16Int8Features i8{};
    i8.setShaderInt8(true);
    vk::PhysicalDevice8BitStorageFeatures s8{};
    s8.setStorageBuffer8BitAccess(true);
    vk::PhysicalDeviceDynamicRenderingFeatures dyn{};
    cm.setPNext(&i8);
    i8.setPNext(&s8);
    s8.setPNext(&dyn);
    vk.createDevice({VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME}, &cm);

    // the driver's supported matrix shapes (all 16x16x16 subgroup on
    // NAVI32); pick the s8 x s8 -> s32 entry
    uint32_t n = 0;
    auto fp = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vk.instance->getProcAddr(
            "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!fp || (fp(vk.phys, &n, nullptr), !n)) {
        printf("SKIP: no cooperative matrix properties\n");
        return 0;
    }
    std::vector<VkCooperativeMatrixPropertiesKHR> props(n);
    for (auto& p : props)
        p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    fp(vk.phys, &n, props.data());
    bool haveS8 = false;
    for (auto& p : props)
        if (p.MSize == 16 && p.NSize == 16 && p.KSize == 16 &&
            p.AType == VK_COMPONENT_TYPE_SINT8_KHR &&
            p.BType == VK_COMPONENT_TYPE_SINT8_KHR &&
            p.ResultType == VK_COMPONENT_TYPE_SINT32_KHR &&
            p.scope == VK_SCOPE_SUBGROUP_KHR)
            haveS8 = true;
    printf("coop matrix shapes: %u, s8*s8->s32 subgroup: %s\n", n,
           haveS8 ? "yes" : "no");
    if (!haveS8) {
        printf("SKIP: required shape unsupported\n");
        return 0;
    }

    uint32_t subSize = vk.phys.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceSubgroupProperties>()
            .get<vk::PhysicalDeviceSubgroupProperties>()
            .subgroupSize;
    printf("subgroupSize = %u\n", subSize);

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dist(-127, 127);
    std::vector<int8_t> A(256), B(256);
    std::vector<int32_t> C(256);
    for (auto& v : A) v = int8_t(dist(rng));
    for (auto& v : B) v = int8_t(dist(rng));
    for (auto& v : C) v = dist(rng);

    auto mk = [&](const void* d, size_t sz, vk::BufferUsageFlags u) {
        auto buf = vk.createBuffer(sz, u,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent, true);
        memcpy(buf.mapped, d, sz);
        return buf;
    };
    auto bA = mk(A.data(), A.size(),
                 vk::BufferUsageFlagBits::eStorageBuffer);
    auto bB = mk(B.data(), B.size(),
                 vk::BufferUsageFlagBits::eStorageBuffer);
    auto bC = mk(C.data(), C.size() * 4,
                 vk::BufferUsageFlagBits::eStorageBuffer);

    // compile with SPIR-V 1.3+ (cooperative matrix needs it)
    shaderc::CompileOptions opts;
    opts.SetTargetEnvironment(shaderc_target_env_vulkan,
                              shaderc_env_version_vulkan_1_3);
    shaderc::Compiler cc;
    auto res = cc.CompileGlslToSpv(kComp, shaderc_compute_shader,
                                   "m.comp", opts);
    if (res.GetCompilationStatus() !=
        shaderc_compilation_status_success)
        throw std::runtime_error(std::string("comp: ") +
                                 res.GetErrorMessage());
    std::vector<uint32_t> spv(res.cbegin(), res.cend());
    auto comp = vk.device->createShaderModuleUnique({{}, spv});

    vk::DescriptorSetLayoutBinding bs[3];
    for (int i = 0; i < 3; ++i)
        bs[i].setBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    vk::DescriptorSetLayoutCreateInfo sli{};
    sli.setBindings(bs);
    auto setLayout = vk.device->createDescriptorSetLayoutUnique(sli);
    vk::DescriptorPoolSize ps{vk::DescriptorType::eStorageBuffer, 3};
    vk::DescriptorPoolCreateInfo dpi{};
    dpi.setMaxSets(1).setPoolSizes(ps);
    auto dpool = vk.device->createDescriptorPoolUnique(dpi);
    vk::DescriptorSetAllocateInfo dai{};
    dai.setDescriptorPool(dpool.get())
        .setSetLayouts(setLayout.get());
    auto sets = vk.device->allocateDescriptorSets(dai);
    vk::DescriptorBufferInfo infos[3] = {
        {bA.buf.get(), 0, VK_WHOLE_SIZE},
        {bB.buf.get(), 0, VK_WHOLE_SIZE},
        {bC.buf.get(), 0, VK_WHOLE_SIZE}};
    vk::WriteDescriptorSet wr[3];
    for (int i = 0; i < 3; ++i)
        wr[i].setDstSet(sets[0])
            .setDstBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(infos[i]);
    vk.device->updateDescriptorSets(wr, {});

    vk::PipelineLayoutCreateInfo li{};
    li.setSetLayouts(setLayout.get());
    auto layout = vk.device->createPipelineLayoutUnique(li);

    // local_size_x = subgroup size via specialization constant
    vk::SpecializationMapEntry me{0, 0, 4};
    vk::SpecializationInfo spec{1, &me, 4, &subSize};
    vk::ComputePipelineCreateInfo ci{};
    ci.stage.setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(comp.get())
        .setPName("main")
        .setPSpecializationInfo(&spec);
    ci.setLayout(layout.get());
    auto pres = vk.device->createComputePipelineUnique({}, ci);
    if (pres.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    auto pipe = std::move(pres.value);

    vk.oneTime([&](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.get());
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             layout.get(), 0, sets[0], {});
        c.dispatch(1, 1, 1);
    });

    // CPU reference: C += A*B
    auto* gc = static_cast<const int32_t*>(bC.mapped);
    size_t bad = 0;
    for (int r = 0; r < 16; ++r)
        for (int col = 0; col < 16; ++col) {
            int32_t ref = C[r * 16 + col];
            for (int k = 0; k < 16; ++k)
                ref += int32_t(A[r * 16 + k]) * int32_t(B[k * 16 + col]);
            if (gc[r * 16 + col] != ref) ++bad;
        }
    printf("16x16x16 int8 matmul+add: bad=%zu\n", bad);
    if (bad)
        throw std::runtime_error("coopmat mismatch");
    printf("done: subgroup-distributed matrix multiply verified\n");
    return 0;
}
