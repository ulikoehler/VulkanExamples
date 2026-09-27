// SPDX-License-Identifier: CC0-1.0
//
// Subgroup (warp/wavefront) operations — the SIMD unit underneath
// your shader. One workgroup of 64 lanes computes a tree reduction,
// a ballot count, and a broadcast — zero shared memory, zero barriers.
//
//   ./app     prints + verifies the subgroup results
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

static const char* kComp = R"GLSL(
#version 460
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
layout(local_size_x = 64) in;

layout(std430, binding = 0) buffer Out {
    uint  sum;        // subgroupAdd over lane values 1..64
    uint  count;      // ballot: lanes where laneId % 3 == 0
    uint  bcast;      // broadcast of lane 0's value
    uint  shuffled;   // lane 0 reads lane 32's value
    uint  isFirst;    // subgroupElect behavior check
    uint  sgSize;     // actual subgroup size
};

void main() {
    uint lane = gl_SubgroupInvocationID;
    uint val  = lane + 1;              // lane l holds l+1

    // subgroup ops MUST be evaluated by all active lanes — putting
    // them inside `if (lane == 0)` narrows the subgroup to lane 0
    // and the "reduction" returns only lane 0's contribution!
    uint s = subgroupAdd(val);
    uvec4 b = subgroupBallot(lane % 3 == 0);
    uint bc = subgroupBroadcastFirst(val);
    uint sh = subgroupShuffle(val, 32);
    if (lane == 0) {
        sum = s;
        count = subgroupBallotBitCount(b);
        bcast = bc;
        shuffled = sh;
        sgSize = gl_SubgroupSize;
    }
}
)GLSL";

int main() {
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice();

    auto sub = vk.phys.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceSubgroupProperties>()
        .get<vk::PhysicalDeviceSubgroupProperties>();
    printf("subgroup size=%u stages=%s ops=%s\n",
           sub.subgroupSize,
           vk::to_string(sub.supportedStages).c_str(),
           vk::to_string(sub.supportedOperations).c_str());

    auto buf = vk.createBuffer(
        64, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);
    memset(buf.mapped, 0, 64);

    vk::DescriptorSetLayoutBinding bind0{
        0, vk::DescriptorType::eStorageBuffer, 1,
        vk::ShaderStageFlagBits::eCompute};
    auto dsl = vk.device->createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo{}.setBindings(bind0));
    vk::DescriptorPoolSize ps{vk::DescriptorType::eStorageBuffer, 1};
    auto dpool = vk.device->createDescriptorPoolUnique(
        {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1, ps});
    vk::DescriptorSetLayout dslH = dsl.get();
    vk::DescriptorSetAllocateInfo dsai{};
    dsai.setDescriptorPool(dpool.get()).setSetLayouts(dslH);
    auto dsets = vk.device->allocateDescriptorSetsUnique(dsai);
    vk::DescriptorBufferInfo dbi{buf.buf.get(), 0, 64};
    vk::WriteDescriptorSet wset{dsets[0].get(), 0, 0,
                                vk::DescriptorType::eStorageBuffer,
                                {}, dbi};
    vk.device->updateDescriptorSets(wset, {});

    // subgroup ops need SPIR-V 1.3+ — vkmini's default targets 1.0
    shaderc::CompileOptions opts;
    opts.SetTargetEnvironment(shaderc_target_env_vulkan,
                              shaderc_env_version_vulkan_1_1);
    shaderc::Compiler cc;
    auto res = cc.CompileGlslToSpv(kComp, shaderc_compute_shader,
                                   "sg.comp", opts);
    if (res.GetCompilationStatus() !=
        shaderc_compilation_status_success)
        throw std::runtime_error(res.GetErrorMessage());
    std::vector<uint32_t> spv(res.cbegin(), res.cend());
    vk::ShaderModuleCreateInfo smci{};
    smci.setCode(spv);
    auto cs = vk.device->createShaderModuleUnique(smci);
    vk::PipelineLayoutCreateInfo plci{};
    plci.setSetLayouts(dslH);
    auto layout = vk.device->createPipelineLayoutUnique(plci);
    vk::PipelineShaderStageCreateInfo cstage{};
    cstage.setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(cs.get()).setPName("main");
    vk::ComputePipelineCreateInfo cpci{};
    cpci.setStage(cstage).setLayout(layout.get());
    auto cres = vk.device->createComputePipelineUnique({}, cpci);
    if (cres.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    auto pipe = std::move(cres.value);

    vk.oneTime([&](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.get());
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             layout.get(), 0, dsets[0].get(), {});
        c.dispatch(1, 1, 1);
    });

    auto* r = static_cast<uint32_t*>(buf.mapped);
    uint32_t sg = sub.subgroupSize;
    printf("sum=%u (want %u)  count=%u (want %u)  bcast=%u "
           "shuffled=%u sgSize=%u\n",
           r[0], sg * (sg + 1) / 2,
           r[1], (sg + 2) / 3,
           r[2], r[3], r[5]);
    bool ok = r[0] == sg * (sg + 1) / 2 &&   // sum 1..sg
              r[2] == 1 &&                  // lane0's value
              r[3] == 33 &&                 // lane32's value
              r[5] == sg;
    printf(ok ? "OK\n" : "MISMATCH\n");
    return ok ? 0 : 1;
}
