// SPDX-License-Identifier: CC0-1.0
//
// Post 6: compute shaders + storage buffers. No window, no graphics
// pipeline — a compute dispatch fills a VkBuffer on the GPU (the
// CPU never touches the data), then we map a second host-visible
// buffer and read the results back.
//
// The shader computes out[i] = i*i + 7*i - 3  for i in 0..N, and a
// second element computes a prefix-sum-ish value per workgroup to
// show shared memory. check.py asserts the exact values.
//
//   ./app           runs once, prints summary + writes values.bin
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc
// (no GLFW needed — this post is purely offscreen)

#include "vkmini.hpp"

constexpr uint32_t kN = 1024;       // elements
constexpr uint32_t kGroups = 4;     // workgroups of 256 threads

static const char* kComp = R"GLSL(
#version 460
layout(local_size_x = 256) in;

// A storage buffer: the shader can both read and write it, at any
// index — unlike a uniform buffer it isn't constant.
layout(set = 0, binding = 0) buffer Out {
    float data[];   // runtime-sized array, kN floats
} buf;

// Shared memory is per-workgroup scratch — the classic pattern for
// reductions. Here each workgroup sums its 256 local results and
// writes one total per group into the tail of the buffer.
shared float partial[256];

layout(set = 0, binding = 1) buffer GroupSums {
    float sums[4];  // == kGroups (GLSL can't see C++ constants)
} gs;

void main() {
    uint i = gl_GlobalInvocationID.x;
    // main data pass
    float v = float(i) * float(i) + 7.0 * float(i) - 3.0;
    buf.data[i] = v;
    partial[gl_LocalInvocationID.x] = v;

    // memoryBarrier()+barrier(): all 256 threads must have written
    // their slot before thread 0 reads them.
    memoryBarrierShared();
    barrier();

    if (gl_LocalInvocationID.x == 0) {
        float s = 0.0;
        for (int k = 0; k < 256; ++k) s += partial[k];
        gs.sums[gl_WorkGroupID.x] = s;
    }
}
)GLSL";

int main() {
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    // No swapchain extension — a pure compute context.
    vk.createDevice({});

    // ---- buffers -------------------------------------------------
    // The GPU writes into a DEVICE_LOCAL buffer (fast); readback goes
    // through a second HOST_VISIBLE buffer via vkCmdCopyBuffer.
    vk::DeviceSize mainBytes = sizeof(float) * kN;
    vk::DeviceSize sumsBytes = sizeof(float) * kGroups;
    auto ssbo = vk.createBuffer(
        mainBytes,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    auto sums = vk.createBuffer(
        sumsBytes,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    auto readMain = vk.createBuffer(
        mainBytes, vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    auto readSums = vk.createBuffer(
        sumsBytes, vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);

    // ---- descriptors ----------------------------------------------
    // Same mechanics as the texture post, but type
    // eStorageBuffer instead of eCombinedImageSampler.
    std::array lbs{
        vk::DescriptorSetLayoutBinding{}
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute),
        vk::DescriptorSetLayoutBinding{}
            .setBinding(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute),
    };
    vk::DescriptorSetLayoutCreateInfo li{};
    li.setBindings(lbs);
    auto setLayout = vk.device->createDescriptorSetLayoutUnique(li);

    vk::DescriptorPoolSize ps{vk::DescriptorType::eStorageBuffer, 2};
    vk::DescriptorPoolCreateInfo pi{};
    pi.setMaxSets(1).setPoolSizes(ps);
    auto pool = vk.device->createDescriptorPoolUnique(pi);
    vk::DescriptorSetAllocateInfo ai{};
    ai.setDescriptorPool(pool.get()).setSetLayouts(setLayout.get());
    auto set = vk.device->allocateDescriptorSets(ai).front();

    std::array infos{
        vk::DescriptorBufferInfo{}.setBuffer(ssbo.buf.get())
            .setOffset(0).setRange(mainBytes),
        vk::DescriptorBufferInfo{}.setBuffer(sums.buf.get())
            .setOffset(0).setRange(sumsBytes),
    };
    std::array writes{
        vk::WriteDescriptorSet{}.setDstSet(set).setDstBinding(0)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(infos[0]),
        vk::WriteDescriptorSet{}.setDstSet(set).setDstBinding(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(infos[1]),
    };
    vk.device->updateDescriptorSets(writes, {});

    // ---- compute pipeline ------------------------------------------
    auto cs = vk.shader(kComp, shaderc_compute_shader, "fill.comp");
    vk::PipelineLayoutCreateInfo pli{};
    pli.setSetLayouts(setLayout.get());
    auto layout = vk.device->createPipelineLayoutUnique(pli);
    vk::PipelineShaderStageCreateInfo st{};
    st.setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(cs.get())
        .setPName("main");
    vk::ComputePipelineCreateInfo ci{};
    ci.setStage(st).setLayout(layout.get());
    auto res = vk.device->createComputePipelineUnique({}, ci);
    if (res.result != vk::Result::eSuccess)
        throw std::runtime_error("compute pipeline failed");
    auto pipe = std::move(res.value);

    // ---- dispatch + readback ----------------------------------------
    vk.oneTime([&](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.get());
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             layout.get(), 0, set, {});
        // kGroups workgroups x 256 threads = 1024 invocations,
        // exactly covering kN.
        c.dispatch(kGroups, 1, 1);

        // Barrier: storage writes must complete before the transfer
        // reads them — same ordering problem as always, now between
        // COMPUTE_SHADER and TRANSFER stages.
        vk::MemoryBarrier mb{};
        mb.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eTransfer, {},
                          mb, {}, {});
        vk::BufferCopy r1{0, 0, mainBytes};
        c.copyBuffer(ssbo.buf.get(), readMain.buf.get(), r1);
        vk::BufferCopy r2{0, 0, sumsBytes};
        c.copyBuffer(sums.buf.get(), readSums.buf.get(), r2);
    });
    // oneTime() ends with queue.waitIdle() -> results are mapped-ready.

    auto* out = static_cast<const float*>(readMain.mapped);
    auto* gsums = static_cast<const float*>(readSums.mapped);
    printf("out[0]=%g out[1]=%g out[255]=%g out[1023]=%g\n", out[0],
           out[1], out[255], out[1023]);
    printf("group sums: %g %g %g %g\n", gsums[0], gsums[1], gsums[2],
           gsums[3]);

    // Raw dump for check.py.
    FILE* f = fopen("values.bin", "wb");
    fwrite(out, sizeof(float), kN, f);
    fwrite(gsums, sizeof(float), kGroups, f);
    fclose(f);
    printf("wrote values.bin\n");
    return 0;
}
