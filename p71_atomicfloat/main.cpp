// SPDX-License-Identifier: CC0-1.0
//
// VK_EXT_shader_atomic_float — atomic floating-point ops on SSBOs.
// atomicAdd() on a float* in a storage buffer: the classic GPU
// histogram/accumulation pattern without manual CAS loops.
//
// Demo: N invocations each generate a pseudo-random float and
// atomically accumulate it into a single counter plus a 16-bin
// histogram — verified against the CPU result (bit-order issues make
// exact float equality impossible for the sum, so we allow ULP-scale
// tolerance; bin counts are exact).
//
//   ./app --headless   (compute only; headless flag kept for shape)
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"
#include <random>

constexpr uint32_t kN = 1 << 20; // 1M atomic adds
constexpr uint32_t kBins = 16;

static const char* kComp = R"GLSL(
#version 460
#extension GL_EXT_shader_atomic_float : require

layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) buffer Out {
    float sum;
    uint  bins[16];
};

// same PRNG on both sides -> deterministic verification
uint hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16; return x;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    float v = float(hash(i) & 0xFFFFFFu) / float(0x1000000);
    atomicAdd(sum, v);                    // float atomic add!
    atomicAdd(bins[uint(v * 16.0) % 16], 1u);
}
)GLSL";

int main() {
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT af{};
    af.setShaderBufferFloat32AtomicAdd(true);
    vk::PhysicalDeviceDynamicRenderingFeatures dyn{};
    af.setPNext(&dyn);
    vk.createDevice({VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME}, &af);

    vkmini::Vk::Buffer out = vk.createBuffer(
        4 + kBins * 4, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memset(out.mapped, 0, 4 + kBins * 4);

    auto comp = vk.shader(kComp, shaderc_compute_shader, "h.comp");

    vk::DescriptorSetLayoutBinding b0{};
    b0.setBinding(0)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    vk::DescriptorSetLayoutCreateInfo sli{};
    sli.setBindings(b0);
    auto setLayout = vk.device->createDescriptorSetLayoutUnique(sli);
    vk::DescriptorPoolSize ps{vk::DescriptorType::eStorageBuffer, 1};
    vk::DescriptorPoolCreateInfo dpi{};
    dpi.setMaxSets(1).setPoolSizes(ps);
    auto dpool = vk.device->createDescriptorPoolUnique(dpi);
    vk::DescriptorSetAllocateInfo dai{};
    dai.setDescriptorPool(dpool.get())
        .setSetLayouts(setLayout.get());
    auto sets = vk.device->allocateDescriptorSets(dai);
    vk::DescriptorBufferInfo dbi{out.buf.get(), 0, VK_WHOLE_SIZE};
    vk::WriteDescriptorSet wr{};
    wr.setDstSet(sets[0])
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setBufferInfo(dbi);
    vk.device->updateDescriptorSets(wr, {});

    vk::PipelineLayoutCreateInfo li{};
    li.setSetLayouts(setLayout.get());
    auto layout = vk.device->createPipelineLayoutUnique(li);
    vk::ComputePipelineCreateInfo ci{};
    ci.stage.setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(comp.get())
        .setPName("main");
    ci.setLayout(layout.get());
    auto pres = vk.device->createComputePipelineUnique({}, ci);
    if (pres.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    auto pipe = std::move(pres.value);

    vk.oneTime([&](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.get());
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             layout.get(), 0, sets[0], {});
        c.dispatch(kN / 256, 1, 1);
    });

    // CPU reference: same hash, same inputs. Order of adds differs ->
    // compare with float tolerance, bins exactly.
    auto hashcpu = [](uint32_t x) {
        x ^= x >> 16; x *= 0x7feb352dU;
        x ^= x >> 15; x *= 0x846ca68bU;
        x ^= x >> 16; return x;
    };
    double refSum = 0;
    uint32_t refBins[kBins] = {};
    for (uint32_t i = 0; i < kN; ++i) {
        float v = float(hashcpu(i) & 0xFFFFFFu) / float(0x1000000);
        refSum += v;
        refBins[uint32_t(v * 16.0f) % 16]++;
    }
    float gpuSum = *static_cast<float*>(out.mapped);
    auto* gpuBins = static_cast<uint32_t*>(out.mapped) + 1;
    uint32_t binTotal = 0;
    bool binsOk = true;
    for (uint32_t i = 0; i < kBins; ++i) {
        binTotal += gpuBins[i];
        binsOk &= gpuBins[i] == refBins[i];
    }
    double relErr = fabs(gpuSum - refSum) / refSum;
    printf("sum: gpu=%.6f ref=%.6f relErr=%.2e | bins: %s "
           "(total=%u)\n",
           gpuSum, refSum, relErr, binsOk ? "exact" : "MISMATCH",
           binTotal);
    if (!binsOk || relErr > 1e-4 || binTotal != kN)
        throw std::runtime_error("verification failed");
    printf("done: %u float atomicAdds, no CAS loop\n", kN);
    return 0;
}
