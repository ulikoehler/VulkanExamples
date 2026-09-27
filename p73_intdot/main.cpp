// SPDX-License-Identifier: CC0-1.0
//
// VK_KHR_shader_integer_dot_product — dot4a in a compute shader.
// dot() on i8/u8 vectors compiles to a single DP4A-style instruction
// on hardware that exposes it — the workhorse of int8 inference.
//
// Demo: two int8 vectors per invocation -> dot product, verified
// exactly (int math!) against the CPU, plus a rough throughput line.
//
//   ./app
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"
#include <chrono>
#include <random>

constexpr uint32_t kN = 1 << 22; // 4M dot products

// uvec4 * i8vec4 is not a thing; the GLSL idiom is u8x4 packed in a
// single uint -> dot(i8x4(a), i8x4(b)) via the 8-bit vector types
// introduced by GL_EXT_shader_explicit_arithmetic_types_int8.
static const char* kComp = R"GLSL(
#version 460
#extension GL_EXT_integer_dot_product : require

layout(local_size_x = 256) in;

// packed 4x int8 per uint32 word -- no 8-bit storage needed
layout(set = 0, binding = 0, std430) readonly buffer A {
    int a[];
};
layout(set = 0, binding = 1, std430) readonly buffer B {
    int b[];
};
layout(set = 0, binding = 2, std430) writeonly buffer Out {
    int dot_[];   // `dot` is a reserved name
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    dot_[i] = dotPacked4x8EXT(a[i], b[i]);   // one SDOT/DP4A
}
)GLSL";

int main() {
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    // shaderInt8 (for i8vec4) lives in the KHR_shader_float16_int8
    // feature struct, core since 1.2.
    vk::PhysicalDeviceShaderIntegerDotProductFeatures idp{};
    idp.setShaderIntegerDotProduct(true);
    vk::PhysicalDeviceDynamicRenderingFeatures dyn{};
    vk::PhysicalDeviceShaderFloat16Int8Features i8{};
    idp.setPNext(&dyn);
    dyn.setPNext(&i8);
    vk.createDevice(
        {VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME}, &idp);

    // Which packed dot instructions does the driver actually have?
    auto props = vk.phys.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceShaderIntegerDotProductProperties>();
    auto& p = props.get<
        vk::PhysicalDeviceShaderIntegerDotProductProperties>();
    printf("integerDotProduct8BitUnsignedAccelerated=%d "
           "signedAccelerated=%d\n",
           p.integerDotProduct8BitUnsignedAccelerated,
           p.integerDotProduct8BitSignedAccelerated);

    auto inB = vk.createBuffer(
        vk::DeviceSize(kN) * 4,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    auto in = vk.createBuffer(
        vk::DeviceSize(kN) * 4,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    auto out = vk.createBuffer(
        vk::DeviceSize(kN) * 4,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-128, 127);
    auto* inp = static_cast<uint8_t*>(in.mapped);   // a[] packed
    auto* inb = static_cast<uint8_t*>(inB.mapped);  // b[] packed
    for (uint32_t i = 0; i < kN; ++i) {
        for (int k = 0; k < 4; ++k) {
            inp[i * 4 + k] = uint8_t(int8_t(dist(rng)));
            inb[i * 4 + k] = uint8_t(int8_t(dist(rng)));
        }
    }

    auto comp = vk.shader(kComp, shaderc_compute_shader, "d.comp");

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
    vk::DescriptorBufferInfo dbi0{in.buf.get(), 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo dbi1{inB.buf.get(), 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo dbi2{out.buf.get(), 0, VK_WHOLE_SIZE};
    vk::WriteDescriptorSet wr[3];
    wr[0].setDstSet(sets[0])
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setBufferInfo(dbi0);
    wr[1].setDstSet(sets[0])
        .setDstBinding(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setBufferInfo(dbi1);
    wr[2].setDstSet(sets[0])
        .setDstBinding(2)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setBufferInfo(dbi2);
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

    auto t0 = std::chrono::steady_clock::now();
    vk.oneTime([&](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.get());
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             layout.get(), 0, sets[0], {});
        c.dispatch(kN / 256, 1, 1);
    });
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0)
                    .count();

    auto* gout = static_cast<const int32_t*>(out.mapped);
    size_t bad = 0;
    for (uint32_t i = 0; i < kN; ++i) {
        int ref = 0;
        for (int k = 0; k < 4; ++k)
            ref += int8_t(inp[i * 4 + k]) * int8_t(inb[i * 4 + k]);
        if (gout[i] != ref && bad++ < 5)
            printf("  mismatch @%u: gpu=%d ref=%d\n", i, gout[i], ref);
    }
    printf("%u int8x4 dots in %.2f ms -> %.1f Gdot/s | bad=%zu\n", kN,
           ms, kN / ms / 1e6, bad);
    if (bad)
        throw std::runtime_error("dp4a mismatch");
    return 0;
}
