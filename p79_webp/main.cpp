// SPDX-License-Identifier: CC0-1.0
//
// Lossless WebP (VP8L) encoding in a Vulkan compute shader.
//
// The VP8L bitstream has no "stored" mode — every pixel is Huffman
// coded. The trick that makes it GPU-friendly: choose a flat canonical
// code where all 256 literals get 8-bit codes. Then every pixel costs
// exactly 24 bits (green + red + blue codes; alpha is a single-symbol
// tree = 0 bits) at the fixed bit offset  headerBits + i*24 — a
// perfectly parallel write, no prefix sums, no LZ77, no transforms.
//
//   CPU:  writes the ~1091-bit VP8L header (signatures, sizes, five
//         prefix-code descriptions) into the output buffer, then the
//         RIFF/WEBP container around the finished stream.
//   GPU:  one invocation per pixel — rev8(g)|rev8(r)<<8|rev8(b)<<16
//         atomically OR-ed into the shared u32 stream words.
//
//   ./app out.webp    encode a procedural 640x480 pattern -> WebP
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc

#include "vkmini.hpp"
#include <fstream>

static constexpr uint32_t W = 640, H = 480;
// header size in bits — must match the bit-exact emission below;
// 40 (sig+w+h+alpha+ver) +1 (no transforms) +1 (no color cache)
// +1 (no meta huffman) + trees 322+298+298+11+11 = 983
static constexpr uint32_t HEADER_BITS = 983;
static constexpr uint32_t STREAM_BYTES = (HEADER_BITS + W * H * 24 + 7) / 8;

// compute: one invocation per pixel -> 24 bits into the shared stream
static const char* kEmit = R"GLSL(
#version 460
layout(local_size_x = 128) in;
layout(set = 0, binding = 0) readonly buffer Src { uint src[]; };
layout(set = 0, binding = 1) buffer Out { uint outb[]; };
layout(push_constant) uniform P { uint w; uint h; uint bitBase; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= w * h) return;
    uint px = src[i];                       // RGBA8 little-endian
    // VP8L emits green, red, blue, alpha per pixel; codes are written
    // MSB-first into the LSB-first bitstream -> bit-reverse each byte
    uint g = bitfieldReverse((px >> 8) & 0xffu) >> 24;
    uint r = bitfieldReverse(px & 0xffu) >> 24;
    uint b = bitfieldReverse((px >> 16) & 0xffu) >> 24;
    uint bits = g | (r << 8) | (b << 16);   // alpha tree: 0 bits
    uint bitpos = bitBase + i * 24u;
    uint wi = bitpos >> 5, sh = bitpos & 31u;
    atomicOr(outb[wi], bits << sh);         // may straddle a word
    if (sh > 8u) atomicOr(outb[wi + 1u], bits >> (32u - sh));
}
)GLSL";

// ---------------- VP8L header (host, LSB-first bits) ---------------
struct BitWriter {
    uint32_t* w; uint32_t bit = 0;
    void put(uint32_t v, int n) {           // n bits, LSB-first
        for (int k = 0; k < n; ++k, ++bit)
            w[bit >> 5] |= ((v >> k) & 1u) << (bit & 31);
    }
};

// Emit a flat full prefix code: literals 0..litN-1 all get 8-bit
// codes (Kraft-complete: 256 * 2^-8 = 1), remaining symbols absent.
static void emitFlatTree(BitWriter& bw, int alphabet, int litN) {
    bw.put(0, 1);                           // not a simple code
    // code-length code: only CL symbols {0, 8} get nonzero lengths.
    // kCodeLengthCodeOrder: 8 sits at index 11 -> send 12 entries
    // (num_code_lengths - 4 = 8), each a 3-bit length value.
    static const int order[19] = {17, 18, 0, 1, 2, 3, 4, 5, 16, 6,
                                  7, 8, 9, 10, 11, 12, 13, 14, 15};
    bw.put(8, 4);                           // num_code_lengths = 12
    for (int i = 0; i < 12; ++i)
        bw.put(order[i] == 0 || order[i] == 8 ? 1u : 0u, 3);
    bw.put(0, 1);                           // no max_symbol truncation
    // CL codes: symbol 8 -> code '1', symbol 0 -> code '0' (1 bit)
    for (int i = 0; i < alphabet; ++i) bw.put(i < litN ? 1u : 0u, 1);
}

// simple-code tree with exactly one symbol (0 bits per read)
static void emitSingleTree(BitWriter& bw, uint32_t sym) {
    bw.put(1, 1);                           // simple code
    bw.put(0, 1);                           // num_symbols - 1 = 0
    bw.put(1, 1);                           // 8-bit symbol follows
    bw.put(sym, 8);
}

int main(int argc, char** argv) {
    const char* out = "out.webp";
    if (argc > 1) out = argv[1];

    // procedural test pattern (alpha stays 255 — the single-symbol
    // alpha tree only encodes opaque pixels)
    std::vector<uint8_t> px(W * H * 4);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            uint8_t* p = &px[(y * W + x) * 4];
            p[0] = uint8_t(x * 255 / W);                    // R ramp
            p[1] = uint8_t(y * 255 / H);                    // G ramp
            p[2] = uint8_t(((x / 40) ^ (y / 40)) & 1)
                       ? 220 : 40;                          // blocks
            p[3] = 255;
            if (x >= W / 2 && x < W / 2 + 8)
                p[0] = 255, p[1] = 0, p[2] = 0;
        }

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice();

    auto memHost = vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent;
    auto srcBuf = vk.createBuffer(px.size(),
        vk::BufferUsageFlagBits::eStorageBuffer, memHost, true);
    memcpy(srcBuf.mapped, px.data(), px.size());
    auto outBuf = vk.createBuffer(vk::DeviceSize(STREAM_BYTES + 4),
        vk::BufferUsageFlagBits::eStorageBuffer, memHost, true);

    // zero the stream + write the VP8L header bits before dispatch
    auto* words = static_cast<uint32_t*>(outBuf.mapped);
    memset(words, 0, STREAM_BYTES + 4);
    BitWriter bw{words};
    bw.put(0x2f, 8);                        // VP8L signature
    bw.put(W - 1, 14); bw.put(H - 1, 14);   // dimensions - 1
    bw.put(1, 1);                           // alpha is used
    bw.put(0, 3);                           // version
    bw.put(0, 1);                           // no transforms
    bw.put(0, 1);                           // no color cache
    bw.put(0, 1);                           // no meta huffman image
    emitFlatTree(bw, 280, 256);             // green (literals+lengths)
    emitFlatTree(bw, 256, 256);             // red
    emitFlatTree(bw, 256, 256);             // blue
    emitSingleTree(bw, 255);                // alpha: only 255
    emitSingleTree(bw, 0);                  // distance: unused
    assert(bw.bit == HEADER_BITS);

    vk::DescriptorSetLayoutBinding binds[2];
    for (int i = 0; i < 2; ++i)
        binds[i].setBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    vk::DescriptorSetLayoutCreateInfo dsli{};
    dsli.setBindings(binds);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dsli);
    vk::DescriptorPoolSize psz{vk::DescriptorType::eStorageBuffer, 2};
    vk::DescriptorPoolCreateInfo dpi{};
    dpi.setMaxSets(1).setPoolSizes(psz);
    auto pool = vk.device->createDescriptorPoolUnique(dpi);
    vk::DescriptorSetAllocateInfo dai{};
    dai.setDescriptorPool(pool.get()).setSetLayouts(dsl.get());
    auto dsets = vk.device->allocateDescriptorSetsUnique(dai);
    auto& dset = dsets.front();
    vk::DescriptorBufferInfo infos[2] = {
        {srcBuf.buf.get(), 0, VK_WHOLE_SIZE},
        {outBuf.buf.get(), 0, VK_WHOLE_SIZE}};
    for (int i = 0; i < 2; ++i) {
        vk::WriteDescriptorSet wr{};
        wr.setDstSet(dset.get()).setDstBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1).setPBufferInfo(&infos[i]);
        vk.device->updateDescriptorSets(wr, {});
    }
    vk::PushConstantRange pcr{vk::ShaderStageFlagBits::eCompute, 0, 12};
    vk::PipelineLayoutCreateInfo pli{};
    pli.setSetLayouts(dsl.get()).setPushConstantRanges(pcr);
    auto layout = vk.device->createPipelineLayoutUnique(pli);
    auto mod = vk.shader(kEmit, shaderc_compute_shader, "emit.comp");
    vk::ComputePipelineCreateInfo ci{};
    ci.stage.setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(mod.get()).setPName("main");
    ci.setLayout(layout.get());
    auto pipe = vk.device->createComputePipelineUnique({}, ci);
    if (pipe.result != vk::Result::eSuccess)
        throw std::runtime_error("compute pipeline failed");

    vk.oneTime([&](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute,
                       pipe.value.get());
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             layout.get(), 0, dset.get(), {});
        uint32_t pc[3] = {W, H, HEADER_BITS};
        c.pushConstants<uint32_t>(layout.get(),
                                  vk::ShaderStageFlagBits::eCompute,
                                  0, pc);
        c.dispatch((W * H + 127) / 128, 1, 1);
    });

    // ---- CPU: RIFF/WEBP/VP8L container ----------------------------
    const uint8_t* stream = static_cast<const uint8_t*>(outBuf.mapped);
    uint32_t riffSize = 4 + 8 + STREAM_BYTES + (STREAM_BYTES & 1);
    std::ofstream f(out, std::ios::binary);
    auto tag = [&](const char* t) { f.write(t, 4); };
    auto le32 = [&](uint32_t v) {
        uint8_t b[4] = {uint8_t(v), uint8_t(v >> 8),
                        uint8_t(v >> 16), uint8_t(v >> 24)};
        f.write(reinterpret_cast<char*>(b), 4);
    };
    tag("RIFF"); le32(riffSize); tag("WEBP");
    tag("VP8L"); le32(STREAM_BYTES);
    f.write(reinterpret_cast<const char*>(stream), STREAM_BYTES);
    if (STREAM_BYTES & 1) f.put(0);         // RIFF pad byte
    f.close();
    printf("wrote %s (%ux%u, %u-byte VP8L stream, %u-bit header)\n",
           out, W, H, STREAM_BYTES, HEADER_BITS);
    return 0;
}
