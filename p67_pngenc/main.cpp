// SPDX-License-Identifier: CC0-1.0
//
// PNG encoding in Vulkan compute shaders. Two dispatches:
//   1. per-scanline adaptive filter (None/Sub/Up/Average/Paeth by min
//      sum of |signed filtered byte|) + per-row Adler-32 partials
//   2. zlib stored-block framing: [0x78 0x01] + [BFINAL|LEN|NLEN] +
//      payload + Adler-32, one invocation per output u32 word
// The CPU only writes the PNG container (IHDR/IDAT/IEND + CRC32) and
// folds the per-row Adler partials — it never touches pixel data.
//
// PNG mandates zlib-wrapped DEFLATE; stored (uncompressed) DEFLATE
// blocks are fully spec-legal, so this encoder produces PNGs any
// decoder accepts — lossless, unlike p51's JPEG.
//
//   ./app out.png     encode a procedural 640x480 pattern -> PNG
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc

#include "vkmini.hpp"
#include <fstream>

static constexpr uint32_t W = 640, H = 480;
static constexpr uint32_t ROWBYTES = W * 4;      // RGBA8
static constexpr uint32_t D = H * (ROWBYTES + 1);// filtered stream size
static constexpr uint32_t BLK = 65535;           // max stored-block payload
static constexpr uint32_t NBLK = (D + BLK - 1) / BLK;
static constexpr uint32_t OUTSZ = 2 + NBLK * 5 + D + 4; // zlib stream

// dispatch 1: one invocation per scanline — try all 5 filters, keep
// the lowest-cost one, emit filter byte + filtered row + adler parts
static const char* kFilter = R"GLSL(
#version 460
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) readonly buffer Src { uint src[]; };
layout(set = 0, binding = 1) writeonly buffer Filt { uint filt[]; };
// one unsized array only per SSBO — meta layout is fixed:
// [0..H) filter types, [H..2H) adler s1 partials, [2H..3H) s2
layout(set = 0, binding = 2) writeonly buffer Meta { uint meta[]; };
layout(push_constant) uniform P { uint w; uint h; uint rb; };

uint srcByte(uint i) { return (src[i >> 2] >> ((i & 3u) * 8u)) & 0xffu; }

int paeth(int a, int b, int c) {
    int p = a + b - c;
    int pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
    if (pa <= pb && pa <= pc) return a;
    return pb <= pc ? b : c;
}

void main() {
    uint y = gl_GlobalInvocationID.x;
    if (y >= h) return;
    uint bpp = 4u;
    // evaluate all five filters; track best by signed-abs sum
    uint best = 0; int bestCost = 0x7fffffff;
    int rowCost[5];
    for (uint f = 0u; f < 5u; ++f) {
        int cost = 0;
        for (uint i = 0u; i < rb; ++i) {
            int x = int(srcByte(y * rb + i));
            int a = i >= bpp ? int(srcByte(y * rb + i - bpp)) : 0;
            int b = y > 0u   ? int(srcByte((y - 1u) * rb + i)) : 0;
            int c = (y > 0u && i >= bpp)
                        ? int(srcByte((y - 1u) * rb + i - bpp)) : 0;
            int v;
            if (f == 0u)      v = x;
            else if (f == 1u) v = x - a;
            else if (f == 2u) v = x - b;
            else if (f == 3u) v = x - ((a + b) >> 1);
            else              v = x - paeth(a, b, c);
            // PNG heuristic: sum of |signed filtered byte|
            int sb = v & 255;
            cost += sb > 127 ? 256 - sb : sb;
        }
        rowCost[f] = cost;
        if (cost < bestCost) { bestCost = cost; best = f; }
    }
    meta[y] = best;
    // emit (accumulate adler from locals — filt is writeonly)
    uint base = y * (rb + 1u);
    filt[base] = best;
    uint s1p = best, s2p = best * (rb + 1u); // filter byte = b_0,
                                           // weight n = rb+1
    for (uint i = 0u; i < rb; ++i) {
        int x = int(srcByte(y * rb + i));
        int a = i >= bpp ? int(srcByte(y * rb + i - bpp)) : 0;
        int b = y > 0u   ? int(srcByte((y - 1u) * rb + i)) : 0;
        int c = (y > 0u && i >= bpp)
                    ? int(srcByte((y - 1u) * rb + i - bpp)) : 0;
        int v;
        if (best == 0u)      v = x;
        else if (best == 1u) v = x - a;
        else if (best == 2u) v = x - b;
        else if (best == 3u) v = x - ((a + b) >> 1);
        else                 v = x - paeth(a, b, c);
        uint fv = uint(v) & 0xffu;
        filt[base + 1u + i] = fv;
        // adler partials: s1 = sum(b_i), s2 = sum((n-i)*b_i) —
        // sequential prefix folding happens on the CPU combine
        s1p += fv;
        s2p += fv * (rb - i);
    }
    meta[h + y] = s1p;
    meta[2u * h + y] = s2p;
}
)GLSL";

// dispatch 2: one invocation per output u32 word — zlib stream
// layout: [0x78 0x01] [blk0: hdr5 + data] ... [adler4]
static const char* kFrame = R"GLSL(
#version 460
layout(local_size_x = 64) in;
layout(set = 0, binding = 1) readonly buffer Filt { uint filt[]; };
layout(set = 0, binding = 3) writeonly buffer Out { uint outb[]; };
layout(push_constant) uniform Q { uint dsz; };

uint outByte(uint i) {
    if (i == 0u) return 0x78u;      // zlib CMF: deflate, 32K window
    if (i == 1u) return 0x01u;      // FLG: no preset dict (check byte
                                    // fixed on CPU via FCHECK — 0x7801
                                    // is already valid: 0x7801%31==0)
    uint j = i - 2u;
    const uint SPAN = 65535u + 5u;
    uint blk = j / SPAN;
    uint off = j % SPAN;
    uint payload = min(65535u, dsz - blk * 65535u);
    if (blk * 65535u >= dsz) return 0u; // adler tail — CPU writes it
    if (off < 5u) {
        bool last = (blk + 1u) * 65535u >= dsz;
        if (off == 0u) return last ? 1u : 0u;   // BFINAL|BTYPE=00
        if (off == 1u) return payload & 0xffu;
        if (off == 2u) return payload >> 8;
        if (off == 3u) return (~payload) & 0xffu;
        return (~payload) >> 8 & 0xffu;
    }
    return filt[blk * 65535u + (off - 5u)];
}

void main() {
    uint wi = gl_GlobalInvocationID.x;
    // outWord count = ceil(OUTSZ/4) — caller bounds via push? use dsz
    uint outsz = 2u + ((dsz + 65534u) / 65535u) * 5u + dsz + 4u;
    if (wi * 4u >= outsz) return;
    uint w0 = outByte(wi * 4u), w1 = outByte(wi * 4u + 1u);
    uint w2 = outByte(wi * 4u + 2u), w3 = outByte(wi * 4u + 3u);
    outb[wi] = w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u);
}
)GLSL";

// ---------------- PNG container (host) ----------------
static uint32_t crc32Table[256];
static void initCrc() {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k)
            c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc32Table[i] = c;
    }
}
static uint32_t crc32(const uint8_t* p, size_t n, uint32_t c = 0xFFFFFFFFu) {
    while (n--) c = crc32Table[(c ^ *p++) & 0xff] ^ (c >> 8);
    return c;
}
static void chunk(std::ofstream& f, const char* type,
                  const uint8_t* data, uint32_t len) {
    auto be = [&](uint32_t v) {
        uint8_t b[4] = {uint8_t(v >> 24), uint8_t(v >> 16),
                        uint8_t(v >> 8), uint8_t(v)};
        f.write(reinterpret_cast<char*>(b), 4);
    };
    be(len);
    f.write(type, 4);
    f.write(reinterpret_cast<const char*>(data), len);
    uint32_t c = crc32(reinterpret_cast<const uint8_t*>(type), 4);
    c = ~crc32(data, len, c);
    be(c);
}

int main(int argc, char** argv) {
    const char* png = "out.png";
    if (argc > 1) png = argv[1];

    // procedural test pattern (CPU) — gradients + sharp edges exercise
    // every PNG filter
    std::vector<uint8_t> px(W * H * 4);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            uint8_t* p = &px[(y * W + x) * 4];
            p[0] = uint8_t(x * 255 / W);                    // R ramp
            p[1] = uint8_t(y * 255 / H);                    // G ramp
            p[2] = uint8_t(((x / 40) ^ (y / 40)) & 1)
                       ? 220 : 40;                          // blocks
            p[3] = 255;
            // hard vertical edge mid-image -> Sub filter territory
            if (x >= W / 2 && x < W / 2 + 8) {
                p[0] = 255; p[1] = 0; p[2] = 0;
            }
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
    // per-byte words (x4 memory, race-free byte addressing)
    auto filtBuf = vk.createBuffer(vk::DeviceSize(D) * 4,
        vk::BufferUsageFlagBits::eStorageBuffer, memHost, true);
    auto metaBuf = vk.createBuffer(vk::DeviceSize(H) * 12,
        vk::BufferUsageFlagBits::eStorageBuffer, memHost, true);
    auto outBuf = vk.createBuffer(vk::DeviceSize(OUTSZ + 8),
        vk::BufferUsageFlagBits::eStorageBuffer, memHost, true);

    vk::DescriptorSetLayoutBinding binds[4];
    for (int i = 0; i < 4; ++i)
        binds[i].setBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    vk::DescriptorSetLayoutCreateInfo dsli{};
    dsli.setBindings(binds);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dsli);
    vk::DescriptorPoolSize psz{vk::DescriptorType::eStorageBuffer, 4};
    vk::DescriptorPoolCreateInfo dpi{};
    dpi.setMaxSets(1).setPoolSizes(psz);
    auto pool = vk.device->createDescriptorPoolUnique(dpi);
    vk::DescriptorSetAllocateInfo dai{};
    dai.setDescriptorPool(pool.get()).setSetLayouts(dsl.get());
    auto dsets = vk.device->allocateDescriptorSetsUnique(dai);
    auto& dset = dsets.front();
    vk::DescriptorBufferInfo infos[4] = {
        {srcBuf.buf.get(), 0, VK_WHOLE_SIZE},
        {filtBuf.buf.get(), 0, VK_WHOLE_SIZE},
        {metaBuf.buf.get(), 0, VK_WHOLE_SIZE},
        {outBuf.buf.get(), 0, VK_WHOLE_SIZE}};
    for (int i = 0; i < 4; ++i) {
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

    auto mkPipe = [&](const char* src) {
        auto mod = vk.shader(src, shaderc_compute_shader, "c.comp");
        vk::ComputePipelineCreateInfo ci{};
        ci.stage.setStage(vk::ShaderStageFlagBits::eCompute)
            .setModule(mod.get()).setPName("main");
        ci.setLayout(layout.get());
        auto r = vk.device->createComputePipelineUnique({}, ci);
        if (r.result != vk::Result::eSuccess)
            throw std::runtime_error("compute pipeline failed");
        return std::make_pair(std::move(r.value), std::move(mod));
    };
    auto [pipeF, modF] = mkPipe(kFilter);
    auto [pipeW, modW] = mkPipe(kFrame);

    vk.oneTime([&](vk::CommandBuffer c) {
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             layout.get(), 0, dset.get(), {});
        uint32_t pc[3] = {W, H, ROWBYTES};
        c.pushConstants<uint32_t>(layout.get(),
                                  vk::ShaderStageFlagBits::eCompute,
                                  0, pc);
        c.bindPipeline(vk::PipelineBindPoint::eCompute, pipeF.get());
        c.dispatch((H + 63) / 64, 1, 1);
        vk::MemoryBarrier mb{};
        mb.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead |
                              vk::AccessFlagBits::eShaderWrite);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eComputeShader,
                          {}, mb, {}, {});
        c.bindPipeline(vk::PipelineBindPoint::eCompute, pipeW.get());
        uint32_t pc2[3] = {D, 0, 0};   // frame shader wants dsz
        c.pushConstants<uint32_t>(layout.get(),
                                  vk::ShaderStageFlagBits::eCompute,
                                  0, pc2);
        c.dispatch((OUTSZ / 4 + 63) / 64 + 1, 1, 1);
    });

    // ---- CPU: fold adler partials, patch tail, write PNG ----------
    auto* ftype = static_cast<uint32_t*>(metaBuf.mapped);
    auto* s1 = ftype + H;
    auto* s2 = ftype + 2 * H;
    // adler32_combine per row: s1 += s1sum; s2 += (n + s2sum) +
    // n*(s1_prev - 1)   [GPU stored raw sums, fresh-start s1 = 1]
    uint64_t a = 1, b = 0;
    for (uint32_t y = 0; y < H; ++y) {
        uint64_t n = ROWBYTES + 1;
        b = (b + n + s2[y] + n * (a - 1)) % 65521;
        a = (a + s1[y]) % 65521;
    }
    uint32_t adler = uint32_t((b << 16) | a);
    auto* out = static_cast<uint8_t*>(outBuf.mapped);
    out[OUTSZ - 4] = uint8_t(adler >> 24);
    out[OUTSZ - 3] = uint8_t(adler >> 16);
    out[OUTSZ - 2] = uint8_t(adler >> 8);
    out[OUTSZ - 1] = uint8_t(adler);

    uint32_t fc[5] = {};
    for (uint32_t y = 0; y < H; ++y) fc[ftype[y] & 7]++;
    printf("filters used: none=%u sub=%u up=%u avg=%u paeth=%u\n",
           fc[0], fc[1], fc[2], fc[3], fc[4]);

    initCrc();
    std::ofstream f(png, std::ios::binary);
    const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    f.write(reinterpret_cast<const char*>(sig), 8);
    uint8_t ihdr[13] = {
        uint8_t(W >> 24), uint8_t(W >> 16), uint8_t(W >> 8),
        uint8_t(W), uint8_t(H >> 24), uint8_t(H >> 16),
        uint8_t(H >> 8), uint8_t(H), 8, 6, 0, 0, 0};
    chunk(f, "IHDR", ihdr, 13);
    chunk(f, "IDAT", out, OUTSZ);
    chunk(f, "IEND", nullptr, 0);
    f.close();
    printf("wrote %s (%u bytes IDAT, %u stored blocks)\n", png,
           OUTSZ, NBLK);
    return 0;
}
