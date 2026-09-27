// SPDX-License-Identifier: CC0-1.0
//
// PNG decoding in Vulkan compute shaders — a REAL inflate, not a
// zlib call: the first dispatch is a single-invocation DEFLATE
// decoder (stored + fixed + dynamic Huffman blocks, LZ77 back-
// references), the second un-filters scanlines in parallel
// (None/Sub/Up/Average/Paeth), the third expands to RGBA8.
// The CPU only parses PNG chunks and concatenates IDAT payloads.
//
//   ./app in.png out.ppm
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc

#include "vkmini.hpp"
#include <fstream>

// ==================== inflate shader ====================
// puff.c-style canonical decoder; one invocation walks the stream
// sequentially (DEFLATE is inherently serial).
static const char* kInflate = R"GLSL(
#version 460
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) readonly buffer In { uint inw[]; };
layout(set = 0, binding = 1) buffer Raw { uint raw[]; }; // byte-words
layout(set = 0, binding = 2) buffer Meta { uint outLen; int err;
                                          uint cap; };
layout(push_constant) uniform P { uint inBytes; };

// all stream state in one struct threaded inout through helpers —
// keeps ordering explicit instead of relying on global variables
struct Inf { uint inPos, bitbuf, bitcnt, outPos; };

uint getByte(inout Inf s) {
    uint i = s.inPos++;
    if (i >= inBytes) return 0u;          // past end: safe zero
    return (inw[i >> 2] >> ((i & 3u) * 8u)) & 0xffu;
}
uint bits(inout Inf s, uint n) {
    while (s.bitcnt < n) {
        s.bitbuf |= getByte(s) << s.bitcnt; s.bitcnt += 8u;
    }
    uint r = s.bitbuf & ((1u << n) - 1u);
    s.bitbuf >>= n; s.bitcnt -= n; return r;
}
void putByte(inout Inf s, uint b) {
    if (s.outPos < cap) raw[s.outPos] = b;  // corrupt stream guard
    s.outPos++;
}

// canonical huffman: cnt[l] = #codes of length l, sym[] sorted
int decode(inout Inf s, uint cnt[16], uint sym[288]) {
    uint code = 0u, first = 0u, index = 0u;
    for (int len = 1; len <= 15; ++len) {
        code |= bits(s, 1u);
        uint count = cnt[len];
        if (code < first + count)
            return int(sym[index + code - first]);
        index += count;
        first = (first + count) << 1;
        code <<= 1;
    }
    err = 5; return -1;
}

// build canonical tables from code lengths (0 = absent)
void buildHuff(uint lens[320], int n,
               out uint cnt[16], out uint sym[288]) {
    for (int i = 0; i < 16; ++i) cnt[i] = 0u;
    for (int i = 0; i < n; ++i) cnt[lens[i]]++;
    cnt[0] = 0u; // incomplete codes allowed (unused lens)
    uint offs[16];
    offs[1] = 0u;
    for (int i = 1; i < 15; ++i) offs[i + 1] = offs[i] + cnt[i];
    for (int i = 0; i < n; ++i)
        if (lens[i] != 0u) sym[offs[lens[i]]++] = uint(i);
}

const uint lbase[29] = {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,
    51,59,67,83,99,115,131,163,195,227,258};
const uint lext[29] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,
    5,5,5,5,0};
const uint dbase[30] = {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,
    385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
const uint dext[30] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,
    10,11,11,12,12,13,13};
const uint clOrder[19] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,
    15};

void inflate() {
    Inf s;
    s.inPos = 0u; s.bitbuf = 0u; s.bitcnt = 0u; s.outPos = 0u;
    // zlib header
    uint cmf = getByte(s), flg = getByte(s);
    if ((cmf & 15u) != 8u || (flg & 32u) != 0u) { err = 2; return; }
    uint litCnt[16], litSym[288], dCnt[16], dSym[288];
    uint lens[320];
    uint last = 0u;
    while (last == 0u && err == 0) {
        last = bits(s, 1u);
        uint type = bits(s, 2u);
        if (type == 0u) {                       // stored
            s.bitbuf = 0u; s.bitcnt = 0u;       // resync to byte
            // | operand order is unspecified — sequence explicitly
            uint lenLo = getByte(s); uint lenHi = getByte(s);
            uint nlLo = getByte(s);  uint nlHi = getByte(s);
            uint len = lenLo | (lenHi << 8);
            uint nlen = nlLo | (nlHi << 8);
            if (len != (nlen ^ 0xffffu)) { err = 3; return; }
            for (uint i = 0u; i < len; ++i) putByte(s, getByte(s));
        } else if (type == 1u || type == 2u) {
            uint nlit, ndist;
            if (type == 1u) {                   // fixed tables
                for (uint i = 0u; i < 288u; ++i)
                    lens[i] = i < 144u ? 8u : i < 256u ? 9u
                              : i < 280u ? 7u : 8u;
                buildHuff(lens, 288, litCnt, litSym);
                for (uint i = 0u; i < 32u; ++i) lens[i] = 5u;
                buildHuff(lens, 30, dCnt, dSym);
                nlit = 288u; ndist = 30u;
            } else {                            // dynamic tables
                nlit = bits(s, 5u) + 257u;
                ndist = bits(s, 5u) + 1u;
                uint ncl = bits(s, 4u) + 4u;
                for (uint i = 0u; i < 19u; ++i) lens[i] = 0u;
                for (uint i = 0u; i < ncl; ++i)
                    lens[clOrder[i]] = bits(s, 3u);
                uint clCnt[16], clSym[288];
                buildHuff(lens, 19, clCnt, clSym);
                uint n = 0u;
                while (n < nlit + ndist) {
                    int sy = decode(s, clCnt, clSym);
                    if (sy < 0) return;
                    if (sy < 16) lens[n++] = uint(sy);
                    else {
                        uint rep, val;
                        if (sy == 16) { rep = 3u + bits(s, 2u);
                            val = n > 0u ? lens[n - 1u] : 0u; }
                        else if (sy == 17) { rep = 3u + bits(s, 3u);
                            val = 0u; }
                        else { rep = 11u + bits(s, 7u); val = 0u; }
                        while (rep-- > 0u && n < 320u)
                            lens[n++] = val;
                    }
                }
                if (n != nlit + ndist) { err = 6; return; }
                // lens[0..nlit) = lit lengths, rest = dist lengths;
                // GLSL array params copy — reuse one scratch copy
                uint lcopy[320];
                for (uint i = 0u; i < nlit; ++i) lcopy[i] = lens[i];
                buildHuff(lcopy, int(nlit), litCnt, litSym);
                for (uint i = 0u; i < ndist; ++i)
                    lcopy[i] = lens[nlit + i];
                buildHuff(lcopy, int(ndist), dCnt, dSym);
            }
            // decode symbols
            for (;;) {
                int sy = decode(s, litCnt, litSym);
                if (sy < 0) return;
                if (sy < 256) putByte(s, uint(sy));
                else if (sy == 256) break;
                else {
                    sy -= 257;
                    if (sy >= 29) { err = 7; return; }
                    uint len = lbase[sy] + bits(s, lext[sy]);
                    int ds = decode(s, dCnt, dSym);
                    if (ds < 0 || ds >= 30) { err = 8; return; }
                    uint dist = dbase[ds] + bits(s, dext[ds]);
                    if (dist > s.outPos) { err = 9; return; }
                    for (uint i = 0u; i < len; ++i)
                        putByte(s, raw[s.outPos - dist]);
                }
            }
        } else { err = 4; return; }
    }
    outLen = s.outPos;
}

void main() {
    if (gl_GlobalInvocationID.x == 0u) inflate();
}
)GLSL";

// ==================== unfilter shader ====================
// PNG recon is inherently SERIAL: Sub/Average/Paeth carry a left
// dependency inside a row, Up/Average/Paeth a row dependency — one
// invocation walks the whole image. (A parallel version needs a
// segmented prefix-scan variant of Paeth — out of scope.)
static const char* kUnfilter = R"GLSL(
#version 460
layout(local_size_x = 1) in;
layout(set = 0, binding = 1) readonly buffer Raw { uint raw[]; };
layout(set = 0, binding = 3) buffer Recon { uint rec[]; }; // byte-words
layout(push_constant) uniform Q { uint w; uint h; uint bpp; };

int paeth(int a, int b, int c) {
    int p = a + b - c;
    int pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
    if (pa <= pb && pa <= pc) return a;
    return pb <= pc ? b : c;
}

void main() {
    if (gl_GlobalInvocationID.x != 0u) return;
    uint rb = w * bpp;
    for (uint y = 0u; y < h; ++y) {
        uint base = y * (rb + 1u);
        uint ft = raw[base];
        for (uint i = 0u; i < rb; ++i) {
            int x = int(raw[base + 1u + i]);
            int a = i >= bpp ? int(rec[y * rb + i - bpp]) : 0;
            int b = y > 0u   ? int(rec[(y - 1u) * rb + i]) : 0;
            int c = (y > 0u && i >= bpp)
                        ? int(rec[(y - 1u) * rb + i - bpp]) : 0;
            int v;
            if (ft == 0u)      v = x;
            else if (ft == 1u) v = x + a;
            else if (ft == 2u) v = x + b;
            else if (ft == 3u) v = x + ((a + b) >> 1);
            else               v = x + paeth(a, b, c);
            rec[y * rb + i] = uint(v) & 0xffu;
        }
    }
}
)GLSL";

// ==================== pack shader ====================
// per-pixel: recon bytes -> packed RGBA u32 (ctype expansion)
static const char* kPack = R"GLSL(
#version 460
layout(local_size_x = 64) in;
layout(set = 0, binding = 3) readonly buffer Recon { uint rec[]; };
layout(set = 0, binding = 4) writeonly buffer Pix { uint pix[]; };
layout(push_constant) uniform R { uint w; uint h; uint bpp; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= w * h) return;
    uint r, g, b, a = 255u;
    if (bpp == 4u) {
        r = rec[i * 4u]; g = rec[i * 4u + 1u];
        b = rec[i * 4u + 2u]; a = rec[i * 4u + 3u];
    } else if (bpp == 3u) {
        r = rec[i * 3u]; g = rec[i * 3u + 1u]; b = rec[i * 3u + 2u];
    } else { r = g = b = rec[i]; }
    pix[i] = r | (g << 8u) | (b << 16u) | (a << 24u);
}
)GLSL";

// ---------------- host: PNG chunk parser ----------------
struct PngInfo {
    uint32_t w = 0, h = 0;
    uint8_t depth = 0, ctype = 0, interlace = 0;
    std::vector<uint8_t> idat;
};

static uint32_t be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) << 8) | p[3];
}

static PngInfo parsePng(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> d{std::istreambuf_iterator<char>(f), {}};
    const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (d.size() < 8 || memcmp(d.data(), sig, 8))
        throw std::runtime_error("not a PNG");
    PngInfo info;
    size_t off = 8;
    while (off + 12 <= d.size()) {
        uint32_t len = be32(&d[off]);
        const uint8_t* payload = &d[off + 8];
        if (!memcmp(&d[off + 4], "IHDR", 4)) {
            info.w = be32(payload); info.h = be32(payload + 4);
            info.depth = payload[8]; info.ctype = payload[9];
            info.interlace = payload[12];
        } else if (!memcmp(&d[off + 4], "IDAT", 4)) {
            info.idat.insert(info.idat.end(), payload, payload + len);
        } else if (!memcmp(&d[off + 4], "IEND", 4)) break;
        off += 12 + len;
    }
    return info;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s in.png [out.ppm]\n", argv[0]);
        return 2;
    }
    const char* png = argv[1];
    const char* ppm = argc > 2 ? argv[2] : "out.ppm";

    auto info = parsePng(png);
    uint32_t bpp = info.ctype == 6 ? 4 : info.ctype == 2 ? 3
                  : info.ctype == 0 ? 1 : 0;
    if (!bpp || info.depth != 8 || info.interlace)
        throw std::runtime_error(
            "unsupported PNG (need 8-bit gray/RGB/RGBA, non-interlaced)");
    uint32_t rb = info.w * bpp;
    uint64_t expectRaw = uint64_t(info.h) * (rb + 1);
    printf("%s: %ux%u ctype=%u idat=%zu bytes -> expect %llu raw\n",
           png, info.w, info.h, info.ctype, info.idat.size(),
           (unsigned long long)expectRaw);

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice();
    auto memHost = vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent;

    uint32_t inWords = (info.idat.size() + 3) / 4 + 1;
    auto inBuf = vk.createBuffer(vk::DeviceSize(inWords) * 4,
        vk::BufferUsageFlagBits::eStorageBuffer, memHost, true);
    memset(inBuf.mapped, 0, inWords * 4);
    memcpy(inBuf.mapped, info.idat.data(), info.idat.size());
    auto rawBuf = vk.createBuffer(vk::DeviceSize(expectRaw) * 4,
        vk::BufferUsageFlagBits::eStorageBuffer, memHost, true);
    auto metaBuf = vk.createBuffer(32,
        vk::BufferUsageFlagBits::eStorageBuffer, memHost, true);
    memset(metaBuf.mapped, 0, 32);
    static_cast<uint32_t*>(metaBuf.mapped)[2] =
        uint32_t(expectRaw);                       // cap
    auto recBuf = vk.createBuffer(vk::DeviceSize(info.h) * rb * 4,
        vk::BufferUsageFlagBits::eStorageBuffer, memHost, true);
    auto pixBuf = vk.createBuffer(vk::DeviceSize(info.w) * info.h * 4,
        vk::BufferUsageFlagBits::eStorageBuffer, memHost, true);

    vk::DescriptorSetLayoutBinding binds[5];
    for (int i = 0; i < 5; ++i)
        binds[i].setBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    vk::DescriptorSetLayoutCreateInfo dsli{};
    dsli.setBindings(binds);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dsli);
    vk::DescriptorPoolSize psz{vk::DescriptorType::eStorageBuffer, 5};
    vk::DescriptorPoolCreateInfo dpi{};
    dpi.setMaxSets(1).setPoolSizes(psz);
    auto pool = vk.device->createDescriptorPoolUnique(dpi);
    vk::DescriptorSetAllocateInfo dai{};
    dai.setDescriptorPool(pool.get()).setSetLayouts(dsl.get());
    auto dsets = vk.device->allocateDescriptorSetsUnique(dai);
    auto& dset = dsets.front();
    vk::DescriptorBufferInfo infos[5] = {
        {inBuf.buf.get(), 0, VK_WHOLE_SIZE},
        {rawBuf.buf.get(), 0, VK_WHOLE_SIZE},
        {metaBuf.buf.get(), 0, VK_WHOLE_SIZE},
        {recBuf.buf.get(), 0, VK_WHOLE_SIZE},
        {pixBuf.buf.get(), 0, VK_WHOLE_SIZE}};
    for (int i = 0; i < 5; ++i) {
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
    auto [pipeI, mI] = mkPipe(kInflate);
    auto [pipeU, mU] = mkPipe(kUnfilter);
    auto [pipeP, mP] = mkPipe(kPack);

    vk::MemoryBarrier mb{};
    mb.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead |
                          vk::AccessFlagBits::eShaderWrite);
    vk.oneTime([&](vk::CommandBuffer c) {
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             layout.get(), 0, dset.get(), {});
        uint32_t pc[3] = {uint32_t(info.idat.size()), info.w, info.h};
        c.pushConstants<uint32_t>(layout.get(),
                                  vk::ShaderStageFlagBits::eCompute,
                                  0, pc);
        c.bindPipeline(vk::PipelineBindPoint::eCompute, pipeI.get());
        c.dispatch(1, 1, 1);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eComputeShader,
                          {}, mb, {}, {});
        // unfilter + pack push {w, h, bpp}
        pc[0] = info.w; pc[1] = info.h; pc[2] = bpp;
        c.pushConstants<uint32_t>(layout.get(),
                                  vk::ShaderStageFlagBits::eCompute,
                                  0, pc);
        c.bindPipeline(vk::PipelineBindPoint::eCompute, pipeU.get());
        c.dispatch(1, 1, 1);   // serial pass — see shader comment
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eComputeShader,
                          {}, mb, {}, {});
        c.bindPipeline(vk::PipelineBindPoint::eCompute, pipeP.get());
        c.dispatch((info.w * info.h + 63) / 64, 1, 1);
    });

    auto* meta = static_cast<uint32_t*>(metaBuf.mapped);
    int err = reinterpret_cast<int*>(meta)[1];
    if (err || meta[0] != expectRaw) {
        fprintf(stderr, "inflate failed: err=%d outLen=%u expected=%llu\n",
                err, meta[0], (unsigned long long)expectRaw);
        return 1;
    }
    printf("inflate ok: %u bytes -> unfiltered -> %ux%u RGBA\n",
           uint32_t(info.idat.size()), info.w, info.h);

    std::ofstream f(ppm, std::ios::binary);
    f << "P6\n" << info.w << " " << info.h << "\n255\n";
    auto* pix = static_cast<const uint8_t*>(pixBuf.mapped);
    for (uint64_t i = 0; i < uint64_t(info.w) * info.h; ++i)
        f.write(reinterpret_cast<const char*>(&pix[i * 4]), 3);
    printf("wrote %s\n", ppm);
    return 0;
}
