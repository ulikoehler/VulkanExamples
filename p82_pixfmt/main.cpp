// SPDX-License-Identifier: CC0-1.0
//
// Pixel-format conversion in a Vulkan compute shader — bit-exact vs
// Pillow's Image.convert().  One dispatch: each invocation produces one
// 4-byte output word, gathering the 1-2 output pixels whose bytes it
// needs (packed layout, no atomics, no CPU repack anywhere).
//
//   in.pnm ──memcpy──> SSBO (packed u8) ──> conv shader ──> packed u8
//                                                        2 out.pnm
//
// Conversions (all PIL-exact, see libImaging/Convert.c + ConvertYCbCr.c):
//   rgb2rgba   rgba2rgb    rgb2bgr    rgb2gray    gray2rgb
//   rgba2gray  gray2rgba   rgb2ycbcr  ycbcr2rgb   rgb2hsv
//   hsv2rgb    gray2ycbcr  ycbcr2gray
//
//   ./app in.pnm out.pnm <conv>
//   ./app --bench N in.pnm <conv>      N conversions, prints GB/s
//
// PNM I/O: P5=gray, P6=rgb, P7 PAM=rgba (what PIL writes).
// Build:   g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc
// Verify:  ./check.py   (bit-exact vs PIL convert + numpy refs)

#include "vkmini.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <vector>

// ============================================================
//  PIL YCbCr tables — regenerated exactly:
//    fwd[i]  = (int)(coef * 64 * i        + 0.5)
//    rev[i]  = (int)(coef * 64 * (i-128)  + 0.5)
//  (the +0.5 before a truncating int() cast = C round-half-up, which
//  is asymmetric for negatives — matching ConvertYCbCr.c verbatim)
// ============================================================
static void buildYccLuts(int32_t* lut /*12*256*/) {
    auto fwd = [&](int t, double c) {
        for (int i = 0; i < 256; ++i) lut[t * 256 + i] = int(c * 64 * i + 0.5);
    };
    auto rev = [&](int t, double c) {
        for (int i = 0; i < 256; ++i)
            lut[t * 256 + i] = int(c * 64 * (i - 128) + 0.5);
    };
    fwd(0, 0.29900);  fwd(1, 0.58700);  fwd(2, 0.11400);   // Y
    fwd(3, -0.16874); fwd(4, -0.33126); fwd(5, 0.50000);   // Cb
    /* Cr_R aliases Cb_B */ fwd(6, -0.41869); fwd(7, -0.08131);  // Cr_G,Cr_B
    rev(8, 1.40200);  rev(9, -0.34414); rev(10, -0.71414);
    rev(11, 1.77200);
}

// ============================================================
//  Shader — one invocation per 4-byte output word
// ============================================================
static const char* kConv = R"GLSL(
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer In  { uint inBuf[]; };
layout(set = 0, binding = 1) readonly buffer Lut { int  lut[]; };
layout(set = 0, binding = 3) writeonly buffer Out{ uint outBuf[]; };
layout(push_constant) uniform P { uint npx, inComp, outComp, conv; };

uint byteAt(uint i) { return (inBuf[i >> 2] >> ((i & 3u) * 8u)) & 0xFFu; }

// The shader optimizer may fold float->double conversions back into the
// producing double expression (removing C's intermediate float32
// rounding — C stores to `float` between steps). A bit-cast round-trip
// is opaque to value tracking and forces the rounding to materialize.
float fr(float x) { return uintBitsToFloat(floatBitsToUint(x)); }

// Knuth TwoSum: s + e == a + b exactly (fp32 add/mul are correctly
// rounded in hardware; only division isn't)
void twoSum(float a, float b, out float s, out float e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

// Exact trunc(x*255) for x in [0,1] via integer mantissa arithmetic
// (fast-math folds an fma-residual formulation away, losing the guard
// bits; integers cannot be rewritten).
int mul255trunc(float x) {
    if (!(x > 0.0)) return 0;
    uint bits = floatBitsToUint(x);
    int e = int((bits >> 23) & 0xFFu) - 150;   // exponent of mantissa lsb
    uint64_t M = uint64_t((bits & 0x7FFFFFu) | 0x800000u);
    uint64_t prod = M * 255u;                  // exact: M <= 2^24
    if (e >= 0) return int(prod << uint(e));
    int sh = -e;
    if (sh >= 64) return 0;
    return int(prod >> uint(sh));
}

// Correctly-rounded fp32 of a/b for integers a,b <= 4096 (integer long
// division at 48 fractional bits: immune to the optimizer's algebraic
// rewrites and to RADV's reciprocal-multiply fp32 division).
float idivRN(uint a, uint b) {
    if (a == 0u) return 0.0;
    uint64_t num = uint64_t(a) << 48u;
    uint64_t q = num / uint64_t(b);
    uint64_t rem = num - q * uint64_t(b);
    int sh = int(findMSB(q)) - 23;
    uint m;
    if (sh <= 0) {
        m = uint(q);
        uint64_t twice = rem * 2u;
        if (twice > uint64_t(b) ||
            (twice == uint64_t(b) && (m & 1u) != 0u)) m += 1u;
        return float(m) * exp2(-48.0);
    }
    m = uint(q >> uint(sh));
    uint64_t lost = q & ((uint64_t(1) << uint(sh)) - 1u);
    uint64_t halfway = uint64_t(1) << uint(sh - 1);
    if (lost > halfway ||
        (lost == halfway && (rem > 0u || (m & 1u) != 0u)))
        m += 1u;
    return float(m) * exp2(float(sh - 48));
}

// Correctly-rounded fp32 of x/6 for any finite normal x: 24-bit
// mantissa scaled by 256, integer divide by 3, round to 24 bits.
float div6rn(float x) {
    uint bits = floatBitsToUint(x);
    int e = int((bits >> 23) & 0xFFu) - 150;      // exponent - 23
    uint64_t M = uint64_t((bits & 0x7FFFFFu) | 0x800000u) << 8u;
    uint64_t q = M / 3u;
    uint64_t rem = M - q * 3u;
    int sh = int(findMSB(q)) - 23;
    uint m;
    if (sh <= 0) {
        m = uint(q);
    } else {
        m = uint(q >> uint(sh));
        uint64_t lost = q & ((uint64_t(1) << uint(sh)) - 1u);
        uint64_t halfway = uint64_t(1) << uint(sh - 1);
        if (lost > halfway ||
            (lost == halfway && (rem > 0u || (m & 1u) != 0u)))
            m += 1u;
    }
    float r = ldexp(float(m), e - 9 + max(sh, 0));
    return (bits & 0x80000000u) != 0u ? -r : r;
}

// PIL L24: luma at 16.16-ish precision then >>16
uint rgb2gray(uint r, uint g, uint b) {
    return (r * 19595u + g * 38470u + b * 7471u + 0x8000u) >> 16;
}
uvec3 convPx(uint px) {
    uint bi = px * inComp;
    uint r = byteAt(bi);
    uint g = inComp > 1u ? byteAt(bi + 1u) : 0u;
    uint b = inComp > 1u ? byteAt(bi + 2u) : 0u;
    switch (conv) {
    case 0u:  return uvec3(r, g, b);          // rgb2rgba (A added below)
    case 1u:  return uvec3(r, g, b);          // rgba2rgb
    case 2u:  return uvec3(b, g, r);          // rgb2bgr
    case 3u:  { uint l = rgb2gray(r, g, b); return uvec3(l, 0, 0); }
    case 4u:  return uvec3(r, r, r);          // gray2rgb
    case 5u:  { uint l = rgb2gray(r, g, b); return uvec3(l, 0, 0); }
    case 6u:  return uvec3(r, r, r);          // gray2rgba
    case 7u: {                                // rgb2ycbcr (LUTs)
        int ri = int(r), gi = int(g), bb = int(b);
        int y  = (lut[ri] + lut[256 + gi] + lut[512 + bb]) >> 6;
        int cb = ((lut[768 + ri] + lut[1024 + gi] + lut[1280 + bb]) >> 6) + 128;
        int cr = ((lut[1280 + ri] + lut[1536 + gi] + lut[1792 + bb]) >> 6) + 128;
        return uvec3(y, cb, cr);
    }
    case 8u: {                                // ycbcr2rgb (LUTs)
        int y = int(r), cb = int(g), cr = int(b);
        int rr = y + (lut[2048 + cr] >> 6);
        int gg = y + ((lut[2304 + cb] + lut[2560 + cr]) >> 6);
        int bb = y + (lut[2816 + cb] >> 6);
        return uvec3(clamp(rr, 0, 255), clamp(gg, 0, 255), clamp(bb, 0, 255));
    }
    case 9u: {                                // rgb2hsv (colorsys order)
        int mx = int(max(r, max(g, b))), mn = int(min(r, min(g, b)));
        uint uh = 0u, us = 0u; uint uv = uint(mx);
        if (mn != mx) {
            float s = idivRN(uint(mx - mn), uint(mx));
            float rc = idivRN(uint(mx - int(r)), uint(mx - mn));
            float gc = idivRN(uint(mx - int(g)), uint(mx - mn));
            float bc = idivRN(uint(mx - int(b)), uint(mx - mn));
            float h;
            // C: 2.0/4.0 are double literals -> double arithmetic,
            // stored back to float; fr() keeps the float rounding
            if (r == uint(mx))      h = fr(bc - gc);
            else if (g == uint(mx))
                h = fr(float(2.0lf + double(rc) - double(bc)));
            else
                h = fr(float(4.0lf + double(gc) - double(rc)));
            // C: h = fmod(h/6.0 + 1.0, 1.0) in double, stored to float.
            // h >= 0 -> h/6;  h < 0 -> h/6 + 1 (double-float keeps the
            // extra bits the +1 needs).
            if (h < 0.0) {
                // C computes this in double; fp64 + the fr() barrier
                // (no fold-away of the intermediate float rounding)
                h = fr(float(mod(double(h) / 6.0lf + 1.0lf, 1.0lf)));
            } else {
                h = div6rn(h);
            }
            uh = uint(clamp(mul255trunc(h), 0, 255));
            us = uint(clamp(mul255trunc(s), 0, 255));
        }
        return uvec3(uh, us, uv);
    }
    case 10u: {                               // hsv2rgb
        uint h = r, s = g, v = b;
        if (s == 0u) return uvec3(v, v, v);
        // C: the whole chain below runs in double (float operands are
        // promoted at the double literals); f/fs are stored back to float
        double hd = double(h);
        int i = int(floor(hd * 6.0lf / 255.0lf));
        float f = float(hd * 6.0lf / 255.0lf - double(i));
        float fs = float(double(s) / 255.0lf);
        // PIL round() on nonneg values == floor(x+0.5)
        int p = int(floor(double(v) * (1.0lf - double(fs)) + 0.5lf));
        int q = int(floor(double(v) * (1.0lf - double(fs) * double(f))
                          + 0.5lf));
        int t = int(floor(double(v) * (1.0lf - double(fs) *
                                       (1.0lf - double(f))) + 0.5lf));
        uint up = uint(clamp(p, 0, 255));
        uint uq = uint(clamp(q, 0, 255));
        uint ut = uint(clamp(t, 0, 255));
        switch (i % 6) {
        case 0: return uvec3(v, ut, up);
        case 1: return uvec3(uq, v, up);
        case 2: return uvec3(up, v, ut);
        case 3: return uvec3(up, uq, v);
        case 4: return uvec3(ut, up, v);
        default: return uvec3(v, up, uq);
        }
    }
    case 11u: return uvec3(r, 128u, 128u);    // gray2ycbcr
    case 12u: return uvec3(r, 0, 0);          // ycbcr2gray (Y only)
    }
    return uvec3(0);
}

void main() {
    uint w = gl_GlobalInvocationID.x;
    if (w * 4u >= npx * outComp) return;
    uint word = 0u;
    for (uint k = 0u; k < 4u; ++k) {
        uint bpos = w * 4u + k;
        if (bpos >= npx * outComp) break;
        uint px = bpos / outComp, ch = bpos - px * outComp;
        uvec3 c = convPx(px);
        uint v = (ch == 3u) ? 255u : (ch == 0u ? c.x : (ch == 1u ? c.y : c.z));
        word |= v << (k * 8u);
    }
    outBuf[w] = word;
}
)GLSL";

// ============================================================
//  Conversions table: name 2 {op, inComp, outComp}
// ============================================================
struct ConvDef { uint32_t op, inC, outC; };
static const std::map<std::string, ConvDef> kConvs = {
    {"rgb2rgba",   {0, 3, 4}}, {"rgba2rgb",  {1, 4, 3}},
    {"rgb2bgr",    {2, 3, 3}}, {"bgr2rgb",   {2, 3, 3}},
    {"rgb2gray",   {3, 3, 1}}, {"gray2rgb",  {4, 1, 3}},
    {"rgba2gray",  {5, 4, 1}}, {"gray2rgba", {6, 1, 4}},
    {"rgb2ycbcr",  {7, 3, 3}}, {"ycbcr2rgb", {8, 3, 3}},
    {"rgb2hsv",    {9, 3, 3}}, {"hsv2rgb",   {10, 3, 3}},
    {"gray2ycbcr", {11, 1, 3}}, {"ycbcr2gray", {12, 3, 1}},
};

// ============================================================
//  Vulkan plumbing
// ============================================================
struct Gpu {
    vkmini::Vk v;
    vk::UniqueDescriptorSetLayout dsetLayout;
    vk::UniqueDescriptorPool dpool;
    vk::DescriptorSet dset;
    vk::UniquePipelineLayout pipeLayout;
    vk::UniquePipeline pipe;

    void init() {
        v.createInstance({});
        v.pickPhysicalDevice();
        // HSV conversions use 64-bit integer division for exact
        // rounding (see idivRN in the shader)
        vk::PhysicalDeviceFeatures2 f2{};
        f2.features.shaderInt64 = VK_TRUE;
        f2.features.shaderFloat64 = VK_TRUE;
        v.createDevice({}, &f2);
        std::vector<vk::DescriptorSetLayoutBinding> binds;
        for (uint32_t i = 0; i < 4; ++i)
            binds.push_back({i, vk::DescriptorType::eStorageBuffer, 1,
                             vk::ShaderStageFlagBits::eCompute});
        dsetLayout = v.device->createDescriptorSetLayoutUnique({{}, binds});
        std::vector<vk::DescriptorPoolSize> sz = {
            {vk::DescriptorType::eStorageBuffer, 4}};
        dpool = v.device->createDescriptorPoolUnique(
            {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, sz});
        dset = v.device->allocateDescriptorSets(
            {dpool.get(), dsetLayout.get()}).front();
        vk::PushConstantRange pc{vk::ShaderStageFlagBits::eCompute, 0, 16};
        pipeLayout = v.device->createPipelineLayoutUnique(
            {{}, dsetLayout.get(), pc});
        auto mod = v.shader(kConv, shaderc_compute_shader, "conv");
        vk::ComputePipelineCreateInfo ci{};
        ci.stage = {vk::PipelineShaderStageCreateFlags{},
                    vk::ShaderStageFlagBits::eCompute, mod.get(), "main"};
        ci.layout = pipeLayout.get();
        pipe = v.device->createComputePipelineUnique({}, ci).value;
    }
};

// ============================================================
//  PNM I/O — P5 gray / P6 rgb / P7 PAM rgba (what PIL emits)
// ============================================================
struct Img {
    int W = 0, H = 0, nComp = 3;
    std::vector<uint8_t> px;
};

static bool loadPnm(const std::string& path, Img& im) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string magic;
    f >> magic;
    if (magic != "P6" && magic != "P5") return false;
    int maxv;
    f >> im.W >> im.H >> maxv;
    f.get();
    im.nComp = magic == "P6" ? 3 : 1;
    if (maxv != 255) return false;
    im.px.resize(size_t(im.W) * im.H * im.nComp);
    f.read(reinterpret_cast<char*>(im.px.data()), im.px.size());
    return bool(f);
}

static void savePnm(const std::string& path, const Img& im) {
    std::ofstream f(path, std::ios::binary);
    if (im.nComp != 1 && im.nComp != 3) {
        f.write(reinterpret_cast<const char*>(im.px.data()), im.px.size());
        return;
    }
    f << (im.nComp == 1 ? "P5\n" : "P6\n") << im.W << " " << im.H
      << "\n255\n";
    f.write(reinterpret_cast<const char*>(im.px.data()), im.px.size());
}

// ============================================================
//  Driver
// ============================================================
static bool gpuConvert(Gpu& g, const Img& in, Img& out, const ConvDef& cd,
                       const int32_t* luts, double* tMs = nullptr) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    size_t inBytes = size_t(in.W) * in.H * in.nComp;
    size_t outBytes = size_t(in.W) * in.H * cd.outC;
    auto mkbuf = [&](vk::DeviceSize sz, bool map, bool cached) {
        auto props = vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent;
        if (cached) props |= vk::MemoryPropertyFlagBits::eHostCached;
        return g.v.createBuffer(
            sz, vk::BufferUsageFlagBits::eStorageBuffer, props, map);
    };
    auto inBuf = mkbuf((inBytes + 3) & ~size_t(3), true, false);
    memcpy(inBuf.mapped, in.px.data(), inBytes);
    auto lutBuf = mkbuf(12 * 256 * 4, true, false);
    memcpy(lutBuf.mapped, luts, 12 * 256 * 4);
    auto outBuf = mkbuf((outBytes + 3) & ~size_t(3), true, true);

    auto bind = [&](uint32_t b, const vkmini::Vk::Buffer& buf) {
        vk::DescriptorBufferInfo bi{buf.buf.get(), 0, buf.size};
        vk::WriteDescriptorSet w{g.dset, b, 0, 1,
                                 vk::DescriptorType::eStorageBuffer,
                                 nullptr, &bi};
        g.v.device->updateDescriptorSets(w, {});
    };
    bind(0, inBuf); bind(1, lutBuf); bind(3, outBuf);

    uint32_t pc[4] = {uint32_t(in.W) * uint32_t(in.H), cd.inC, cd.outC,
                      cd.op};
    g.v.oneTime([&](vk::CommandBuffer c) {
        c.pushConstants(g.pipeLayout.get(), vk::ShaderStageFlagBits::eCompute,
                        0, 16, pc);
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             g.pipeLayout.get(), 0, g.dset, {});
        c.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipe.get());
        c.dispatch(uint32_t((outBytes + 1023) / 1024), 1, 1);
    });
    if (tMs)
        *tMs = std::chrono::duration<double, std::milli>(clk::now() - t0)
                   .count();
    out.px.resize(outBytes);
    memcpy(out.px.data(), outBuf.mapped, outBytes);
    return true;
}

int main(int argc, char** argv) {
    int bench = 0;
    int rawW = 0, rawH = 0, rawC = 0;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bench") bench = atoi(argv[++i]);
        else if (a == "--raw") {
            if (sscanf(argv[++i], "%dx%dx%d", &rawW, &rawH, &rawC) != 3) {
                fprintf(stderr, "bad --raw WxHxC\n");
                return 2;
            }
        } else pos.push_back(a);
    }
    if (pos.size() < 3) {
        fprintf(stderr,
            "usage: app [--bench N] [--raw WxHxC] in out conv\n"
            "  --raw: headerless bytes (output also raw), C in {1,3,4}\n"
            " convs:");
        for (auto& [k, v] : kConvs) fprintf(stderr, " %s", k.c_str());
        fprintf(stderr, "\n");
        return 2;
    }
    Img in;
    if (rawW > 0) {
        std::ifstream f(pos[0], std::ios::binary);
        in.W = rawW; in.H = rawH; in.nComp = rawC;
        in.px.resize(size_t(rawW) * rawH * rawC);
        if (!f || !f.read(reinterpret_cast<char*>(in.px.data()),
                          in.px.size())) {
            fprintf(stderr, "bad raw input\n");
            return 1;
        }
    } else if (!loadPnm(pos[0], in)) {
        fprintf(stderr, "bad input pnm\n");
        return 1;
    }
    auto it = kConvs.find(pos[2]);
    if (it == kConvs.end() || it->second.inC != uint32_t(in.nComp)) {
        fprintf(stderr, "unknown/mismatched conv %s (in has %d ch)\n",
                pos[2].c_str(), in.nComp);
        return 2;
    }
    Img out{in.W, in.H, int(it->second.outC), {}};

    int32_t luts[12 * 256];
    buildYccLuts(luts);
    Gpu g;
    g.init();
    fprintf(stderr, "%s (%dx%d) 2 %s\n", pos[0].c_str(), in.W, in.H,
            pos[2].c_str());

    if (bench > 0) {
        gpuConvert(g, in, out, it->second, luts);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < bench; ++i)
            gpuConvert(g, in, out, it->second, luts);
        double s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        double gb = double(out.px.size()) * bench / 1e9;
        printf("%d conversions in %.2fs -> %.2f GB out/s\n", bench, s, gb / s);
        return 0;
    }
    if (!gpuConvert(g, in, out, it->second, luts)) return 1;
    if (rawW > 0) {
        std::ofstream f(pos[1], std::ios::binary);
        f.write(reinterpret_cast<const char*>(out.px.data()), out.px.size());
    } else {
        savePnm(pos[1], out);
    }
    printf("wrote %s\n", pos[1].c_str());
    return 0;
}
