// SPDX-License-Identifier: CC0-1.0
//
// Image resizing in Vulkan compute shaders — bit-exact vs Pillow's
// Image.resize(). Two separable passes, like PIL:
//
//   1. horizontal: one invocation per (outX, inY) -> u8 temp image
//   2. vertical:   one invocation per (outX, outY) -> final u8 image
//
// All six PIL filters: NEAREST, BOX, BILINEAR, HAMMING, BICUBIC,
// LANCZOS — replicating libImaging/Resample.c exactly:
//
//   * support-scaled window: filterscale = max(inSize/outSize, 1),
//     support = filter.support * filterscale, ksize = ceil(support)*2+1
//   * per-output-pixel weights w = filter((x+xmin-center+0.5)/filterscale),
//     normalized to sum 1.0 in doubles, then quantized to int32 at
//     PRECISION_BITS=22 with round-half-away-from-zero
//   * accumulate ss = 2^21 + sum(in*k) as INT32, out = clamp(ss>>22,0,255)
//   * the horizontal-pass intermediate is itself u8 (clamped), matching
//     PIL's UINT8 temp image
//
//   ./app in.ppm out.ppm 640x480 lanczos     resize
//   ./app --bench N in.ppm WxH filter      N resizes, prints MP/s
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc
// Verify: ./check.py  (PIL reference, bit-exact)

#include "vkmini.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <vector>

// ============================================================
//  PIL filter kernels — verbatim libImaging/Resample.c
// ============================================================
using FilterFn = double (*)(double);
struct FilterDef { FilterFn fn; double support; };

static double box_filter(double x) {
    return (x > -0.5 && x <= 0.5) ? 1.0 : 0.0;
}
static double bilinear_filter(double x) {
    if (x < 0.0) x = -x;
    return (x < 1.0) ? 1.0 - x : 0.0;
}
static double hamming_filter(double x) {
    if (x < 0.0) x = -x;
    if (x == 0.0) return 1.0;
    if (x >= 1.0) return 0.0;
    x = x * M_PI;
    return sin(x) / x * (0.54 + 0.46 * cos(x));
}
static double bicubic_filter(double x) {
    const double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}
static double sinc_filter(double x) {
    if (x == 0.0) return 1.0;
    x = x * M_PI;
    return sin(x) / x;
}
static double lanczos_filter(double x) {
    if (-3.0 <= x && x < 3.0) return sinc_filter(x) * sinc_filter(x / 3);
    return 0.0;
}

static const std::map<std::string, FilterDef> kFilters = {
    {"box",      {box_filter, 0.5}},
    {"bilinear", {bilinear_filter, 1.0}},
    {"hamming",  {hamming_filter, 1.0}},
    {"bicubic",  {bicubic_filter, 2.0}},
    {"lanczos",  {lanczos_filter, 3.0}},
};

// ============================================================
//  PIL precompute_coeffs -> int32 tables (PRECISION_BITS = 22)
// ============================================================
static constexpr int kPrecBits = 22;

struct Axis {
    int inSize, outSize, ksize;
    std::vector<int32_t> bounds;   // 2 per out px: {xmin, xmax}
    std::vector<int32_t> coefs;    // outSize * ksize
};

static Axis precomputeCoeffs(int inSize, int outSize, const FilterDef& f) {
    double scale = double(inSize) / outSize;
    double filterscale = std::max(scale, 1.0);
    double support = f.support * filterscale;
    int ksize = int(ceil(support)) * 2 + 1;
    double inv_fs = 1.0 / filterscale;

    Axis a{inSize, outSize, ksize,
           std::vector<int32_t>(outSize * 2),
           std::vector<int32_t>(size_t(outSize) * ksize, 0)};
    std::vector<double> k(ksize);
    for (int xx = 0; xx < outSize; ++xx) {
        double center = (xx + 0.5) * scale;
        int xmin = int(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = int(center + support + 0.5);
        if (xmax > inSize) xmax = inSize;
        xmax -= xmin;
        double ww = 0.0;
        for (int x = 0; x < xmax; ++x) {
            double w = f.fn((x + xmin - center + 0.5) * inv_fs);
            k[x] = w;
            ww += w;
        }
        if (ww != 0.0)
            for (int x = 0; x < xmax; ++x) k[x] /= ww;
        for (int x = xmax; x < ksize; ++x) k[x] = 0.0;
        for (int x = 0; x < ksize; ++x) {
            double v = k[x] * (1 << kPrecBits);
            a.coefs[size_t(xx) * ksize + x] =
                int32_t(v < 0 ? v - 0.5 : v + 0.5);
        }
        a.bounds[xx * 2] = xmin;
        a.bounds[xx * 2 + 1] = xmax;
    }
    return a;
}

// PIL NEAREST goes through ImagingGenericTransform (Geometry.c), not
// Resample.c: xin = (x + 0.5) * inSize / outSize evaluated in double,
// source index = (int)xin (floor for xin >= 0, always true here).
// Expressed as a degenerate 1-tap resample so the same shader path
// reproduces it exactly.
static Axis nearestCoeffs(int inSize, int outSize) {
    Axis a{inSize, outSize, 1,
           std::vector<int32_t>(outSize * 2),
           std::vector<int32_t>(outSize)};
    // PIL ImagingScaleAffine accumulates iteratively:
    //   xo = 0.5*a0;  sx = (int)xo;  xo += a0;  ...
    // — NOT xin = a0*(x+0.5).  Repeated additions accumulate rounding
    // error, so e.g. a mathematically-exact 8.0 can land as 7.999…96
    // and pick the previous pixel.  Replicate the accumulation.
    double a0 = double(inSize) / outSize;
    double xo = 0.5 * a0;
    for (int xx = 0; xx < outSize; ++xx) {
        int sx = int(xo);
        if (sx >= inSize) sx = inSize - 1;
        a.bounds[xx * 2] = sx;
        a.bounds[xx * 2 + 1] = 1;
        a.coefs[xx] = 1 << kPrecBits;
        xo += a0;
    }
    return a;
}

// ============================================================
//  Shaders — separable resample + nearest
// ============================================================
static const char* kResample = R"GLSL(
#version 460
layout(local_size_x = 8, local_size_y = 8) in;

// axis 0 (horizontal): inBuf = packed u8 (nComp bytes/px), Out = RGBA
//                      u32 temp
// axis 1 (vertical):   inBuf = RGBA u32 temp,             Out = RGBA
//                      u32 result (host repacks to u8)
layout(set = 0, binding = 0) readonly buffer In  { uint inBuf[]; };
layout(set = 0, binding = 1) readonly buffer Bnd { ivec2 bnd[]; };
layout(set = 0, binding = 2) readonly buffer Cfs { int cf[]; };
layout(set = 0, binding = 3) writeonly buffer Out{ uint outBuf[]; };
layout(push_constant) uniform P {
    uint inW, inH, outW, outH, ksize, axis, nComp;
};

uint byteAt(uint i) { return (inBuf[i >> 2] >> ((i & 3u) * 8u)) & 0xFFu; }

void main() {
    uint ox = gl_GlobalInvocationID.x, oy = gl_GlobalInvocationID.y;
    if (ox >= outW || oy >= outH) return;
    uint o = (axis == 0u) ? ox : oy;
    ivec2 b = bnd[o];
    // ss = 1 << (PRECISION_BITS-1) per channel
    ivec4 ss = ivec4(1 << 21);
    for (int i = 0; i < b.y; ++i) {
        int k = cf[o * ksize + uint(i)];
        uint src = uint(b.x + i);
        if (axis == 0u) {
            // packed input: one byte extract per channel
            uint bi = (oy * inW + src) * nComp;
            ss += ivec4(int(byteAt(bi)),
                        nComp > 1u ? int(byteAt(bi + 1u)) : 0,
                        nComp > 1u ? int(byteAt(bi + 2u)) : 0, 0) * k;
        } else {
            uint px = inBuf[src * inW + ox];
            ss += ivec4(int(px & 0xFFu), int((px >> 8) & 0xFFu),
                        int((px >> 16) & 0xFFu), 0) * k;
        }
    }
    uint r = uint(clamp(ss.r >> 22, 0, 255));
    uint g = uint(clamp(ss.g >> 22, 0, 255));
    uint bch = uint(clamp(ss.b >> 22, 0, 255));
    outBuf[oy * outW + ox] = r | (g << 8) | (bch << 16) | 0xFF000000u;
}
)GLSL";

// Pack RGBA u32 pixels -> packed u8 (nComp bytes/px). One invocation per
// output word: gathers the 4 bytes spanning 1-2 source pixels, no
// atomics needed.
static const char* kPack = R"GLSL(
#version 460
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer In  { uint inBuf[]; };
layout(set = 0, binding = 3) writeonly buffer Out{ uint outBuf[]; };
layout(push_constant) uniform P { uint npx, nComp; };
void main() {
    uint w = gl_GlobalInvocationID.x;
    if (w * 4u >= npx * nComp) return;
    uint word = 0u;
    for (uint k = 0u; k < 4u; ++k) {
        uint bpos = w * 4u + k;
        if (bpos >= npx * nComp) break;
        uint px = bpos / nComp, ch = bpos - px * nComp;
        word |= ((inBuf[px] >> (ch * 8u)) & 0xFFu) << (k * 8u);
    }
    outBuf[w] = word;
}
)GLSL";

// ============================================================
//  Vulkan plumbing (same minimal pattern as p80)
// ============================================================
struct Gpu {
    vkmini::Vk v;
    vk::UniqueDescriptorSetLayout dsetLayout;
    vk::UniqueDescriptorPool dpool;
    vk::DescriptorSet dset;
    vk::UniquePipelineLayout pipeLayout;
    vk::UniquePipeline pResample, pPack;

    void init() {
        v.createInstance({});
        v.pickPhysicalDevice();
        v.createDevice();
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
        vk::PushConstantRange pc{vk::ShaderStageFlagBits::eCompute, 0, 28};
        pipeLayout = v.device->createPipelineLayoutUnique(
            {{}, dsetLayout.get(), pc});
        auto mk = [&](const char* src, const char* name) {
            auto mod = v.shader(src, shaderc_compute_shader, name);
            vk::ComputePipelineCreateInfo ci{};
            ci.stage = {vk::PipelineShaderStageCreateFlags{},
                        vk::ShaderStageFlagBits::eCompute, mod.get(), "main"};
            ci.layout = pipeLayout.get();
            return v.device->createComputePipelineUnique({}, ci).value;
        };
        pResample = mk(kResample, "resample");
        pPack = mk(kPack, "pack");
    }

    vkmini::Vk::Buffer ssbo(vk::DeviceSize bytes, bool map = false,
                            bool cached = false) {
        auto props = vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent;
        if (cached) props |= vk::MemoryPropertyFlagBits::eHostCached;
        return v.createBuffer(bytes, vk::BufferUsageFlagBits::eStorageBuffer,
                              props, map);
    }

    void bindBuf(uint32_t binding, const vkmini::Vk::Buffer& b) {
        vk::DescriptorBufferInfo bi{b.buf.get(), 0, b.size};
        vk::WriteDescriptorSet w{dset, binding, 0, 1,
                                 vk::DescriptorType::eStorageBuffer,
                                 nullptr, &bi};
        v.device->updateDescriptorSets(w, {});
    }

    void barrier(vk::CommandBuffer c) {
        vk::MemoryBarrier mb{vk::AccessFlagBits::eShaderWrite,
                             vk::AccessFlagBits::eShaderRead};
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eComputeShader,
                          {}, mb, {}, {});
    }
};

// ============================================================
//  PPM I/O + resize driver
// ============================================================
struct Img {
    int W = 0, H = 0, nComp = 3;
    std::vector<uint8_t> px;   // raw, nComp per pixel
};

static bool loadPpm(const std::string& path, Img& im) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string magic;
    int maxv = 0;
    f >> magic >> im.W >> im.H >> maxv;
    f.get();
    if (magic == "P6") im.nComp = 3;
    else if (magic == "P5") im.nComp = 1;
    else return false;
    if (maxv != 255 || im.W <= 0 || im.H <= 0) return false;
    im.px.resize(size_t(im.W) * im.H * im.nComp);
    f.read(reinterpret_cast<char*>(im.px.data()), im.px.size());
    return bool(f);
}

static void savePpm(const std::string& path, const Img& im) {
    std::ofstream f(path, std::ios::binary);
    f << (im.nComp == 1 ? "P5\n" : "P6\n") << im.W << " " << im.H
      << "\n255\n";
    f.write(reinterpret_cast<const char*>(im.px.data()), im.px.size());
}

struct StageTimes { double conv, alloc, up, h, v, down; };
static StageTimes gTimes;

// GPU resize: input packed bytes -> RGBA u32 -> 2 passes -> RGBA u32 ->
// repack. f == nullptr selects PIL NEAREST (1-tap degenerate coeffs).
// Returns false on Vulkan failure.
static bool gpuResize(Gpu& g, const Img& in, Img& out,
                      const FilterDef* f) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    Axis ax = f ? precomputeCoeffs(in.W, out.W, *f)
                : nearestCoeffs(in.W, out.W);
    Axis ay = f ? precomputeCoeffs(in.H, out.H, *f)
                : nearestCoeffs(in.H, out.H);
    gTimes.conv = std::chrono::duration<double>(clk::now() - t0).count();
    t0 = clk::now();

    // input goes up packed (no RGBA expansion); round up to word size
    size_t inBytes = size_t(in.W) * in.H * in.nComp;
    auto inBuf = g.ssbo((inBytes + 3) & ~size_t(3), true);
    memcpy(inBuf.mapped, in.px.data(), inBytes);
    auto bndX = g.ssbo(ax.bounds.size() * 4, true);
    auto bndY = g.ssbo(ay.bounds.size() * 4, true);
    auto cfX = g.ssbo(ax.coefs.size() * 4, true);
    auto cfY = g.ssbo(ay.coefs.size() * 4, true);
    memcpy(bndX.mapped, ax.bounds.data(), ax.bounds.size() * 4);
    memcpy(bndY.mapped, ay.bounds.data(), ay.bounds.size() * 4);
    memcpy(cfX.mapped, ax.coefs.data(), ax.coefs.size() * 4);
    memcpy(cfY.mapped, ay.coefs.data(), ay.coefs.size() * 4);
    gTimes.up = std::chrono::duration<double>(clk::now() - t0).count();
    t0 = clk::now();
    auto tmpBuf = g.ssbo(size_t(out.W) * in.H * 4);
    auto outRgba = g.ssbo(size_t(out.W) * out.H * 4);
    size_t outBytes = size_t(out.W) * out.H * out.nComp;
    auto packBuf = g.ssbo((outBytes + 3) & ~size_t(3), true, true);
    gTimes.alloc = std::chrono::duration<double>(clk::now() - t0).count();
    t0 = clk::now();

    uint32_t pc[7] = {uint32_t(in.W),  uint32_t(in.H), uint32_t(out.W),
                      uint32_t(out.H), 0u,             0u,
                      uint32_t(in.nComp)};
    try {
        {                            // filtered: H then V
            g.bindBuf(0, inBuf);
            g.bindBuf(1, bndX);
            g.bindBuf(2, cfX);
            g.bindBuf(3, tmpBuf);
            g.v.oneTime([&](vk::CommandBuffer c) {
                // pass 1: input (inW x inH) -> temp (outW x inH)
                pc[2] = uint32_t(out.W); pc[3] = uint32_t(in.H);
                pc[4] = uint32_t(ax.ksize); pc[5] = 0u;
                c.pushConstants(g.pipeLayout.get(),
                                vk::ShaderStageFlagBits::eCompute,
                                0, 28, pc);
                c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                     g.pipeLayout.get(), 0, g.dset, {});
                c.bindPipeline(vk::PipelineBindPoint::eCompute,
                               g.pResample.get());
                c.dispatch((out.W + 7) / 8, (in.H + 7) / 8, 1);
            });
            gTimes.h = std::chrono::duration<double>(clk::now() - t0).count();
            t0 = clk::now();
            // second dispatch reads tmpBuf -> bind it as input
            g.bindBuf(0, tmpBuf);
            g.bindBuf(1, bndY);
            g.bindBuf(2, cfY);
            g.bindBuf(3, outRgba);
            g.v.oneTime([&](vk::CommandBuffer c) {
                // pass 2: temp (outW x inH) -> out (outW x outH)
                pc[0] = uint32_t(out.W);
                pc[2] = uint32_t(out.W); pc[3] = uint32_t(out.H);
                pc[4] = uint32_t(ay.ksize); pc[5] = 1u;
                c.pushConstants(g.pipeLayout.get(),
                                vk::ShaderStageFlagBits::eCompute,
                                0, 28, pc);
                c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                     g.pipeLayout.get(), 0, g.dset, {});
                c.bindPipeline(vk::PipelineBindPoint::eCompute,
                               g.pResample.get());
                c.dispatch((out.W + 7) / 8, (out.H + 7) / 8, 1);
            });
            gTimes.v = std::chrono::duration<double>(clk::now() - t0).count();
            t0 = clk::now();
        }
        // third pass: pack RGBA temp -> packed u8 (output of gpuResize is
        // then a plain memcpy — no repack loop on the CPU)
        g.bindBuf(0, outRgba);
        g.bindBuf(3, packBuf);
        uint32_t ppc[2] = {uint32_t(out.W) * uint32_t(out.H),
                           uint32_t(out.nComp)};
        g.v.oneTime([&](vk::CommandBuffer c) {
            c.pushConstants(g.pipeLayout.get(),
                            vk::ShaderStageFlagBits::eCompute, 0, 8, ppc);
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                 g.pipeLayout.get(), 0, g.dset, {});
            c.bindPipeline(vk::PipelineBindPoint::eCompute, g.pPack.get());
            c.dispatch(uint32_t((outBytes + 1023) / 1024), 1, 1);
        });
        gTimes.down = std::chrono::duration<double>(clk::now() - t0).count();
        t0 = clk::now();
    } catch (...) {
        return false;
    }

    out.px.resize(outBytes);
    memcpy(out.px.data(), packBuf.mapped, outBytes);
    gTimes.down += std::chrono::duration<double>(clk::now() - t0).count();
    return true;
}

int main(int argc, char** argv) {
    int bench = 0;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bench") bench = atoi(argv[++i]);
        else pos.push_back(a);
    }
    if (pos.size() < 4) {
        fprintf(stderr,
            "usage: app [--bench N] in.ppm out.ppm WxH filter\n"
            "  filter: nearest box bilinear hamming bicubic lanczos\n");
        return 2;
    }
    Img in;
    if (!loadPpm(pos[0], in)) { fprintf(stderr, "bad input ppm\n"); return 1; }
    int ow, oh;
    if (sscanf(pos[2].c_str(), "%dx%d", &ow, &oh) != 2 || ow <= 0 || oh <= 0) {
        fprintf(stderr, "bad size %s\n", pos[2].c_str()); return 2;
    }
    Img out{ow, oh, in.nComp, {}};
    const FilterDef* f = nullptr;
    if (pos[3] == "nearest") f = nullptr;
    else {
        auto it = kFilters.find(pos[3]);
        if (it == kFilters.end()) {
            fprintf(stderr, "unknown filter %s\n", pos[3].c_str());
            return 2;
        }
        f = &it->second;
    }

    Gpu g;
    g.init();
    fprintf(stderr, "%s -> %dx%d %s\n", pos[0].c_str(), ow, oh,
            pos[3].c_str());

    if (bench > 0) {
        gpuResize(g, in, out, f);       // warm-up
        StageTimes acc{};
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < bench; ++i) {
            gpuResize(g, in, out, f);
            acc.conv += gTimes.conv; acc.alloc += gTimes.alloc;
            acc.up += gTimes.up; acc.h += gTimes.h; acc.v += gTimes.v;
            acc.down += gTimes.down;
        }
        double s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        double mp = double(ow) * oh * bench / 1e6;
        printf("%d resizes in %.2fs -> %.1f MPix out/s\n", bench, s, mp / s);
        fprintf(stderr,
            "  per-iter: rgba-conv %.1fms upload %.1fms bufs %.1fms "
            "H-pass %.1fms V-pass %.1fms readback %.1fms\n",
            acc.conv * 1e3 / bench, acc.up * 1e3 / bench,
            acc.alloc * 1e3 / bench, acc.h * 1e3 / bench,
            acc.v * 1e3 / bench, acc.down * 1e3 / bench);
        return 0;
    }
    if (!gpuResize(g, in, out, f)) return 1;
    savePpm(pos[1], out);
    printf("wrote %s\n", pos[1].c_str());
    return 0;
}
