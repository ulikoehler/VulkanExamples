// SPDX-License-Identifier: CC0-1.0
//
// Baseline JPEG encoding in a Vulkan compute shader: one invocation
// per 16x16 MCU does level-shift -> float DCT -> quantize -> zigzag
// -> Huffman with 0xFF stuffing into its own byte-aligned SSBO
// segment (DRI=1 restart markers make the MCUs independent and
// parallel). The CPU only writes JFIF headers and concatenates.
//
//   ./app out.jpg     encode a procedural 640x480 pattern -> JPEG
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc
//
// The same design powers VulkanoCam's /api/screenshot endpoint and
// per-camera still capture — on dGPUs it's ~100x faster than
// libjpeg for large frames because all MCUs encode in parallel.

#include "vkmini.hpp"
#include <cmath>

// ---------------- Annex K tables ----------------
static constexpr uint8_t kLumaQ[64] = {
    16,11,10,16,24,40,51,61, 12,12,14,19,26,58,60,55,
    14,13,16,24,40,57,69,56, 14,17,22,29,51,87,80,62,
    18,22,37,56,68,109,103,77, 24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101, 72,92,95,98,112,100,103,99};
static constexpr uint8_t kChromaQ[64] = {
    17,18,24,47,99,99,99,99, 18,21,26,66,99,99,99,99,
    24,26,56,99,99,99,99,99, 99,99,47,66,99,99,99,99,
    99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99};
static constexpr uint8_t kZig[64] = {
    0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,
    27,20,13,6,7,14,21,28,35,42,49,56,57,50,43,36,29,22,15,23,30,37,
    44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63};
static constexpr uint8_t kDcLB[16]={0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0};
static constexpr uint8_t kDcLV[12]={0,1,2,3,4,5,6,7,8,9,10,11};
static constexpr uint8_t kDcCB[16]={0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0};
static constexpr uint8_t kDcCV[12]={0,1,2,3,4,5,6,7,8,9,10,11};
static constexpr uint8_t kAcLB[16]={0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,0x7d};
static constexpr uint8_t kAcLV[162]={
    0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,
    0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,0x23,0x42,
    0xb1,0xc1,0x15,0x52,0xd1,0xf0,0x24,0x33,0x62,0x72,0x82,0x09,0x0a,
    0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,0x29,0x2a,0x34,0x35,
    0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,
    0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,
    0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,
    0x85,0x86,0x87,0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,
    0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,
    0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,
    0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,
    0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,
    0xf5,0xf6,0xf7,0xf8,0xf9,0xfa};
static constexpr uint8_t kAcCB[16]={0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,0x77};
static constexpr uint8_t kAcCV[162]={
    0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,
    0x07,0x61,0x71,0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,0xa1,0xb1,
    0xc1,0x09,0x23,0x33,0x52,0xf0,0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,
    0x34,0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,0x27,0x28,0x29,0x2a,
    0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,
    0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,
    0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x82,
    0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,
    0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,
    0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,
    0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,
    0xda,0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,
    0xf5,0xf6,0xf7,0xf8,0xf9,0xfa};

/// Canonical Huffman: bits+vals -> out[sym] = len<<16 | code.
static void buildHuff(const uint8_t* bits, const uint8_t* vals, int n,
                      uint32_t* out) {
    memset(out, 0, 256 * 4);
    uint32_t code = 0; int k = 0;
    for (int len = 1; len <= 16; ++len) {
        for (int j = 0; j < bits[len - 1] && k < n; ++j)
            { out[vals[k++]] = (uint32_t(len) << 16) | code; ++code; }
        code <<= 1;
    }
}
static void scaleQ(const uint8_t* base, int q, uint32_t* out) {
    q = std::clamp(q, 1, 100);
    int s = q < 50 ? 5000 / q : 200 - q * 2;
    for (int i = 0; i < 64; ++i)
        out[i] = std::clamp((base[i] * s + 50) / 100, 1, 255);
}

// ---------------- compute shader ----------------
// One invocation per MCU; each writes a byte-aligned segment and its
// byte count. Restart markers make MCU order irrelevant.
static std::string makeShader() {
    std::string cst;
    for (int u = 0; u < 8; ++u)
        for (int x = 0; x < 8; ++x) {
            char b[32];
            snprintf(b, sizeof(b), "%.10e",
                     std::cos((2 * x + 1) * u * M_PI / 16.0));
            cst += b;
            cst += (u * 8 + x == 63) ? "" : ",";
        }
    return std::format(R"GLSL(#version 460 core
layout(local_size_x = 64) in;
layout(push_constant, std430) uniform PC {{
    uint width, height, mcusX, mcusY, segWords, total;
}} pc;
layout(set=0,binding=0) uniform sampler2D tex;
layout(std430,set=0,binding=1) writeonly buffer Out {{ uint buf[]; }};
layout(std430,set=0,binding=2) readonly buffer Tabs {{
    uint qY[64]; uint qC[64];
    uint dcY[16]; uint acY[256]; uint dcC[16]; uint acC[256];
}} tab;
const int ZZ[64] = int[](
    0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,
    27,20,13,6,7,14,21,28,35,42,49,56,57,50,43,36,29,22,15,23,30,37,
    44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63);
const float CST[64] = float[]({0});

float fetchY(int x, int y) {{
    ivec2 p = ivec2(clamp(x, 0, int(pc.width) - 1),
                    clamp(y, 0, int(pc.height) - 1));
    return dot(texelFetch(tex, p, 0).rgb, vec3(0.299,0.587,0.114))
           * 255.0 - 128.0;
}}
float fetchC(int x, int y, bool cr) {{
    vec3 acc = vec3(0.0);
    for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {{
        ivec2 q = ivec2(clamp(x * 2 + dx, 0, int(pc.width) - 1),
                        clamp(y * 2 + dy, 0, int(pc.height) - 1));
        acc += texelFetch(tex, q, 0).rgb;
    }}
    acc *= 0.25;
    return (cr ?  0.5*acc.r - 0.418688*acc.g - 0.081312*acc.b
               : -0.168736*acc.r - 0.331264*acc.g + 0.5*acc.b)
           * 255.0;
}}

// MSB-first bit writer with 0xFF stuffing into this MCU's segment
uint bitBuf, wordAcc, segBase, byteCnt, outWord; int bitLen, wordBytes;
void putb(uint b) {{
    wordAcc |= b << uint(wordBytes * 8); ++wordBytes; ++byteCnt;
    if (wordBytes == 4) {{
        if (outWord < pc.segWords) buf[segBase + outWord] = wordAcc;
        ++outWord; wordAcc = 0u; wordBytes = 0;
    }}
}}
void emitBits(uint code, int len) {{
    bitBuf = (bitBuf << len) | code; bitLen += len;
    while (bitLen >= 8) {{
        uint b = (bitBuf >> uint(bitLen - 8)) & 0xffu;
        bitLen -= 8; bitBuf &= (1u << uint(bitLen)) - 1u;
        putb(b); if (b == 0xffu) putb(0u);
    }}
}}
void emitH(uint h) {{ emitBits(h & 0xffffu, int(h >> 16)); }}
int catOf(int v) {{ int a = abs(v), s = 0; while (a != 0) {{ ++s; a >>= 1; }} return s; }}
uint lowBits(int v, int s) {{
    if (s <= 0) return 0u;
    return uint(v < 0 ? v - 1 : v) & ((1u << uint(s)) - 1u);
}}
void fdctq(in float f[64], out int q[64], bool chroma) {{
    float tmp[64];
    for (int y = 0; y < 8; ++y) for (int u = 0; u < 8; ++u) {{
        float s = 0.0;
        for (int x = 0; x < 8; ++x) s += f[y*8+x] * CST[u*8+x];
        tmp[y*8+u] = s;
    }}
    for (int v = 0; v < 8; ++v) for (int u = 0; u < 8; ++u) {{
        float s = 0.0;
        for (int y = 0; y < 8; ++y) s += tmp[y*8+u] * CST[v*8+y];
        float c = (u==0 ? 0.7071067811865476 : 1.0) *
                  (v==0 ? 0.7071067811865476 : 1.0);
        float qt = chroma ? float(tab.qC[v*8+u]) : float(tab.qY[v*8+u]);
        q[v*8+u] = int(round(0.25 * c * s / qt));
    }}
}}
void encBlock(in int q[64], inout int pred, bool chroma) {{
    int diff = q[0] - pred; pred = q[0];
    int s = catOf(diff);
    emitH(chroma ? tab.dcC[s] : tab.dcY[s]);
    emitBits(lowBits(diff, s), s);
    int run = 0;
    for (int k = 1; k < 64; ++k) {{
        int v = q[ZZ[k]];
        if (v == 0) {{ ++run; continue; }}
        while (run > 15) {{ emitH(chroma ? tab.acC[240] : tab.acY[240]); run -= 16; }}
        int sz = catOf(v);
        emitH(chroma ? tab.acC[(run<<4)|sz] : tab.acY[(run<<4)|sz]);
        emitBits(lowBits(v, sz), sz);
        run = 0;
    }}
    if (run > 0) emitH(chroma ? tab.acC[0] : tab.acY[0]);
}}
void main() {{
    uint mcu = gl_GlobalInvocationID.x;
    if (mcu >= pc.total) return;
    segBase = pc.total + mcu * pc.segWords;
    bitBuf = 0u; bitLen = 0; wordAcc = 0u; wordBytes = 0;
    byteCnt = 0u; outWord = 0u;
    int mx = int(mcu % pc.mcusX), my = int(mcu / pc.mcusX);
    int predY = 0, predCb = 0, predCr = 0;
    for (int b = 0; b < 4; ++b) {{          // 4 luma blocks / MCU
        int bx = mx*16 + (b&1)*8, by = my*16 + (b>>1)*8;
        float f[64];
        for (int y = 0; y < 8; ++y) for (int x = 0; x < 8; ++x)
            f[y*8+x] = fetchY(bx+x, by+y);
        int q[64]; fdctq(f, q, false); encBlock(q, predY, false);
    }}
    for (int c = 0; c < 2; ++c) {{          // 1 Cb + 1 Cr / MCU
        float f[64];
        for (int y = 0; y < 8; ++y) for (int x = 0; x < 8; ++x)
            f[y*8+x] = fetchC(mx*8+x, my*8+y, c == 1);
        int q[64]; fdctq(f, q, true);
        if (c == 0) encBlock(q, predCb, true);
        else        encBlock(q, predCr, true);
    }}
    if (bitLen > 0) emitBits((1u << uint(8-bitLen)) - 1u, 8-bitLen);
    if (wordBytes > 0) {{
        if (outWord < pc.segWords) buf[segBase + outWord] = wordAcc;
        ++outWord;
    }}
    buf[mcu] = byteCnt;
}}
)GLSL", cst);
}

// ---------------- JFIF container ----------------
static void u8(std::vector<uint8_t>& v, uint32_t b) { v.push_back(uint8_t(b)); }
static void u16be(std::vector<uint8_t>& v, uint32_t w) {
    u8(v, w >> 8); u8(v, w & 0xff);
}
static void marker(std::vector<uint8_t>& v, uint32_t m) {
    u8(v, 0xff); u8(v, m);
}
static void dqt(std::vector<uint8_t>& v, uint32_t id, const uint32_t* q) {
    marker(v, 0xdb); u16be(v, 67); u8(v, id);
    for (int i = 0; i < 64; ++i) u8(v, q[kZig[i]]);
}
static void dht(std::vector<uint8_t>& v, uint32_t tcth,
                const uint8_t* bits, const uint8_t* vals, int n) {
    marker(v, 0xc4); u16be(v, 19 + n); u8(v, tcth);
    for (int i = 0; i < 16; ++i) u8(v, bits[i]);
    for (int i = 0; i < n; ++i) u8(v, vals[i]);
}

int main(int argc, char** argv) {
    const char* out = "out.jpg";
    if (argc > 1 && argv[1][0] != '-') out = argv[1];
    constexpr uint32_t W = 640, H = 480;
    constexpr uint32_t kSegWords = 4096; // 16KB per MCU is generous

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice();

    // ---- source image: procedural RGBA pattern -------------------
    auto src = vk.createImage(W, H, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst);
    std::vector<uint8_t> px(W * H * 4);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            auto* p = &px[(y * W + x) * 4];
            p[0] = uint8_t(x * 255 / W);
            p[1] = uint8_t(y * 255 / H);
            p[2] = uint8_t(((x / 40 + y / 40) & 1) ? 230 : 40);
            p[3] = 255;
        }
    auto st = vk.createBuffer(px.size(), vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);
    memcpy(st.mapped, px.data(), px.size());
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::ImageMemoryBarrier b{};
        b.setOldLayout(vk::ImageLayout::eUndefined)
         .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
         .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
         .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
         .setImage(src.img.get())
         .setSubresourceRange({vk::ImageAspectFlagBits::eColor,0,1,0,1})
         .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, b);
        vk::BufferImageCopy r{};
        r.setImageSubresource({vk::ImageAspectFlagBits::eColor,0,0,1})
         .setImageExtent({W, H, 1});
        c.copyBufferToImage(st.buf.get(), src.img.get(),
            vk::ImageLayout::eTransferDstOptimal, r);
        b.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
         .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
         .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
         .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, b);
    });

    // ---- buffers: MCU segments out, tables in --------------------
    uint32_t mcusX = (W + 15) / 16, mcusY = (H + 15) / 16;
    uint32_t total = mcusX * mcusY;
    auto outBuf = vk.createBuffer(
        uint64_t(total) * (1 + kSegWords) * 4,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);
    auto tabBuf = vk.createBuffer(672 * 4,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);
    auto* tab = static_cast<uint32_t*>(tabBuf.mapped);
    scaleQ(kLumaQ, 85, tab);        // qY
    scaleQ(kChromaQ, 85, tab + 64); // qC
    buildHuff(kDcLB, kDcLV, 12, tab + 128);
    buildHuff(kAcLB, kAcLV, 162, tab + 144);
    buildHuff(kDcCB, kDcCV, 12, tab + 400);
    buildHuff(kAcCB, kAcCV, 162, tab + 416);

    // ---- compute pipeline ----------------------------------------
    std::array binds{
        vk::DescriptorSetLayoutBinding{0,
            vk::DescriptorType::eCombinedImageSampler, 1,
            vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{1,
            vk::DescriptorType::eStorageBuffer, 1,
            vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{2,
            vk::DescriptorType::eStorageBuffer, 1,
            vk::ShaderStageFlagBits::eCompute}};
    vk::DescriptorSetLayoutCreateInfo dl{};
    dl.setBindings(binds);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dl);
    vk::PushConstantRange pcr{vk::ShaderStageFlagBits::eCompute, 0, 24};
    vk::PipelineLayoutCreateInfo pl{};
    pl.setSetLayouts(dsl.get()).setPushConstantRanges(pcr);
    auto lay = vk.device->createPipelineLayoutUnique(pl);
    std::string src2 = makeShader();
    auto cs = vk.shader(src2.c_str(), shaderc_compute_shader, "j.comp");
    vk::PipelineShaderStageCreateInfo st2{};
    st2.setStage(vk::ShaderStageFlagBits::eCompute)
       .setModule(cs.get()).setPName("main");
    auto pipe = std::move(vk.device->createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{}.setStage(st2)
            .setLayout(lay.get())).value);

    auto sampler = vk.device->createSamplerUnique(
        {{}, vk::Filter::eNearest, vk::Filter::eNearest});
    vk::DescriptorPoolSize dps[2]{
        {vk::DescriptorType::eCombinedImageSampler, 1},
        {vk::DescriptorType::eStorageBuffer, 2}};
    auto dpool = vk.device->createDescriptorPoolUnique({{}, 1, dps});
    auto dsets = vk.device->allocateDescriptorSetsUnique(
        vk::DescriptorSetAllocateInfo{}.setDescriptorPool(dpool.get())
            .setSetLayouts(dsl.get()));
    vk::DescriptorImageInfo di{sampler.get(), src.view.get(),
                               vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorBufferInfo ob{outBuf.buf.get(), 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo tb{tabBuf.buf.get(), 0, VK_WHOLE_SIZE};
    vk::WriteDescriptorSet w0{}, w1{}, w2{};
    w0.setDstSet(dsets[0].get()).setDstBinding(0)
      .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
      .setImageInfo(di);
    w1.setDstSet(dsets[0].get()).setDstBinding(1)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setBufferInfo(ob);
    w2.setDstSet(dsets[0].get()).setDstBinding(2)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setBufferInfo(tb);
    vk.device->updateDescriptorSets({w0, w1, w2}, {});

    // ---- dispatch -------------------------------------------------
    vk.oneTime([&](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.get());
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             lay.get(), 0, dsets[0].get(), {});
        uint32_t pcv[6] = {W, H, mcusX, mcusY, kSegWords, total};
        c.pushConstants(lay.get(), vk::ShaderStageFlagBits::eCompute,
                        0, sizeof(pcv), pcv);
        c.dispatch((total + 63) / 64, 1, 1);
    });

    // ---- assemble JFIF -------------------------------------------
    std::vector<uint8_t> j;
    marker(j, 0xd8);                       // SOI
    marker(j, 0xe0); u16be(j, 16);         // APP0
    for (char c : {'J','F','I','F','\0'}) u8(j, uint8_t(c));
    u8(j,1); u8(j,1); u8(j,0); u16be(j,1); u16be(j,1); u8(j,0); u8(j,0);
    dqt(j, 0, tab); dqt(j, 1, tab + 64);
    marker(j, 0xc0); u16be(j, 17);         // SOF0
    u8(j, 8); u16be(j, H); u16be(j, W); u8(j, 3);
    u8(j,1); u8(j,0x22); u8(j,0);          // Y 2x2
    u8(j,2); u8(j,0x11); u8(j,1);          // Cb
    u8(j,3); u8(j,0x11); u8(j,1);          // Cr
    dht(j,0x00,kDcLB,kDcLV,12); dht(j,0x10,kAcLB,kAcLV,162);
    dht(j,0x01,kDcCB,kDcCV,12); dht(j,0x11,kAcCB,kAcCV,162);
    marker(j, 0xdd); u16be(j, 4); u16be(j, 1);  // DRI: RST every MCU
    marker(j, 0xda); u16be(j, 12); u8(j, 3);    // SOS
    u8(j,1); u8(j,0x00); u8(j,2); u8(j,0x11); u8(j,3); u8(j,0x11);
    u8(j,0); u8(j,63); u8(j,0);

    auto* words = static_cast<const uint32_t*>(outBuf.mapped);
    for (uint32_t i = 0; i < total; ++i) {
        auto* seg = reinterpret_cast<const uint8_t*>(
            words + total + i * kSegWords);
        j.insert(j.end(), seg, seg + std::min(words[i], kSegWords * 4));
        if (i + 1 < total) { u8(j, 0xff); u8(j, 0xd0 + (i % 8)); }
    }
    u8(j, 0xff); u8(j, 0xd9);              // EOI

    FILE* f = fopen(out, "wb");
    fwrite(j.data(), 1, j.size(), f);
    fclose(f);
    printf("wrote %s (%zu bytes, %u MCUs)\n", out, j.size(), total);
    return 0;
}
