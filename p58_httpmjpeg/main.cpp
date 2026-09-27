// SPDX-License-Identifier: CC0-1.0
//
// HTTP MJPEG camera ingest -> FFmpeg decode -> Vulkan upload.
// The cheapest camera stream there is: multipart/x-mixed-replace —
// a never-ending HTTP response full of JPEG frames. This parses the
// stream with a raw socket (no curl), decodes frames with libavcodec,
// uploads the newest one to a VkImage and draws it.
//
//   ./app --headless o.ppm http://127.0.0.1:8137/cam
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app \
//             -lvulkan -lglfw -lshaderc -lavcodec -lavutil -lswscale
//
// Prints SKIP when the stream can't be reached/decoded.

#include "vkmini.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0,
                  (gl_VertexIndex == 2) ? 3.0 : -1.0);
    vUV = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0, 1);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(texture(tex, vUV).rgb, 1); }
)GLSL";

// ---------------- tiny HTTP/MJPEG client -------------------------
struct MjpegStream {
    int fd = -1;
    std::string boundary;       // without leading dashes
    std::vector<uint8_t> buf;   // unread stream bytes
    size_t scan = 0;

    ~MjpegStream() { if (fd >= 0) close(fd); }

    bool connectTo(const char* url) {
        // url: http://host[:port]/path
        std::string u = url;
        if (u.rfind("http://", 0) != 0) return false;
        u = u.substr(7);
        std::string hostport = u.substr(0, u.find('/'));
        std::string path = u.find('/') == std::string::npos
            ? "/" : u.substr(u.find('/'));
        std::string host = hostport, port = "80";
        if (auto c = hostport.find(':'); c != std::string::npos) {
            host = hostport.substr(0, c);
            port = hostport.substr(c + 1);
        }
        addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res))
            return false;
        fd = socket(res->ai_family, res->ai_socktype,
                    res->ai_protocol);
        timeval tv{5, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        if (::connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
            freeaddrinfo(res); close(fd); fd = -1; return false;
        }
        freeaddrinfo(res);
        std::string req = "GET " + path + " HTTP/1.0\r\n"
                          "Host: " + hostport + "\r\n\r\n";
        if (send(fd, req.data(), req.size(), 0) < 0) return false;

        // read HTTP headers (terminated by \r\n\r\n)
        std::string head;
        while (head.find("\r\n\r\n") == std::string::npos) {
            char tmp[4096];
            ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
            if (n <= 0) return false;
            head.append(tmp, n);
        }
        auto body = head.find("\r\n\r\n");
        buf.assign(head.begin() + body + 4, head.end());
        if (head.find("200") == std::string::npos) {
            fprintf(stderr, "HTTP error: %.40s\n", head.c_str());
            return false;
        }
        // Content-Type: multipart/x-mixed-replace;boundary=xxx
        auto bp = head.find("boundary=");
        if (bp == std::string::npos) bp = head.find("boundary=");
        if (bp != std::string::npos) {
            auto e = head.find_first_of("\r\n; \t", bp + 9);
            boundary = head.substr(bp + 9, e - (bp + 9));
            // trim optional quotes
            while (!boundary.empty() &&
                   (boundary.front() == '"' || boundary.front() == ' '))
                boundary.erase(boundary.begin());
            while (!boundary.empty() &&
                   (boundary.back() == '"' || boundary.back() == ' '))
                boundary.pop_back();
        }
        return true;
    }

    // next complete JPEG frame (boundary-delimited), or empty
    std::vector<uint8_t> nextFrame() {
        std::string delim = "\r\n--" + boundary;
        // also accept bare --boundary\n (some cameras)
        for (;;) {
            // find first delimiter
            auto pos = search(delim);
            if (pos == npos) { fill(); continue; }
            auto start = pos + delim.size();
            auto end = findFrom(delim, start);
            if (end == npos) { fill(); if (buf.size() > (1<<22)) return {};
                                       continue; }
            // frame body = start..end minus its part headers
            std::vector<uint8_t> part(buf.begin() + start,
                                      buf.begin() + end);
            // strip part headers (\r\n\r\n after Content-Length etc.)
            for (size_t i = 0; i + 3 < part.size(); ++i)
                if (part[i]=='\r'&&part[i+1]=='\n'&&part[i+2]=='\r'&&
                    part[i+3]=='\n') {
                    part.erase(part.begin(), part.begin()+i+4);
                    break;
                }
            buf.erase(buf.begin(), buf.begin() + end);
            // keep only actual JPEG data (SOI..EOI)
            // NB: a char* pattern would compare signed chars —
            // 0xff as char is -1 and never matches uint8_t 255!
            static const uint8_t soiPat[] = {0xff, 0xd8};
            auto soi = std::search(part.begin(), part.end(),
                                   soiPat, soiPat + 2);
            if (soi == part.end()) continue;
            return std::vector<uint8_t>(soi, part.end());
        }
    }
    static constexpr size_t npos = ~size_t(0);
    size_t search(const std::string& s) {
        return findFrom(s, scan ? scan : 0);
    }
    size_t findFrom(const std::string& s, size_t from) {
        auto it = std::search(buf.begin() + std::min(from, buf.size()),
                              buf.end(), s.begin(), s.end());
        return it == buf.end() ? npos : size_t(it - buf.begin());
    }
    void fill() {
        uint8_t tmp[65536];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) throw std::runtime_error("stream ended");
        buf.insert(buf.end(), tmp, tmp + n);
    }
};

// ---------------- FFmpeg MJPEG decode ----------------------------
struct RgbFrame {
    std::vector<uint8_t> rgba;
    int w = 0, h = 0;
};

static RgbFrame decodeJpeg(AVCodecContext* dec, SwsContext* sws,
                         const std::vector<uint8_t>& jpg) {
    AVPacket* pkt = av_packet_alloc();
    av_new_packet(pkt, jpg.size());
    memcpy(pkt->data, jpg.data(), jpg.size());
    AVFrame* f = av_frame_alloc();
    RgbFrame out;
    if (avcodec_send_packet(dec, pkt) == 0 &&
        avcodec_receive_frame(dec, f) == 0) {
        out.w = f->width; out.h = f->height;
        out.rgba.resize(size_t(out.w) * out.h * 4);
        SwsContext* s = sws_getCachedContext(
            sws, out.w, out.h, (AVPixelFormat)f->format,
            out.w, out.h, AV_PIX_FMT_RGBA, SWS_BILINEAR,
            nullptr, nullptr, nullptr);
        uint8_t* dst[4] = {out.rgba.data()};
        int dstStride[4] = {out.w * 4};
        sws_scale(s, f->data, f->linesize, 0, out.h, dst, dstStride);
    }
    av_frame_free(&f);
    av_packet_free(&pkt);
    return out;
}

int main(int argc, char** argv) {
    const char* ppm = "out.ppm";
    const char* url = "http://127.0.0.1:8137/cam";
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--headless") && i + 1 < argc)
            ppm = argv[++i];
        else if (argv[i][0] != '-')
            url = argv[i];

    // ---------------- connect + grab frames ----------------------
    MjpegStream stream;
    try {
        if (!stream.connectTo(url)) {
            printf("SKIP: cannot connect to %s\n", url); return 0;
        }
    } catch (...) {
        printf("SKIP: cannot connect to %s\n", url); return 0;
    }

    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
    AVCodecContext* dec = avcodec_alloc_context3(codec);
    if (avcodec_open2(dec, codec, nullptr) < 0) {
        printf("SKIP: no MJPEG decoder\n"); return 0;
    }

    // grab several frames, keep the last complete one — real cameras
    // emit ~5-30 fps, we just want a recent frame
    RgbFrame img;
    for (int i = 0; i < 4; ++i) {
        std::vector<uint8_t> jpg;
        try { jpg = stream.nextFrame(); }
        catch (...) { break; }
        if (jpg.empty()) break;
        auto f = decodeJpeg(dec, nullptr, jpg);
        if (!f.rgba.empty()) img = std::move(f);
    }
    if (img.rgba.empty()) {
        printf("SKIP: no decodable MJPEG frame\n"); return 0;
    }
    printf("got %dx%d MJPEG frame from %s\n", img.w, img.h, url);
    const uint32_t W = img.w, Hh = img.h;

    // ---------------- Vulkan -------------------------------------
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice();

    // upload RGBA to an optimal-tiled sampled image via staging
    auto tex = vk.createImage(
        W, Hh, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled |
        vk::ImageUsageFlagBits::eTransferDst);
    auto stage = vk.createBuffer(
        img.rgba.size(), vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);
    memcpy(stage.mapped, img.rgba.data(), img.rgba.size());
    vk.oneTime([&](vk::CommandBuffer c) {
        vk::ImageMemoryBarrier b{};
        b.setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(tex.img.get())
            .setSubresourceRange(
                {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})
            .setSrcAccessMask({})
            .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::eTransfer,
                          {}, {}, {}, b);
        vk::BufferImageCopy rc{};
        rc.setImageSubresource(
              {vk::ImageAspectFlagBits::eColor, 0, 0, 1})
            .setImageExtent({W, Hh, 1});
        c.copyBufferToImage(stage.buf.get(), tex.img.get(),
                            vk::ImageLayout::eTransferDstOptimal, rc);
        b.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
            .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                          vk::PipelineStageFlagBits::eFragmentShader,
                          {}, {}, {}, b);
    });

    auto sampler = vk.device->createSamplerUnique(
        {{}, vk::Filter::eLinear, vk::Filter::eLinear,
         vk::SamplerMipmapMode::eNearest,
         vk::SamplerAddressMode::eClampToEdge,
         vk::SamplerAddressMode::eClampToEdge,
         vk::SamplerAddressMode::eClampToEdge});
    vk::DescriptorSetLayoutBinding bind0{
        0, vk::DescriptorType::eCombinedImageSampler, 1,
        vk::ShaderStageFlagBits::eFragment};
    auto dsl = vk.device->createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo{}.setBindings(bind0));
    vk::DescriptorPoolSize ps{vk::DescriptorType::eCombinedImageSampler,1};
    auto dpool = vk.device->createDescriptorPoolUnique(
        {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1, ps});
    vk::DescriptorSetLayout dslH = dsl.get();
    vk::DescriptorSetAllocateInfo dsai{};
    dsai.setDescriptorPool(dpool.get()).setSetLayouts(dslH);
    auto dsets = vk.device->allocateDescriptorSetsUnique(dsai);
    vk::DescriptorImageInfo dii{sampler.get(), tex.view.get(),
                                vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::WriteDescriptorSet wset{dsets[0].get(), 0, 0,
                                vk::DescriptorType::eCombinedImageSampler,
                                dii};
    vk.device->updateDescriptorSets(wset, {});

    auto vs = vk.shader(kVert, shaderc_vertex_shader, "s.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "s.frag");
    vk::PipelineShaderStageCreateInfo stages[2];
    stages[0].setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(vs.get()).setPName("main");
    stages[1].setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(fs.get()).setPName("main");
    vk::PipelineVertexInputStateCreateInfo vin{};
    vk::PipelineInputAssemblyStateCreateInfo ia{};
    ia.setTopology(vk::PrimitiveTopology::eTriangleList);
    vk::PipelineViewportStateCreateInfo vps{};
    vps.setViewportCount(1).setScissorCount(1);
    std::array dynSt{vk::DynamicState::eViewport,
                     vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dyn{};
    dyn.setDynamicStates(dynSt);
    vk::PipelineRasterizationStateCreateInfo rs{};
    rs.setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eNone)
        .setFrontFace(vk::FrontFace::eCounterClockwise)
        .setLineWidth(1.0f);
    vk::PipelineMultisampleStateCreateInfo ms{};
    ms.setRasterizationSamples(vk::SampleCountFlagBits::e1);
    vk::PipelineColorBlendAttachmentState cb{};
    cb.setColorWriteMask(vk::ColorComponentFlagBits::eR |
                         vk::ColorComponentFlagBits::eG |
                         vk::ColorComponentFlagBits::eB |
                         vk::ColorComponentFlagBits::eA);
    vk::PipelineColorBlendStateCreateInfo blend{};
    blend.setAttachments(cb);
    vk::PipelineLayoutCreateInfo plci{};
    plci.setSetLayouts(dslH);
    auto layout = vk.device->createPipelineLayoutUnique(plci);
    vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(cfmt);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(stages)
        .setPVertexInputState(&vin).setPInputAssemblyState(&ia)
        .setPViewportState(&vps).setPDynamicState(&dyn)
        .setPRasterizationState(&rs).setPMultisampleState(&ms)
        .setPColorBlendState(&blend).setLayout(layout.get())
        .setPNext(&rendering);
    auto pres = vk.device->createGraphicsPipelineUnique({}, gi);
    if (pres.result != vk::Result::eSuccess)
        throw std::runtime_error("pipeline failed");
    auto pipe = std::move(pres.value);

    vkmini::Headless hl;
    hl.init(vk, W, Hh);
    hl.render(vk::ImageLayout::eColorAttachmentOptimal,
              [&](vk::CommandBuffer c) {
                  vk::RenderingAttachmentInfo att{};
                  att.setImageView(hl.color.view.get())
                      .setImageLayout(
                          vk::ImageLayout::eColorAttachmentOptimal)
                      .setLoadOp(vk::AttachmentLoadOp::eClear)
                      .setStoreOp(vk::AttachmentStoreOp::eStore);
                  vk::RenderingInfo ri{};
                  ri.setRenderArea({{0, 0}, hl.extent})
                      .setLayerCount(1).setColorAttachments(att);
                  c.beginRendering(ri);
                  vk::Viewport v{0, 0, float(W), float(Hh), 0.f, 1.f};
                  c.setViewport(0, v);
                  vk::Rect2D sc{{0,0}, hl.extent}; c.setScissor(0, sc);
                  c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 pipe.get());
                  c.bindDescriptorSets(
                      vk::PipelineBindPoint::eGraphics, layout.get(),
                      0, dsets[0].get(), {});
                  c.draw(3, 1, 0, 0);
                  c.endRendering();
              });
    hl.savePpm(ppm);
    printf("wrote %s\n", ppm);
    return 0;
}
