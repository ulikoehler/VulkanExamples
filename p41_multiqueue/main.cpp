// SPDX-License-Identifier: CC0-1.0
//
// Real async queue work: a dedicated TRANSFER queue streams a
// texture upload while the GRAPHICS queue renders a pass that
// doesn't need it yet. A timeline semaphore orders "transfer done"
// -> "gfx may sample it"; an ownership transfer
// (release/acquire barriers) moves the image between families.
//
// Output: pass A (gfx) paints a dark quad; meanwhile the transfer
// queue fills a 8x8 checkerboard texture; pass B (gfx) waits on the
// timeline value then samples it fullscreen.
//
//   ./app --headless o.ppm     one frame -> PPM for check.py
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>
#include <set>

static const char* kVert = R"GLSL(
#version 460
layout(push_constant) uniform Push { vec2 ofs; float scl; } pc;
layout(location = 0) out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0,
                  (gl_VertexIndex == 2) ? 3.0 : -1.0);
    vUV = p * 0.5 + 0.5;
    gl_Position = vec4(p * pc.scl + pc.ofs, 0, 1);
}
)GLSL";

static const char* kFragSolid = R"GLSL(
#version 460
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(0.2, 0.2, 0.9, 1.0); }   // blue bg
)GLSL";

static const char* kFragTex = R"GLSL(
#version 460
layout(binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() { outColor = texture(tex, vUV); }
)GLSL";

int main() {
    setbuf(stdout, nullptr);
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    // enable timeline semaphores; we need TWO queue families:
    // gfx (as usual) + a dedicated transfer family if present.
    vk::PhysicalDeviceVulkan12Features f12{};
    f12.setTimelineSemaphore(VK_TRUE);
    vk.createDevice({}, &f12);

    auto qfams = vk.phys.getQueueFamilyProperties();
    uint32_t xferFam = vk.qfam;
    for (uint32_t i = 0; i < qfams.size(); ++i) {
        bool gfx = !!(qfams[i].queueFlags &
                      vk::QueueFlagBits::eGraphics);
        bool xfer = !!(qfams[i].queueFlags &
                       vk::QueueFlagBits::eTransfer);
        if (xfer && !gfx) {   // dedicated transfer family
            xferFam = i;
            break;
        }
    }
    // device was created only with the graphics family; if the
    // transfer family differs we must rebuild the device with both.
    vk::Queue xferQueue;
    vk::UniqueCommandPool xferPool;
    if (xferFam != vk.qfam) {
        std::set<uint32_t> fams{vk.qfam, xferFam};
        float prio = 1.f;
        std::vector<vk::DeviceQueueCreateInfo> qis;
        for (auto f : fams) {
            vk::DeviceQueueCreateInfo qi{};
            qi.setQueueFamilyIndex(f).setQueueCount(1)
                .setPQueuePriorities(&prio);
            qis.push_back(qi);
        }
        vk::PhysicalDeviceDynamicRenderingFeatures dyn{};
        dyn.setDynamicRendering(true).setPNext(&f12);
        vk::DeviceCreateInfo di{};
        di.setQueueCreateInfos(qis)
            .setPEnabledExtensionNames({})
            .setPNext(&dyn);
        vk.pool.reset();      // pool belongs to the OLD device —
        vk.queue = nullptr;   // destroy both BEFORE vk.device dies
        vk.device = vk.phys.createDeviceUnique(di);
        vk.queue = vk.device->getQueue(vk.qfam, 0);
        vk.pool = vk.device->createCommandPoolUnique(
            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
             vk.qfam});
        xferQueue = vk.device->getQueue(xferFam, 0);
        xferPool = vk.device->createCommandPoolUnique(
            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
             xferFam});
        printf("using dedicated transfer queue fam %u "
               "(gfx fam %u)\n", xferFam, vk.qfam);
    } else {
        xferQueue = vk.queue;
        xferPool = vk.device->createCommandPoolUnique(
            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
             xferFam});
        printf("no dedicated transfer family; using gfx fam %u\n",
               vk.qfam);
    }

    const uint32_t TW = 8, TH = 8;
    std::array<uint32_t, TW * TH> dat;
    for (uint32_t y = 0; y < TH; ++y)
        for (uint32_t x = 0; x < TW; ++x)
            dat[y * TW + x] =
                ((x + y) & 1) ? 0xFFFFFFFFu : 0xFF333333u;
    auto stg = vk.createBuffer(
        dat.size() * 4, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(stg.mapped, dat.data(), dat.size() * 4);

    // image: CONCURRENT sharing would also work but hides the lesson;
    // use exclusive + explicit ownership transfer.
    auto tex = vk.createImage(
        TW, TH, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eTransferDst |
            vk::ImageUsageFlagBits::eSampled);

    // timeline semaphore shared across both queues
    vk::SemaphoreTypeCreateInfo sti{};
    sti.setSemaphoreType(vk::SemaphoreType::eTimeline)
        .setInitialValue(0);
    vk::SemaphoreCreateInfo sci{};
    sci.setPNext(&sti);
    auto tl = vk.device->createSemaphoreUnique(sci);

    // ---- transfer-queue submission: copy + RELEASE ownership -----
    auto xcmds = vk.device->allocateCommandBuffersUnique(
        {xferPool.get(), vk::CommandBufferLevel::ePrimary, 1});
    auto& xc = xcmds[0];
    xc->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::ImageSubresourceRange sr{vk::ImageAspectFlagBits::eColor,
                                 0, 1, 0, 1};
    vk::ImageMemoryBarrier toDst{};
    toDst.setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(tex.img.get())
        .setSubresourceRange(sr)
        .setSrcAccessMask({})
        .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
    xc->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                        vk::PipelineStageFlagBits::eTransfer, {},
                        {}, {}, toDst);
    vk::BufferImageCopy cp{};
    cp.setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1})
        .setImageExtent({TW, TH, 1});
    xc->copyBufferToImage(stg.buf.get(), tex.img.get(),
                          vk::ImageLayout::eTransferDstOptimal, cp);
    // RELEASE to the graphics family (dst = gfx fam)
    vk::ImageMemoryBarrier rel{};
    rel.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcQueueFamilyIndex(xferFam)
        .setDstQueueFamilyIndex(vk.qfam)
        .setImage(tex.img.get())
        .setSubresourceRange(sr)
        .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    xc->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eBottomOfPipe,
                        {}, {}, {}, rel);
    xc->end();
    vk::TimelineSemaphoreSubmitInfo tlSub{};
    uint64_t one = 1;
    tlSub.setSignalSemaphoreValues(one);
    vk::SubmitInfo si{};
    si.setCommandBuffers(xc.get())
        .setSignalSemaphores(tl.get())
        .setPNext(&tlSub);
    xferQueue.submit(si);   // <- no wait: returns immediately
    printf("transfer submitted on xfer queue, gfx continues\n");

    // ---- descriptors + pipeline -----------------------------------
    auto sampler = vk.device->createSamplerUnique(
        {{},
         vk::Filter::eNearest,
         vk::Filter::eNearest,
         vk::SamplerMipmapMode::eNearest,
         vk::SamplerAddressMode::eClampToEdge,
         vk::SamplerAddressMode::eClampToEdge,
         vk::SamplerAddressMode::eClampToEdge});
    vk::DescriptorSetLayoutBinding binding{
        0, vk::DescriptorType::eCombinedImageSampler, 1,
        vk::ShaderStageFlagBits::eFragment};
    vk::DescriptorSetLayoutCreateInfo dslci{};
    dslci.setBindings(binding);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dslci);
    vk::DescriptorPoolSize ps{
        vk::DescriptorType::eCombinedImageSampler, 1};
    auto dpool = vk.device->createDescriptorPoolUnique(
        {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1,
         ps});
    vk::DescriptorSetLayout dslH = dsl.get();
    vk::DescriptorSetAllocateInfo dsai{};
    dsai.setDescriptorPool(dpool.get()).setSetLayouts(dslH);
    auto dsets = vk.device->allocateDescriptorSetsUnique(dsai);
    vk::DescriptorImageInfo dii{sampler.get(), tex.view.get(),
                                vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::WriteDescriptorSet w{};
    w.setDstSet(dsets[0].get())
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setImageInfo(dii);
    vk.device->updateDescriptorSets(w, {});

    auto mkPipe = [&](vk::UniqueShaderModule fs) {
        auto vs = vk.shader(kVert, shaderc_vertex_shader, "s.vert");
        vk::PipelineShaderStageCreateInfo stages[2];
        stages[0].setStage(vk::ShaderStageFlagBits::eVertex)
            .setModule(vs.get())
            .setPName("main");
        stages[1].setStage(vk::ShaderStageFlagBits::eFragment)
            .setModule(fs.get())
            .setPName("main");
        vk::PipelineVertexInputStateCreateInfo vin{};
        vk::PipelineInputAssemblyStateCreateInfo ia{};
        ia.setTopology(vk::PrimitiveTopology::eTriangleList);
        vk::PipelineViewportStateCreateInfo vp{};
        vp.setViewportCount(1).setScissorCount(1);
        std::array dynStates{vk::DynamicState::eViewport,
                             vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dyn{};
        dyn.setDynamicStates(dynStates);
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
        vk::PushConstantRange pcr{};
        pcr.setStageFlags(vk::ShaderStageFlagBits::eVertex)
            .setSize(16);
        vk::PipelineLayoutCreateInfo plci{};
        plci.setSetLayouts(dslH).setPushConstantRanges(pcr);
        auto layout = vk.device->createPipelineLayoutUnique(plci);
        vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
        vk::PipelineRenderingCreateInfo rendering{};
        rendering.setColorAttachmentFormats(cfmt);
        vk::GraphicsPipelineCreateInfo gi{};
        gi.setStages(stages)
            .setPVertexInputState(&vin)
            .setPInputAssemblyState(&ia)
            .setPViewportState(&vp)
            .setPDynamicState(&dyn)
            .setPRasterizationState(&rs)
            .setPMultisampleState(&ms)
            .setPColorBlendState(&blend)
            .setLayout(layout.get())
            .setPNext(&rendering);
        auto pr = vk.device->createGraphicsPipelineUnique({}, gi);
        struct P {
            vk::UniqueShaderModule vs;
            vk::UniquePipelineLayout layout;
            vk::UniquePipeline pipe;
        };
        return P{std::move(vs), std::move(layout),
                 std::move(pr.value)};
    };
    auto solidP = mkPipe(
        vk.shader(kFragSolid, shaderc_fragment_shader, "solid.frag"));
    auto texP = mkPipe(
        vk.shader(kFragTex, shaderc_fragment_shader, "tex.frag"));

    vkmini::Headless hl;
    hl.init(vk, 384, 384);
    hl.render(vk::ImageLayout::eColorAttachmentOptimal,
              [&](vk::CommandBuffer c) {
                  vk::RenderingAttachmentInfo att{};
                  att.setImageView(hl.color.view.get())
                      .setImageLayout(
                          vk::ImageLayout::eColorAttachmentOptimal)
                      .setLoadOp(vk::AttachmentLoadOp::eClear)
                      .setStoreOp(vk::AttachmentStoreOp::eStore)
                      .setClearValue(vk::ClearValue{
                          vk::ClearColorValue{
                              std::array{0.05f, 0.05f, 0.08f,
                                         1.f}}});
                  vk::RenderingInfo ri{};
                  ri.setRenderArea({{0, 0}, hl.extent})
                      .setLayerCount(1)
                      .setColorAttachments(att);
                  c.beginRendering(ri);
                  vk::Viewport v{0, 0, 384, 384, 0.f, 1.f};
                  c.setViewport(0, v);
                  vk::Rect2D sc{{0, 0}, hl.extent};
                  c.setScissor(0, sc);
                  // pass A: blue fullscreen quad (no texture dep)
                  c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 solidP.pipe.get());
                  float pc0[4] = {0, 0, 1.f, 0};
                  c.pushConstants(solidP.layout.get(),
                                  vk::ShaderStageFlagBits::eVertex,
                                  0, 16, pc0);
                  c.draw(3, 1, 0, 0);
                  c.endRendering();
              });
    // note: hl.render submits+waits — pass A ran BEFORE knowing the
    // texture upload finished. The transfer ran concurrently.

    // ---- pass B: wait timeline, ACQUIRE ownership, sample ---------
    // Host-side wait first: the acquire barrier must execute AFTER
    // the release completed. (A submit-level waitSemaphore would be
    // the GPU-side equivalent — post 17.)
    vk::SemaphoreWaitInfo wi{};
    wi.setSemaphores(tl.get()).setValues(one);
    vk.device->waitSemaphores(wi, UINT64_MAX);
    printf("timeline signaled; texture now gfx-owned\n");

    vk.oneTime([&](vk::CommandBuffer c) {
        vk::ImageMemoryBarrier acq{};
        acq.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
            .setNewLayout(
                vk::ImageLayout::eShaderReadOnlyOptimal)
            .setSrcQueueFamilyIndex(xferFam)
            .setDstQueueFamilyIndex(vk.qfam)
            .setImage(tex.img.get())
            .setSubresourceRange(sr)
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::eFragmentShader,
                          {}, {}, {}, acq);
    });

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
                      .setLayerCount(1)
                      .setColorAttachments(att);
                  c.beginRendering(ri);
                  vk::Viewport v{0, 0, 384, 384, 0.f, 1.f};
                  c.setViewport(0, v);
                  vk::Rect2D sc{{0, 0}, hl.extent};
                  c.setScissor(0, sc);
                  c.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 texP.pipe.get());
                  c.bindDescriptorSets(
                      vk::PipelineBindPoint::eGraphics,
                      texP.layout.get(), 0, dsets[0].get(), {});
                  float pc0[4] = {0, 0, 1.f, 0};
                  c.pushConstants(texP.layout.get(),
                                  vk::ShaderStageFlagBits::eVertex,
                                  0, 16, pc0);
                  c.draw(3, 1, 0, 0);
                  c.endRendering();
              });
    hl.savePpm("out.ppm");
    printf("wrote out.ppm\n");
    return 0;
}
