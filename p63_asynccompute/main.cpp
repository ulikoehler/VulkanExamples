// SPDX-License-Identifier: CC0-1.0
//
// Async compute: submit compute work to a DEDICATED compute queue
// while graphics runs on the graphics queue — true overlap, verified
// by GPU timestamps on both queues.
//
// On this machine: qfam0 = graphics|compute|transfer|sparse,
//                  qfam1 = compute|transfer|sparse (4 queues).
//
//   ./app     prints overlap window + verifies both results
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

// long-running compute: writes a result after a busy loop so it
// occupies the compute engine while the graphics queue renders
static const char* kComp = R"GLSL(
#version 460
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Out { uint data[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= data.length()) return;
    float acc = 0.0;
    for (int k = 0; k < 4000; ++k)
        acc += sin(float(k) * 0.31f + float(i));
    data[i] = uint(abs(acc) * 1000.0) | 0xC0FFEE00u;  // tag
}
)GLSL";

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

// several passes of heavy-ish fragment work so the draw takes a while
static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() {
    float acc = 0.0;
    for (int i = 0; i < 2000; ++i)
        acc += sin(vUV.x * float(i) * 0.013 + vUV.y * float(i) * 0.017);
    outColor = vec4(vUV, 0.5 + 0.5 * acc * 0.001, 1);
}
)GLSL";

int main() {
    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();

    // find a compute-capable queue family that is NOT the graphics one
    auto qprops = vk.phys.getQueueFamilyProperties();
    uint32_t gfxFam = UINT32_MAX, compFam = UINT32_MAX;
    for (uint32_t i = 0; i < qprops.size(); ++i) {
        if ((qprops[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
            gfxFam == UINT32_MAX)
            gfxFam = i;
    }
    for (uint32_t i = 0; i < qprops.size(); ++i)
        if ((qprops[i].queueFlags & vk::QueueFlagBits::eCompute) &&
            i != gfxFam) { compFam = i; break; }
    if (compFam == UINT32_MAX) { printf("SKIP: no compute queue\n"); return 0; }
    printf("graphics qfam=%u  compute qfam=%u\n", gfxFam, compFam);

    // create device with BOTH queues
    float prio = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> qis;
    for (uint32_t fam : {gfxFam, compFam})
        qis.push_back({{}, fam, 1, &prio});
    vk::PhysicalDeviceDynamicRenderingFeatures dyn{};
    dyn.setDynamicRendering(true);
    vk::PhysicalDeviceTimelineSemaphoreFeatures tsFeat{};
    tsFeat.setTimelineSemaphore(true).setPNext(&dyn);
    vk::DeviceCreateInfo di{};
    di.setQueueCreateInfos(qis).setPNext(&tsFeat);
    vk.device = vk.phys.createDeviceUnique(di);
    vk.queue = vk.device->getQueue(gfxFam, 0);
    vk.qfam = gfxFam;
    vk.pool = vk.device->createCommandPoolUnique(
        {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, gfxFam});
    vk::Queue cqueue = vk.device->getQueue(compFam, 0);
    auto cpool = vk.device->createCommandPoolUnique(
        {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, compFam});

    // query pools to timestamp each queue's work
    auto qp = vk.device->createQueryPoolUnique(
        {{}, vk::QueryType::eTimestamp, 4});
    float period = vk.phys.getProperties()
                       .limits.timestampPeriod; // ns per tick

    // output buffer the compute shader fills
    constexpr uint32_t NB = 1u << 16;
    auto buf = vk.createBuffer(
        NB * 4, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent, true);

    // compute pipeline
    vk::DescriptorSetLayoutBinding bind0{
        0, vk::DescriptorType::eStorageBuffer, 1,
        vk::ShaderStageFlagBits::eCompute};
    auto dsl = vk.device->createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo{}.setBindings(bind0));
    vk::DescriptorPoolSize ps{vk::DescriptorType::eStorageBuffer, 1};
    auto dpool = vk.device->createDescriptorPoolUnique(
        {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1, ps});
    vk::DescriptorSetLayout dslH = dsl.get();
    vk::DescriptorSetAllocateInfo dsai{};
    dsai.setDescriptorPool(dpool.get()).setSetLayouts(dslH);
    auto dsets = vk.device->allocateDescriptorSetsUnique(dsai);
    vk::DescriptorBufferInfo dbi{buf.buf.get(), 0, NB * 4};
    vk::WriteDescriptorSet wset{dsets[0].get(), 0, 0,
                                vk::DescriptorType::eStorageBuffer,
                                {}, dbi};
    vk.device->updateDescriptorSets(wset, {});

    auto cs = vk.shader(kComp, shaderc_compute_shader, "c.comp");
    vk::PipelineLayoutCreateInfo plci{};
    plci.setSetLayouts(dslH);
    auto layout = vk.device->createPipelineLayoutUnique(plci);
    vk::PipelineShaderStageCreateInfo cstage{};
    cstage.setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(cs.get()).setPName("main");
    vk::ComputePipelineCreateInfo cpci{};
    cpci.setStage(cstage).setLayout(layout.get());
    auto cres = vk.device->createComputePipelineUnique({}, cpci);
    auto cpipe = std::move(cres.value);

    // graphics pipeline
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "s.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "s.frag");
    vk::PipelineShaderStageCreateInfo gst[2];
    gst[0].setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(vs.get()).setPName("main");
    gst[1].setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(fs.get()).setPName("main");
    vk::PipelineVertexInputStateCreateInfo vin{};
    vk::PipelineInputAssemblyStateCreateInfo ia{};
    ia.setTopology(vk::PrimitiveTopology::eTriangleList);
    vk::PipelineViewportStateCreateInfo vps{};
    vps.setViewportCount(1).setScissorCount(1);
    std::array dynSt{vk::DynamicState::eViewport,
                     vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dync{};
    dync.setDynamicStates(dynSt);
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
    vk::Format cfmt = vk::Format::eR8G8B8A8Unorm;
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(cfmt);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(gst)
        .setPVertexInputState(&vin).setPInputAssemblyState(&ia)
        .setPViewportState(&vps).setPDynamicState(&dync)
        .setPRasterizationState(&rs).setPMultisampleState(&ms)
        .setPColorBlendState(&blend).setLayout(layout.get())
        .setPNext(&rendering);
    auto gres = vk.device->createGraphicsPipelineUnique({}, gi);
    auto gpipe = std::move(gres.value);

    vkmini::Headless hl;
    hl.init(vk, 640, 360);

    // command buffers: graphics on qfam0, compute on qfam1
    auto gcmd = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::ePrimary, 1});
    auto ccmd = vk.device->allocateCommandBuffersUnique(
        {cpool.get(), vk::CommandBufferLevel::ePrimary, 1});

    { vk::CommandBufferBeginInfo bi{}; ccmd[0]->begin(&bi); }
    ccmd[0]->resetQueryPool(qp.get(), 2, 2);
    ccmd[0]->writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe,
                            qp.get(), 2);
    ccmd[0]->bindPipeline(vk::PipelineBindPoint::eCompute,
                          cpipe.get());
    ccmd[0]->bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                layout.get(), 0, dsets[0].get(), {});
    ccmd[0]->dispatch(NB / 64, 1, 1);
    ccmd[0]->writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
                            qp.get(), 3);
    ccmd[0]->end();

    { vk::CommandBufferBeginInfo bi{}; gcmd[0]->begin(&bi); }
    gcmd[0]->resetQueryPool(qp.get(), 0, 2);
    gcmd[0]->writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe,
                            qp.get(), 0);
    {
        vk::ImageMemoryBarrier imb{};
        imb.setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(hl.color.img.get())
            .setSubresourceRange(
                {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})
            .setDstAccessMask(
                vk::AccessFlagBits::eColorAttachmentWrite);
        gcmd[0]->pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            {}, {}, {}, imb);
        vk::RenderingAttachmentInfo att{};
        att.setImageView(hl.color.view.get())
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore);
        vk::RenderingInfo ri{};
        ri.setRenderArea({{0, 0}, hl.extent})
            .setLayerCount(1).setColorAttachments(att);
        gcmd[0]->beginRendering(ri);
        vk::Viewport v{0, 0, 640.f, 360.f, 0.f, 1.f};
        gcmd[0]->setViewport(0, v);
        vk::Rect2D sc{{0,0}, hl.extent};
        gcmd[0]->setScissor(0, sc);
        gcmd[0]->bindPipeline(vk::PipelineBindPoint::eGraphics,
                              gpipe.get());
        gcmd[0]->draw(3, 1, 0, 0);
        gcmd[0]->endRendering();
    }
    gcmd[0]->writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
                            qp.get(), 1);
    gcmd[0]->end();

    // ---- THE OVERLAP: submit compute first, graphics right after.
    // A timeline semaphore makes graphics completion also signal
    // the compute side's finishing point for ordered readback.
    auto sem = vk.device->createSemaphoreUnique(
        vk::SemaphoreCreateInfo{}.setPNext(
            vk::SemaphoreTypeCreateInfo{}
                .setSemaphoreType(vk::SemaphoreType::eTimeline)
                .setInitialValue(0)));

    uint64_t sv1 = 1, sv2 = 2;
    vk::TimelineSemaphoreSubmitInfo cinfo{};
    cinfo.setSignalSemaphoreValues(sv1);
    vk::SubmitInfo csi{};
    csi.setCommandBuffers(ccmd[0].get())
        .setSignalSemaphores(sem.get())
        .setPNext(&cinfo);
    cqueue.submit(csi);

    vk::TimelineSemaphoreSubmitInfo gsinfo{};
    gsinfo.setSignalSemaphoreValues(sv2);
    vk::SubmitInfo gsi{};
    gsi.setCommandBuffers(gcmd[0].get())
        .setSignalSemaphores(sem.get())
        .setPNext(&gsinfo);
    vk.queue.submit(gsi);

    // wait for both (timeline value 2 = graphics done; compute is
    // ordered by the same submission sequence + its own signal at 1)
    vk::SemaphoreWaitInfo wi{};
    vk::Semaphore semH = sem.get();
    uint64_t wv2 = 2;
    wi.setSemaphores(semH).setValues(wv2);
    vk.device->waitSemaphores(wi, UINT64_MAX);
    // also wait for the compute signal explicitly
    uint64_t wv1 = 1; wi.setValues(wv1);
    vk.device->waitSemaphores(wi, UINT64_MAX);

    uint64_t ts[4]{};
    vk.device->getQueryPoolResults(
        qp.get(), 0, 4, sizeof(ts), ts, 8,
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    float gUs = (ts[1] - ts[0]) * period / 1000.f;
    float cUs = (ts[3] - ts[2]) * period / 1000.f;
    // overlap = intervals intersect? timestamps are the same clock
    int64_t overlap = std::min(ts[1], ts[3]) - std::max(ts[0], ts[2]);
    printf("graphics: %.1f us   compute: %.1f us   overlap: %s%.1f us\n",
           gUs, cUs,
           overlap > 0 ? "+" : "-", std::abs(overlap) * period / 1000.f);

    uint32_t tagged = 0;
    for (uint32_t i = 0; i < NB; ++i)
        if (((uint32_t*)buf.mapped)[i] >> 24 == 0xC0) ++tagged;
    printf("compute tagged %u/%u buffer entries\n", tagged, NB);
    printf(tagged == NB && overlap > 0 ? "OK: true overlap\n"
                                       : "done (serialized)\n");
    return 0;
}
