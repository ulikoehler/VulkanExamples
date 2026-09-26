// SPDX-License-Identifier: CC0-1.0
//
// VK_KHR_dynamic_rendering_local_read: with classic render passes,
// reading "what the previous draw wrote at THIS pixel" inside the
// same pass needed a subpass + input attachment. Dynamic rendering
// had no subpasses — this extension adds the mechanism back:
// declare attachment locations + input attachment indices, split
// the rendering with a BY_REGION barrier, and the second draw's
// fragment shader does subpassLoad() on the attachment it renders
// into. On tile-based GPUs this never leaves on-chip tile memory.
//
// Demo: draw1 paints the LEFT half red; by-region barrier; draw2
// reads the attachment per-pixel and tints green + 0.5*prev ->
// left half = orange-ish, right half = pure green.
//
//   ./app --headless o.ppm     one frame -> PPM for check.py
//   ./app                      windowed
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

static const char* kVert = R"GLSL(
#version 460
void main() {
    vec2 p = vec2((gl_VertexIndex << 1 & 2), gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0, 1);
}
)GLSL";

// draw 1: red on the left half only
static const char* kFragA = R"GLSL(
#version 460
layout(location = 0) out vec4 outColor;
void main() {
    if (gl_FragCoord.x < 64.0)
        outColor = vec4(1.0, 0.2, 0.2, 1.0);
    else
        discard;   // leave right half untouched
}
)GLSL";

// draw 2: green everywhere + 50% of what draw 1 left here
static const char* kFragB = R"GLSL(
#version 460
layout(input_attachment_index = 0, set = 0, binding = 0)
    uniform subpassInput prev;
layout(location = 0) out vec4 outColor;
void main() {
    vec3 p = subpassLoad(prev).rgb;
    outColor = vec4(p * 0.5 + vec3(0.0, 0.6, 0.0), 1.0);
}
)GLSL";

int main(int argc, char** argv) {
    bool headless = argc > 1 && std::string(argv[1]) == "--headless";
    const char* ppmOut = argc > 2 ? argv[2] : "out.ppm";
    (void)headless;

    vkmini::Vk vk;
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR lread{};
    lread.setDynamicRenderingLocalRead(true);
    vk.createDevice(
        {VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME},
        &lread);

    const uint32_t S = 128;
    vk::ImageSubresourceRange sr{vk::ImageAspectFlagBits::eColor,
                                 0, 1, 0, 1};
    // attachment is written as color AND read as input attachment ->
    // stays in eGeneral the whole time (the only layout valid for
    // both roles inside local-read rendering)
    auto img = vk.createImage(
        S, S, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eColorAttachment |
            vk::ImageUsageFlagBits::eInputAttachment |
            vk::ImageUsageFlagBits::eTransferSrc);

    // ---- pipelines -------------------------------------------------
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "s.vert");
    auto fsA = vk.shader(kFragA, shaderc_fragment_shader, "a.frag");
    auto fsB = vk.shader(kFragB, shaderc_fragment_shader, "b.frag");

    // descriptor: binding 0 = input attachment (same image!)
    vk::DescriptorSetLayoutBinding b0{
        0, vk::DescriptorType::eInputAttachment, 1,
        vk::ShaderStageFlagBits::eFragment};
    vk::DescriptorSetLayoutCreateInfo dslci{};
    dslci.setBindings(b0);
    auto dsl = vk.device->createDescriptorSetLayoutUnique(dslci);
    vk::DescriptorSetLayout dslH = dsl.get();
    vk::DescriptorPoolSize ps{vk::DescriptorType::eInputAttachment,
                              1};
    auto dpool = vk.device->createDescriptorPoolUnique(
        {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1,
         ps});
    vk::DescriptorSetAllocateInfo dsai{};
    dsai.setDescriptorPool(dpool.get()).setSetLayouts(dslH);
    auto dsets = vk.device->allocateDescriptorSetsUnique(dsai);
    vk::DescriptorImageInfo ii{VK_NULL_HANDLE, img.view.get(),
                               vk::ImageLayout::eGeneral};
    vk::WriteDescriptorSet w{};
    w.setDstSet(dsets[0].get())
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eInputAttachment)
        .setImageInfo(ii);
    vk.device->updateDescriptorSets(w, {});

    auto mkpipe = [&](vk::ShaderModule fs,
                      vk::PipelineLayout layout) {
        vk::PipelineShaderStageCreateInfo stages[2];
        stages[0].setStage(vk::ShaderStageFlagBits::eVertex)
            .setModule(vs.get())
            .setPName("main");
        stages[1].setStage(vk::ShaderStageFlagBits::eFragment)
            .setModule(fs)
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
        cb.setColorWriteMask(
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA);
        vk::PipelineColorBlendStateCreateInfo blend{};
        blend.setAttachments(cb);
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
            .setLayout(layout)
            .setPNext(&rendering);
        auto r = vk.device->createGraphicsPipelineUnique({}, gi);
        if (r.result != vk::Result::eSuccess)
            throw std::runtime_error("pipeline failed");
        return std::move(r.value);
    };

    auto layoutA = vk.device->createPipelineLayoutUnique({});
    vk::PipelineLayoutCreateInfo plciB{};
    plciB.setSetLayouts(dslH);
    auto layoutB =
        vk.device->createPipelineLayoutUnique(plciB);
    auto pipeA = mkpipe(fsA.get(), layoutA.get());
    auto pipeB = mkpipe(fsB.get(), layoutB.get());

    // ---- the frame -------------------------------------------------
    vkmini::Headless hl;
    hl.init(vk, S, S);
    auto cbuf = std::move(
        vk.device->allocateCommandBuffersUnique(
            {vk.pool.get(), vk::CommandBufferLevel::ePrimary, 1})
            .front());
    // extension functions aren't exported by libvulkan — resolve
    // via vkGetInstanceProcAddr (same mechanism as the debug-utils
    // post), then pass `dldi` to every KHR call.
    vk::detail::DispatchLoaderDynamic dldi(vk.instance.get(),
                                           vkGetInstanceProcAddr);
    vk::CommandBuffer c = cbuf.get();
    c.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // img: undefined -> general (stays general all frame)
    vk::ImageMemoryBarrier2 t1{};
    t1.setSrcStageMask(vk::PipelineStageFlagBits2::eNone)
        .setSrcAccessMask(vk::AccessFlagBits2::eNone)
        .setDstStageMask(
            vk::PipelineStageFlagBits2::eColorAttachmentOutput)
        .setDstAccessMask(
            vk::AccessFlagBits2::eColorAttachmentWrite |
            vk::AccessFlagBits2::eInputAttachmentRead)
        .setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eGeneral)
        .setImage(img.img.get())
        .setSubresourceRange(sr);
    vk::DependencyInfo dep1{};
    dep1.setImageMemoryBarriers(t1);
    c.pipelineBarrier2(dep1);

    // REQUIRED for local read: map color attachment 0 -> location 0
    // and input_attachment_index 0 -> color attachment 0
    vk::RenderingAttachmentLocationInfoKHR locInfo{};
    uint32_t locs[1] = {0};
    locInfo.setColorAttachmentLocations(locs);
    uint32_t idx[1] = {0};
    vk::RenderingInputAttachmentIndexInfoKHR idxInfo{
        idx};

    vk::RenderingAttachmentInfo att{};
    att.setImageView(img.view.get())
        .setImageLayout(vk::ImageLayout::eGeneral)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearValue{vk::ClearColorValue{
            std::array{0.f, 0.f, 0.f, 1.f}}});
    vk::RenderingInfo ri{};
    ri.setRenderArea({{0, 0}, {S, S}})
        .setLayerCount(1)
        .setColorAttachments(att);
    c.beginRendering(ri);
    vk::Viewport v{0, 0, float(S), float(S), 0.f, 1.f};
    c.setViewport(0, v);
    vk::Rect2D sc{{0, 0}, {S, S}};
    c.setScissor(0, sc);

    // draw 1: red on the left half
    c.setRenderingAttachmentLocationsKHR(locInfo, dldi);
    c.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeA.get());
    c.draw(3, 1, 0, 0);

    // THE local-read barrier: BY_REGION inside the rendering
    vk::MemoryBarrier2 mr{};
    mr.setSrcStageMask(
            vk::PipelineStageFlagBits2::eColorAttachmentOutput)
        .setSrcAccessMask(
            vk::AccessFlagBits2::eColorAttachmentWrite)
        .setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
        .setDstAccessMask(
            vk::AccessFlagBits2::eInputAttachmentRead);
    vk::DependencyInfo dep2{};
    dep2.setDependencyFlags(vk::DependencyFlagBits::eByRegion)
        .setMemoryBarriers(mr);
    c.pipelineBarrier2(dep2);

    // draw 2: reads what draw 1 wrote at THIS pixel
    c.setRenderingInputAttachmentIndicesKHR(idxInfo, dldi);
    c.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeB.get());
    c.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                         layoutB.get(), 0, dsets[0].get(), {});
    c.draw(3, 1, 0, 0);
    c.endRendering();

    // readback
    vk::ImageMemoryBarrier t3{};
    t3.setOldLayout(vk::ImageLayout::eGeneral)
        .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(img.img.get())
        .setSubresourceRange(sr)
        .setSrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
        .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
    c.pipelineBarrier(
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, t3);
    vk::BufferImageCopy rc{};
    rc.setImageSubresource(
          {vk::ImageAspectFlagBits::eColor, 0, 0, 1})
        .setImageExtent(vk::Extent3D{S, S, 1});
    c.copyImageToBuffer(img.img.get(),
                        vk::ImageLayout::eTransferSrcOptimal,
                        hl.readback.buf.get(), rc);
    c.end();
    vk::SubmitInfo si{};
    si.setCommandBuffers(c);
    auto f = vk.device->createFenceUnique({});
    vk.queue.submit(si, f.get());
    (void)vk.device->waitForFences(f.get(), true, UINT64_MAX);
    hl.savePpm(ppmOut);
    printf("wrote %s\n", ppmOut);
    return 0;
}
