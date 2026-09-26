// SPDX-License-Identifier: CC0-1.0
//
// Post 7: compute -> graphics in one frame. A compute shader writes
// a plasma pattern into a *storage image*; an image-layout barrier
// later, the graphics pipeline samples the same image on a fullscreen
// triangle. One descriptor set serves both stages (binding 0 =
// storage image for compute, binding 1 = combined sampler for
// fragment) because both descriptors point at the same VkImageView.
//
//   ./app                 windowed: plasma texture
//   ./app --headless o.ppm     one frame -> PPM for check.py
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"

#include <GLFW/glfw3.h>

constexpr uint32_t kTex = 256;

// ---------------- shaders -------------------------------------------
static const char* kComp = R"GLSL(
#version 460
layout(local_size_x = 16, local_size_y = 16) in;
// A storage image: writable pixels at integer coordinates, no
// sampler involved. The format qualifier (rgba8) is mandatory.
layout(set = 0, binding = 0, rgba8) uniform image2D img;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(p) / 255.0;
    // plasma: r = x gradient, g = y gradient, b = xor pattern
    float b = float((p.x ^ p.y) & 255) / 255.0;
    imageStore(img, p, vec4(uv.x, uv.y, b, 1.0));
}
)GLSL";

static const char* kVert = R"GLSL(
#version 460
layout(location = 0) out vec2 vUV;
void main() {
    vUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(vUV * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 460
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 1) uniform sampler2D tex;
void main() { outColor = texture(tex, vUV); }
)GLSL";

// ---------------- resources ------------------------------------------
struct Shared {
    vkmini::Vk::Image img;           // storage + sampled
    vk::UniqueSampler sampler;
    vk::UniqueDescriptorSetLayout setLayout;
    vk::UniqueDescriptorPool pool;
    vk::DescriptorSet set;
    vk::ImageLayout layout = vk::ImageLayout::eUndefined; // tracked
};

/// One image, two descriptor roles. The set layout has two bindings:
///   binding 0  eStorageImage          (compute stage)
///   binding 1  eCombinedImageSampler  (fragment stage)
/// and a single VkDescriptorSet points both at the same image view.
static Shared makeShared(vkmini::Vk& vk) {
    Shared o;
    o.img = vk.createImage(kTex, kTex, vk::Format::eR8G8B8A8Unorm,
                           vk::ImageUsageFlagBits::eStorage |
                               vk::ImageUsageFlagBits::eSampled);
    vk::SamplerCreateInfo si{};
    si.setMagFilter(vk::Filter::eNearest)
        .setMinFilter(vk::Filter::eNearest)
        .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeW(vk::SamplerAddressMode::eClampToEdge);
    o.sampler = vk.device->createSamplerUnique(si);

    std::array lbs{
        vk::DescriptorSetLayoutBinding{}
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eStorageImage)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute),
        vk::DescriptorSetLayoutBinding{}
            .setBinding(1)
            .setDescriptorType(
                vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment),
    };
    vk::DescriptorSetLayoutCreateInfo li{};
    li.setBindings(lbs);
    o.setLayout = vk.device->createDescriptorSetLayoutUnique(li);

    std::array pss{
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1},
        vk::DescriptorPoolSize{
            vk::DescriptorType::eCombinedImageSampler, 1},
    };
    vk::DescriptorPoolCreateInfo pi{};
    pi.setMaxSets(1).setPoolSizes(pss);
    o.pool = vk.device->createDescriptorPoolUnique(pi);
    vk::DescriptorSetAllocateInfo ai{};
    ai.setDescriptorPool(o.pool.get())
        .setSetLayouts(o.setLayout.get());
    o.set = vk.device->allocateDescriptorSets(ai).front();

    // Binding 0 refers to the eGeneral layout the compute stage
    // writes in; binding 1 refers to eShaderReadOnlyOptimal. The
    // layout recorded in the descriptor must match the image's
    // layout *when that binding is used*.
    vk::DescriptorImageInfo store{};
    store.setImageView(o.img.view.get())
        .setImageLayout(vk::ImageLayout::eGeneral);
    vk::DescriptorImageInfo sample{};
    sample.setSampler(o.sampler.get())
        .setImageView(o.img.view.get())
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    std::array writes{
        vk::WriteDescriptorSet{}.setDstSet(o.set).setDstBinding(0)
            .setDescriptorType(vk::DescriptorType::eStorageImage)
            .setImageInfo(store),
        vk::WriteDescriptorSet{}.setDstSet(o.set).setDstBinding(1)
            .setDescriptorType(
                vk::DescriptorType::eCombinedImageSampler)
            .setImageInfo(sample),
    };
    vk.device->updateDescriptorSets(writes, {});
    return o;
}

// ---------------- pipelines ------------------------------------------
static vk::UniquePipeline makeCompute(vkmini::Vk& vk,
                                    vk::UniquePipelineLayout& lay,
                                    vk::DescriptorSetLayout sl) {
    auto cs = vk.shader(kComp, shaderc_compute_shader, "plasma.comp");
    vk::PipelineLayoutCreateInfo li{};
    li.setSetLayouts(sl);
    lay = vk.device->createPipelineLayoutUnique(li);
    vk::PipelineShaderStageCreateInfo st{};
    st.setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(cs.get())
        .setPName("main");
    vk::ComputePipelineCreateInfo ci{};
    ci.setStage(st).setLayout(lay.get());
    auto res = vk.device->createComputePipelineUnique({}, ci);
    if (res.result != vk::Result::eSuccess)
        throw std::runtime_error("compute pipeline failed");
    return std::move(res.value);
}

static vk::UniquePipeline makeGraphics(
    vkmini::Vk& vk, vk::UniquePipelineLayout& lay,
    vk::DescriptorSetLayout sl, vk::Format fmt) {
    auto vs = vk.shader(kVert, shaderc_vertex_shader, "fs.vert");
    auto fs = vk.shader(kFrag, shaderc_fragment_shader, "fs.frag");
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
    cb.setColorWriteMask(
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    vk::PipelineColorBlendStateCreateInfo blend{};
    blend.setAttachments(cb);
    vk::PipelineLayoutCreateInfo li{};
    li.setSetLayouts(sl);
    lay = vk.device->createPipelineLayoutUnique(li);
    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(fmt);
    vk::GraphicsPipelineCreateInfo gi{};
    gi.setStages(stages)
        .setPVertexInputState(&vin)
        .setPInputAssemblyState(&ia)
        .setPViewportState(&vp)
        .setPDynamicState(&dyn)
        .setPRasterizationState(&rs)
        .setPMultisampleState(&ms)
        .setPColorBlendState(&blend)
        .setLayout(lay.get())
        .setPNext(&rendering);
    auto res = vk.device->createGraphicsPipelineUnique({}, gi);
    if (res.result != vk::Result::eSuccess)
        throw std::runtime_error("graphics pipeline failed");
    return std::move(res.value);
}

static void imgBarrier(vk::CommandBuffer c, vk::Image img,
                       vk::ImageLayout ol, vk::ImageLayout nl,
                       vk::AccessFlags src, vk::AccessFlags dst,
                       vk::PipelineStageFlags srcStage,
                       vk::PipelineStageFlags dstStage) {
    vk::ImageMemoryBarrier b{};
    b.setOldLayout(ol)
        .setNewLayout(nl)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(img)
        .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1,
                              0, 1})
        .setSrcAccessMask(src)
        .setDstAccessMask(dst);
    c.pipelineBarrier(srcStage, dstStage, {}, {}, {}, b);
}

/// The heart of the post — inside ONE command buffer:
///   1. transition the shared image sh.layout -> eGeneral
///   2. dispatch the compute shader writing into it
///   3. barrier eGeneral -> eShaderReadOnlyOptimal
///      (eShaderWrite -> eShaderRead, compute -> fragment)
///   4. draw fullscreen sampling the same image
/// The host tracks the image's layout in `sh.layout` so frame 2+
/// transitions from eShaderReadOnlyOptimal back to eGeneral.
static void record(vk::CommandBuffer c, Shared& sh,
                   vk::Pipeline cpipe, vk::PipelineLayout clay,
                   vk::Pipeline gpipe, vk::PipelineLayout glay,
                   vk::ImageView target, vk::Extent2D extent) {
    imgBarrier(c, sh.img.img.get(), sh.layout,
               vk::ImageLayout::eGeneral,
               sh.layout == vk::ImageLayout::eUndefined
                   ? vk::AccessFlags{}
                   : vk::AccessFlagBits::eShaderRead,
               vk::AccessFlagBits::eShaderWrite,
               vk::PipelineStageFlagBits::eTopOfPipe,
               vk::PipelineStageFlagBits::eComputeShader);

    c.bindPipeline(vk::PipelineBindPoint::eCompute, cpipe);
    c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, clay, 0,
                         sh.set, {});
    c.dispatch(kTex / 16, kTex / 16, 1);

    // compute write -> fragment read: layout + visibility in one
    // barrier. This is the entire synchronization this post adds.
    imgBarrier(c, sh.img.img.get(), vk::ImageLayout::eGeneral,
               vk::ImageLayout::eShaderReadOnlyOptimal,
               vk::AccessFlagBits::eShaderWrite,
               vk::AccessFlagBits::eShaderRead,
               vk::PipelineStageFlagBits::eComputeShader,
               vk::PipelineStageFlagBits::eFragmentShader);
    sh.layout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::RenderingAttachmentInfo att{};
    att.setImageView(target)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearValue{
            vk::ClearColorValue{std::array{0.f, 0.f, 0.f, 1.f}}});
    vk::RenderingInfo ri{};
    ri.setRenderArea({{0, 0}, extent})
        .setLayerCount(1)
        .setColorAttachments(att);
    c.beginRendering(ri);
    c.bindPipeline(vk::PipelineBindPoint::eGraphics, gpipe);
    c.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, glay, 0,
                         sh.set, {});
    vk::Viewport vp{0, 0, float(extent.width), float(extent.height),
                    0.f, 1.f};
    c.setViewport(0, vp);
    vk::Rect2D sc{{0, 0}, extent};
    c.setScissor(0, sc);
    c.draw(3, 1, 0, 0);
    c.endRendering();
}

int main(int argc, char** argv) {
    bool headless = argc > 1 && std::string(argv[1]) == "--headless";
    const char* ppmOut = argc > 2 ? argv[2] : "out.ppm";

    vkmini::Vk vk;
    GLFWwindow* win = nullptr;
    vk::UniqueSurfaceKHR surface;
    if (!headless) {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        win = glfwCreateWindow(640, 480, "vkmini p07", nullptr, nullptr);
    }
    std::vector<const char*> instExts;
    if (!headless) {
        uint32_t n = 0;
        const char** e = glfwGetRequiredInstanceExtensions(&n);
        instExts.assign(e, e + n);
    }
    vk.createInstance(instExts);
    if (!headless) {
        VkSurfaceKHR raw;
        glfwCreateWindowSurface(vk.instance.get(), win, nullptr, &raw);
        surface = vk::UniqueSurfaceKHR(raw, vk.instance.get());
    }
    vk.pickPhysicalDevice();
    vk.createDevice(headless ? std::vector<const char*>{}
                             : std::vector<const char*>{
                                   VK_KHR_SWAPCHAIN_EXTENSION_NAME});

    Shared sh = makeShared(vk);
    vk::UniquePipelineLayout clay, glay;
    auto cpipe = makeCompute(vk, clay, sh.setLayout.get());

    if (headless) {
        vkmini::Headless hl;
        hl.init(vk, 640, 480);
        auto gpipe = makeGraphics(vk, glay, sh.setLayout.get(),
                                  vk::Format::eR8G8B8A8Unorm);
        hl.render(vk::ImageLayout::eColorAttachmentOptimal,
                  [&](vk::CommandBuffer c) {
                      record(c, sh, cpipe.get(), clay.get(),
                             gpipe.get(), glay.get(),
                             hl.color.view.get(), hl.extent);
                  });
        hl.savePpm(ppmOut);
        printf("wrote %s\n", ppmOut);
        return 0;
    }

    vkmini::Swapchain sc;
    sc.create(vk, vk.phys, surface.get(), 640, 480);
    auto gpipe = makeGraphics(vk, glay, sh.setLayout.get(), sc.format);
    auto cmd = vk.device->allocateCommandBuffersUnique(
        {vk.pool.get(), vk::CommandBufferLevel::ePrimary, 1});
    auto acqSem = vk.device->createSemaphoreUnique({});
    auto doneSem = vk.device->createSemaphoreUnique({});
    auto fence = vk.device->createFenceUnique({});

    for (int f = 0; f < 240 && !glfwWindowShouldClose(win); ++f) {
        glfwPollEvents();
        vk.device->resetFences(fence.get());
        auto acq = vk.device->acquireNextImageKHR(
            sc.sc.get(), UINT64_MAX, acqSem.get(), {});
        vk::Image img = sc.images[acq.value];
        cmd[0]->reset();
        cmd[0]->begin(
            {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        // swapchain image: UNDEFINED -> COLOR_ATTACHMENT
        imgBarrier(cmd[0].get(), img, vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eColorAttachmentOptimal,
                   vk::AccessFlags{},
                   vk::AccessFlagBits::eColorAttachmentWrite,
                   vk::PipelineStageFlagBits::eTopOfPipe,
                   vk::PipelineStageFlagBits::eColorAttachmentOutput);
        record(cmd[0].get(), sh, cpipe.get(), clay.get(), gpipe.get(),
               glay.get(), sc.views[acq.value].get(), sc.extent);
        imgBarrier(cmd[0].get(), img,
                   vk::ImageLayout::eColorAttachmentOptimal,
                   vk::ImageLayout::ePresentSrcKHR,
                   vk::AccessFlagBits::eColorAttachmentWrite,
                   vk::AccessFlags{},
                   vk::PipelineStageFlagBits::eColorAttachmentOutput,
                   vk::PipelineStageFlagBits::eBottomOfPipe);
        cmd[0]->end();
        vk::PipelineStageFlags waitStage =
            vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::SubmitInfo si{};
        si.setWaitSemaphores(acqSem.get())
            .setWaitDstStageMask(waitStage)
            .setCommandBuffers(cmd[0].get())
            .setSignalSemaphores(doneSem.get());
        vk.queue.submit(si, fence.get());
        vk::PresentInfoKHR pi{};
        pi.setWaitSemaphores(doneSem.get())
            .setSwapchains(sc.sc.get())
            .setImageIndices(acq.value);
        (void)vk.queue.presentKHR(pi);
        (void)vk.device->waitForFences(fence.get(), VK_TRUE,
                                     UINT64_MAX);
    }
    vk.queue.waitIdle();
    sc.views.clear();
    sc.sc.reset();
    surface.reset();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
