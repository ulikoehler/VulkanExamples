// SPDX-License-Identifier: CC0-1.0
//
// VK_KHR_external_memory_fd — GPU memory shared between PROCESSES.
// Parent allocates a VkDeviceMemory marked exportable, fills the
// buffer on the GPU (vkCmdFillBuffer), exports the memory as a POSIX
// fd, then forks: the child (its own VkInstance/VkDevice, same GPU
// matched by driverUUID) imports the fd and reads the bytes.
//
// This is the primitive dma-buf importers, multi-process GL/Vulkan
// compositors and FFmpeg/VAAPI pipelines are built on.
//
//   ./app            parent renders + child verifies
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lglfw -lshaderc

#include "vkmini.hpp"
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

constexpr uint32_t kFill = 0xDEADBEEF;
constexpr vk::DeviceSize kSize = 4096;

static void makeVk(vkmini::Vk& vk) {
    vk.createInstance({});
    vk.pickPhysicalDevice();
    vk.createDevice({VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME});
}

/// driverUUID+deviceUUID uniquely identify the GPU across processes
/// — a dma-buf fd is only meaningful to the driver that created it.
static std::string gpuId(vkmini::Vk& vk) {
    auto p = vk.phys.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceIDProperties>();
    auto& id = p.get<vk::PhysicalDeviceIDProperties>();
    char buf[80];
    snprintf(buf, sizeof buf, "%02x%02x%02x%02x-%02x%02x%02x%02x",
             id.deviceUUID[0], id.deviceUUID[1], id.deviceUUID[2],
             id.deviceUUID[3], id.driverUUID[0], id.driverUUID[1],
             id.driverUUID[2], id.driverUUID[3]);
    return buf;
}

/// Pick the physical device whose UUID prefix matches `want`
/// (parent and child must land on the same GPU — see p47/p49 for
/// what happens when they don't).
static void pickByUuid(vkmini::Vk& vk, const std::string& want) {
    for (auto pd : vk.instance->enumeratePhysicalDevices())
        if (pd.getProperties().deviceType ==
                vk::PhysicalDeviceType::eCpu ||
            true) {
            vkmini::Vk probe;
            probe.phys = pd;
            if (gpuId(probe) == want) { vk.phys = pd; return; }
        }
    throw std::runtime_error("GPU with matching UUID not found");
}

int main(int argc, char** argv) {
    if (argc >= 3 && !strcmp(argv[1], "--child")) {
        // ---- child: import the fd, verify GPU-written bytes ----
        int fd = atoi(argv[2]);
        std::string want = getenv("VKIPC_UUID") ?: "";
        vkmini::Vk vk;
        vk.createInstance({});
        pickByUuid(vk, want);
        vk.createDevice({VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME});

        vk::BufferCreateInfo bi{};
        bi.setSize(kSize).setUsage(
            vk::BufferUsageFlagBits::eTransferSrc);
        vk::UniqueBuffer buf = vk.device->createBufferUnique(bi);
        auto req = vk.device->getBufferMemoryRequirements(buf.get());

        vk::ImportMemoryFdInfoKHR imp{};
        imp.setHandleType(
               vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT)
            .setFd(fd); // import CONSUMES the fd
        vk::MemoryAllocateInfo ai{};
        ai.setAllocationSize(kSize)
            .setMemoryTypeIndex(vk.memoryType(
                req.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible))
            .setPNext(&imp);
        auto mem = vk.device->allocateMemoryUnique(ai);
        vk.device->bindBufferMemory(buf.get(), mem.get(), 0);

        void* p = vk.device->mapMemory(mem.get(), 0, kSize);
        auto* w = static_cast<const uint32_t*>(p);
        size_t bad = 0;
        for (size_t i = 0; i < kSize / 4; ++i)
            if (w[i] != kFill) ++bad;
        vk.device->unmapMemory(mem.get());
        printf("child: imported fd=%d, %zu/%zu words == 0x%08x\n",
               fd, kSize / 4 - bad, kSize / 4, kFill);
        return bad ? 1 : 0;
    }

    // ---- parent: allocate exportable memory, GPU-fill, export ----
    vkmini::Vk vk;
    makeVk(vk);
    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr, vk.device.get());

    vk::BufferCreateInfo bi{};
    bi.setSize(kSize)
        .setUsage(vk::BufferUsageFlagBits::eTransferDst |
                  vk::BufferUsageFlagBits::eTransferSrc);
    vk::UniqueBuffer buf = vk.device->createBufferUnique(bi);
    auto req = vk.device->getBufferMemoryRequirements(buf.get());

    // Exportable: declare the handle type at ALLOCATION time. We want
    // host-visible so the child can simply map + memcmp.
    vk::ExportMemoryAllocateInfo em{};
    em.setHandleTypes(
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd |
        vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT);
    vk::MemoryAllocateInfo ai{};
    ai.setAllocationSize(req.size)
        .setMemoryTypeIndex(vk.memoryType(
            req.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent))
        .setPNext(&em);
    auto mem = vk.device->allocateMemoryUnique(ai);
    vk.device->bindBufferMemory(buf.get(), mem.get(), 0);

    // GPU writes the pattern — the child never sees CPU writes.
    vk.oneTime([&](vk::CommandBuffer c) {
        c.fillBuffer(buf.get(), 0, kSize, kFill);
    });

    // vkGetMemoryFdKHR: each call returns a NEW fd referencing the
    // same memory object (like dup()). The child gets it via fork
    // inheritance — in production it would go over a unix socket
    // (SCM_RIGHTS).
    vk::MemoryGetFdInfoKHR gfi{};
    gfi.setMemory(mem.get())
        .setHandleType(
            vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
    int fd = vk.device->getMemoryFdKHR(gfi, dldi);
    if (fd < 0)
        throw std::runtime_error("getMemoryFd failed");
    // The exported fd carries FD_CLOEXEC — it would be closed by the
    // execl() below. Clear it so the child inherits the handle (in
    // production you'd pass it over a unix socket instead).
    fcntl(fd, F_SETFD, 0);
    printf("parent: exported %llu bytes as fd %d, GPU-filled 0x%08x\n",
           (unsigned long long)req.size, fd, kFill);

    setenv("VKIPC_UUID", gpuId(vk).c_str(), 1);
    pid_t pid = fork();
    if (pid == 0) {
        char fdarg[16];
        snprintf(fdarg, sizeof fdarg, "%d", fd);
        execl("/proc/self/exe", argv[0], "--child", fdarg, nullptr);
        _exit(127);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    close(fd);
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
        fprintf(stderr, "child failed\n");
        return 1;
    }
    printf("done: GPU memory shared across processes, bytes verified\n");
    return 0;
}
