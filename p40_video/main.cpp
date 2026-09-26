// SPDX-License-Identifier: CC0-1.0
//
// Vulkan Video decode, minimal end-to-end: one H.264 IDR frame
// decoded by VK_KHR_video_decode_queue + VK_KHR_video_decode_h264 on
// a dedicated VIDEO queue family. Vulkan Video is "codec-agnostic
// plumbing": the APP parses SPS/PPS (Exp-Golomb) into the Std*
// structs — only slice data goes through the HW decoder.
//
//   ffmpeg -f lavfi -i "color=red:size=64x64:rate=1" -frames:v 1 \
//     -c:v libx264 -profile:v baseline -pix_fmt yuv420p -g 1 -bf 0 \
//     -f h264 one_red.h264
//   ./app one_red.h264 out.ppm
//
// Build: g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc

#include "vkmini.hpp"
#include <fstream>
#include <cstring>
#include <set>

// ---------------- Exp-Golomb bit reader ---------------------------
struct Br {
    const uint8_t* p; size_t n; int bit = 0;
    uint32_t u(int nb) {
        uint32_t v = 0;
        for (int i = 0; i < nb; ++i) {
            int byte = bit >> 3;
            if (byte >= (int)n) return v;
            // strip emulation-prevention 0x03 after 00 00 on the fly:
            // (RBSP already unescaped by caller)
            v = (v << 1) | ((p[byte] >> (7 - (bit & 7))) & 1);
            ++bit;
        }
        return v;
    }
    uint32_t ue() {
        int z = 0;
        while (u(1) == 0 && bit < (int)n * 8) ++z;
        return (1u << z) - 1 + u(z);
    }
    int32_t se() {
        uint32_t k = ue();
        return (k & 1) ? (int32_t)(k + 1) / 2 : -(int32_t)(k / 2);
    }
};

static std::vector<uint8_t> unescape(const uint8_t* p, size_t n) {
    std::vector<uint8_t> o;
    for (size_t i = 0; i < n; ++i) {
        if (i >= 2 && p[i] == 3 && o.size() >= 2 &&
            o[o.size() - 1] == 0 && o[o.size() - 2] == 0) {
            // skip emulation prevention byte
        } else
            o.push_back(p[i]);
    }
    return o;
}

struct Stream {
    StdVideoH264SequenceParameterSet sps{};
    StdVideoH264PictureParameterSet pps{};
    std::vector<uint8_t> slices;        // all slice NALs, start codes incl.
    std::vector<uint32_t> sliceOfs;
    int idr = 0;
};

static void parseSps(const std::vector<uint8_t>& rb, Stream& st) {
    Br b{rb.data(), rb.size()};
    uint32_t profile = b.u(8);
    uint8_t constraint = (uint8_t)b.u(8);
    st.sps.level_idc = (StdVideoH264LevelIdc)b.u(8); // idc value maps 1:1
    st.sps.seq_parameter_set_id = b.ue();
    st.sps.profile_idc = (StdVideoH264ProfileIdc)profile;
    st.sps.flags.constraint_set0_flag = constraint & 0x80;
    st.sps.flags.constraint_set1_flag = constraint & 0x40;
    st.sps.flags.constraint_set2_flag = constraint & 0x20;
    st.sps.flags.constraint_set3_flag = constraint & 0x10;
    st.sps.flags.constraint_set4_flag = constraint & 0x08;
    st.sps.flags.constraint_set5_flag = constraint & 0x04;
    if (profile >= 100) {  // high-family extensions
        uint32_t cf = b.ue();
        st.sps.chroma_format_idc = (StdVideoH264ChromaFormatIdc)cf;
        if (cf == 3) st.sps.flags.separate_colour_plane_flag = b.u(1);
        st.sps.bit_depth_luma_minus8 = b.ue();
        st.sps.bit_depth_chroma_minus8 = b.ue();
        st.sps.flags.qpprime_y_zero_transform_bypass_flag = b.u(1);
        if (b.u(1)) {  // seq_scaling_matrix_present
            st.sps.flags.seq_scaling_matrix_present_flag = 1;
            int n = (cf == 3) ? 12 : 8;
            for (int i = 0; i < n; ++i)
                if (b.u(1)) {  // scaling list present: skip deltas
                    int size = i < 6 ? 16 : 64, last = 8, next = 8;
                    for (int j = 0; j < size && next; ++j) {
                        next = (last + b.se() + 256) % 256;
                        last = next ? next : last;
                    }
                }
        }
    } else {
        st.sps.chroma_format_idc =
            STD_VIDEO_H264_CHROMA_FORMAT_IDC_420;
    }
    st.sps.log2_max_frame_num_minus4 = b.ue();
    uint32_t poc = b.ue();
    st.sps.pic_order_cnt_type = (StdVideoH264PocType)poc;
    if (poc == 0)
        st.sps.log2_max_pic_order_cnt_lsb_minus4 = b.ue();
    else if (poc == 1) {
        st.sps.flags.delta_pic_order_always_zero_flag = b.u(1);
        st.sps.offset_for_non_ref_pic = b.se();
        st.sps.offset_for_top_to_bottom_field = b.se();
        uint32_t c = b.ue();
        st.sps.num_ref_frames_in_pic_order_cnt_cycle = c;
        static std::vector<int32_t> keep;   // outlives: struct holds ptr
        keep.resize(c);
        for (auto& v : keep) v = b.se();
        st.sps.pOffsetForRefFrame = c ? keep.data() : nullptr;
    }
    st.sps.max_num_ref_frames = b.ue();
    st.sps.flags.gaps_in_frame_num_value_allowed_flag = b.u(1);
    st.sps.pic_width_in_mbs_minus1 = b.ue();
    st.sps.pic_height_in_map_units_minus1 = b.ue();
    st.sps.flags.frame_mbs_only_flag = b.u(1);
    if (!st.sps.flags.frame_mbs_only_flag)
        st.sps.flags.mb_adaptive_frame_field_flag = b.u(1);
    st.sps.flags.direct_8x8_inference_flag = b.u(1);
    if (b.u(1)) {  // frame_cropping
        st.sps.flags.frame_cropping_flag = 1;
        st.sps.frame_crop_left_offset = b.ue();
        st.sps.frame_crop_right_offset = b.ue();
        st.sps.frame_crop_top_offset = b.ue();
        st.sps.frame_crop_bottom_offset = b.ue();
    }
    st.sps.flags.vui_parameters_present_flag = b.u(1);
}

static void parsePps(const std::vector<uint8_t>& rb, Stream& st) {
    Br b{rb.data(), rb.size()};
    st.pps.pic_parameter_set_id = b.ue();
    st.pps.seq_parameter_set_id = b.ue();
    st.pps.flags.entropy_coding_mode_flag = b.u(1);
    st.pps.flags.bottom_field_pic_order_in_frame_present_flag = b.u(1);
    uint32_t nsg = b.ue();               // num_slice_groups_minus1
    if (nsg > 0) throw std::runtime_error("slice groups unsupported");
    st.pps.num_ref_idx_l0_default_active_minus1 = b.ue();
    st.pps.num_ref_idx_l1_default_active_minus1 = b.ue();
    st.pps.flags.weighted_pred_flag = b.u(1);
    st.pps.weighted_bipred_idc =
        (StdVideoH264WeightedBipredIdc)b.u(2);
    st.pps.pic_init_qp_minus26 = b.se();
    st.pps.pic_init_qs_minus26 = b.se();
    st.pps.chroma_qp_index_offset = b.se();
    st.pps.flags.deblocking_filter_control_present_flag = b.u(1);
    st.pps.flags.constrained_intra_pred_flag = b.u(1);
    st.pps.flags.redundant_pic_cnt_present_flag = b.u(1);
    if (b.bit + 4 < (int)rb.size() * 8) {   // optional tail fields
        st.pps.flags.transform_8x8_mode_flag = b.u(1);
        st.pps.flags.pic_scaling_matrix_present_flag = b.u(1);
        if (st.pps.flags.pic_scaling_matrix_present_flag)
            throw std::runtime_error("pps scaling list unsupported");
        st.pps.second_chroma_qp_index_offset = b.se();
    }
}

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    const char* inPath = argc > 1 ? argv[1] : "one_red.h264";
    const char* ppmOut = argc > 2 ? argv[2] : "out.ppm";
    std::ifstream fi(inPath, std::ios::binary);
    std::vector<uint8_t> bs((std::istreambuf_iterator<char>(fi)), {});
    if (bs.empty()) throw std::runtime_error("empty stream");

    // ---- split Annex-B NAL units --------------------------------
    Stream st;
    // (startCodeAt, payloadAt, endAt)
    std::vector<std::array<size_t, 3>> nals;
    for (size_t i = 0; i + 3 < bs.size(); ++i) {
        size_t sc = 0;
        if (!bs[i] && !bs[i + 1] && bs[i + 2] == 1) sc = 3;
        else if (i + 4 < bs.size() && !bs[i] && !bs[i + 1] &&
                 !bs[i + 2] && bs[i + 3] == 1) sc = 4;
        if (!sc) continue;
        size_t body = i + sc;
        if (nals.size()) nals.back()[2] = i;
        nals.push_back({i, body, bs.size()});
    }
    for (auto [sc0, b, e] : nals) {
        if (e <= b + 1) continue;   // empty NAL (start code with no body)
        int type = bs[b] & 0x1F;
        auto rb = unescape(bs.data() + b + 1, e - b - 1);
        if (type == 7) parseSps(rb, st);
        else if (type == 8) parsePps(rb, st);
        else if (type == 5 || type == 1) {
            st.idr |= (type == 5);
            st.sliceOfs.push_back(st.slices.size());
            st.slices.insert(st.slices.end(), bs.begin() + sc0,
                             bs.begin() + e);   // include start code
        }
    }
    uint32_t W = (st.sps.pic_width_in_mbs_minus1 + 1) * 16;
    uint32_t H = (st.sps.pic_height_in_map_units_minus1 + 1) * 16;
    printf("stream: %ux%u, %zu NALs, idr=%d, %zu slice(s)\n", W, H,
           nals.size(), st.idr, st.sliceOfs.size());

    // ---- device on the VIDEO DECODE queue family ------------------
    // Video decode is NOT guaranteed on every GPU: scan all physical
    // devices for one exposing a queue family with eVideoDecodeKHR.
    // If none exists the program skips gracefully — that IS the
    // capability-detection contract.
    vkmini::Vk vk;
    vk.createInstance({});
    uint32_t decFam = UINT32_MAX, gfxFam = UINT32_MAX;
    for (auto& d : vk.instance->enumeratePhysicalDevices()) {
        auto qfs = d.getQueueFamilyProperties();
        for (uint32_t i = 0; i < qfs.size(); ++i)
            if (qfs[i].queueFlags &
                vk::QueueFlagBits::eVideoDecodeKHR) {
                vk.phys = d;
                decFam = i;
                break;
            }
        if (decFam != UINT32_MAX) break;
    }
    if (!vk.phys) {
        printf("SKIP: no device with a video-decode queue family\n");
        return 0;
    }
    printf("device: %s\n", vk.phys.getProperties().deviceName.data());
    auto qfams = vk.phys.getQueueFamilyProperties();
    for (uint32_t i = 0; i < qfams.size(); ++i)
        if (qfams[i].queueFlags & vk::QueueFlagBits::eGraphics &&
            gfxFam == UINT32_MAX)
            gfxFam = i;
    printf("queue families: decode=%u gfx=%u\n", decFam, gfxFam);

    float prio = 1.f;
    std::set<uint32_t> fams{decFam, gfxFam};
    std::vector<vk::DeviceQueueCreateInfo> qis;
    for (auto f : fams) {
        vk::DeviceQueueCreateInfo qi{};
        qi.setQueueFamilyIndex(f).setQueueCount(1)
            .setPQueuePriorities(&prio);
        qis.push_back(qi);
    }
    vk::DeviceCreateInfo dci{};
    std::array exts{VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
                    VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
                    VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME};
    dci.setQueueCreateInfos(qis).setPEnabledExtensionNames(exts);
    vk.device = vk.phys.createDeviceUnique(dci);
    vk.qfam = gfxFam;
    vk.queue = vk.device->getQueue(gfxFam, 0);
    vk.pool = vk.device->createCommandPoolUnique(
        {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, gfxFam});
    auto decQueue = vk.device->getQueue(decFam, 0);
    auto decPool = vk.device->createCommandPoolUnique(
        {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, decFam});
    vk::detail::DispatchLoaderDynamic dldi(
        vk.instance.get(), vkGetInstanceProcAddr, vk.device.get());

    // ---- video profile + caps -------------------------------------
    vk::VideoDecodeH264ProfileInfoKHR h264prof{};
    h264prof.setStdProfileIdc(STD_VIDEO_H264_PROFILE_IDC_BASELINE)
        .setPictureLayout(
            vk::VideoDecodeH264PictureLayoutFlagBitsKHR::
                eProgressive);
    vk::VideoProfileInfoKHR profile{};
    profile.setVideoCodecOperation(
               vk::VideoCodecOperationFlagBitsKHR::eDecodeH264)
        .setPNext(&h264prof)
        .setChromaSubsampling(
            vk::VideoChromaSubsamplingFlagBitsKHR::e420)
        .setLumaBitDepth(vk::VideoComponentBitDepthFlagBitsKHR::e8)
        .setChromaBitDepth(vk::VideoComponentBitDepthFlagBitsKHR::e8);
    // spec: for decode ops the CAPS chain must carry BOTH
    // VkVideoDecodeCapabilitiesKHR and the codec-specific
    // VkVideoDecodeH264CapabilitiesKHR — drivers write both
    // unconditionally (missing link = NULL write crash, not a
    // clean error!).
    vk::VideoDecodeH264CapabilitiesKHR h264caps{};
    vk::VideoDecodeCapabilitiesKHR decCaps{};
    decCaps.pNext = &h264caps;
    vk::VideoCapabilitiesKHR caps{};
    caps.pNext = &decCaps;
    vk.phys.getVideoCapabilitiesKHR(&profile, &caps, dldi);
    printf("caps: maxDpbSlots %u maxLevel %u stdHeader %s v%u\n",
           caps.maxDpbSlots, (uint32_t)h264caps.maxLevelIdc,
           caps.stdHeaderVersion.extensionName.data(),
           caps.stdHeaderVersion.specVersion);

    // supported picture formats for DST and DPB usage
    auto pickFmt = [&](vk::ImageUsageFlags usage) {
        vk::PhysicalDeviceVideoFormatInfoKHR finfo{};
        finfo.setPNext(&profile).setImageUsage(usage);
        auto vfmt = vk.phys.getVideoFormatPropertiesKHR(finfo, dldi);
        return vfmt.empty() ? vk::Format::eUndefined
                            : vfmt[0].format;
    };
    vk::Format dstFmt = pickFmt(vk::ImageUsageFlagBits::eVideoDecodeDstKHR);
    vk::Format dpbFmt = pickFmt(vk::ImageUsageFlagBits::eVideoDecodeDpbKHR);
    printf("formats: dst=%s dpb=%s\n",
           vk::to_string(dstFmt).c_str(), vk::to_string(dpbFmt).c_str());

    // ---- session + parameters -------------------------------------
    vk::VideoSessionCreateInfoKHR sci{};
    sci.setQueueFamilyIndex(decFam)
        .setPVideoProfile(&profile)
        .setPictureFormat(dstFmt)
        .setMaxCodedExtent({W, H})
        .setReferencePictureFormat(dpbFmt)
        .setMaxDpbSlots(1)
        .setMaxActiveReferencePictures(1)
        .setPStdHeaderVersion(&caps.stdHeaderVersion);
    auto session =
        vk.device->createVideoSessionKHR(sci, nullptr, dldi);

    auto reqs = vk.device->getVideoSessionMemoryRequirementsKHR(
        session, dldi);
    std::vector<vk::BindVideoSessionMemoryInfoKHR> binds;
    std::vector<vk::UniqueDeviceMemory> sessMem;
    for (auto& r : reqs) {
        vk::MemoryAllocateInfo mai{};
        mai.setAllocationSize(r.memoryRequirements.size)
            .setMemoryTypeIndex(
                vk.memoryType(r.memoryRequirements.memoryTypeBits,
                              {}));
        sessMem.push_back(vk.device->allocateMemoryUnique(mai));
        vk::BindVideoSessionMemoryInfoKHR bi{};
        bi.setMemoryBindIndex(r.memoryBindIndex)
            .setMemory(sessMem.back().get())
            .setMemoryOffset(0)
            .setMemorySize(r.memoryRequirements.size);
        binds.push_back(bi);
    }
    vk.device->bindVideoSessionMemoryKHR(session, binds, dldi);

    StdVideoH264SequenceParameterSet spsArr[1]{st.sps};
    StdVideoH264PictureParameterSet ppsArr[1]{st.pps};
    vk::VideoDecodeH264SessionParametersAddInfoKHR add{};
    add.setStdSPSs(spsArr).setStdPPSs(ppsArr);
    vk::VideoDecodeH264SessionParametersCreateInfoKHR spci{};
    spci.setMaxStdSPSCount(1).setMaxStdPPSCount(1)
        .setPParametersAddInfo(&add);
    vk::VideoSessionParametersCreateInfoKHR pci{};
    pci.setVideoSession(session).setPNext(&spci);
    auto params = vk.device->createVideoSessionParametersKHR(
        pci, nullptr, dldi);

    // ---- bitstream buffer -----------------------------------------
    auto srcBuf = vk.createBuffer(
        st.slices.size(), vk::BufferUsageFlagBits::eVideoDecodeSrcKHR,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    memcpy(srcBuf.mapped, st.slices.data(), st.slices.size());

    // ---- DPB + dst images (usage-tagged, profile-chained) ----------
    auto mkVideoImage = [&](vk::Format f, vk::ImageUsageFlags u) {
        vk::VideoProfileListInfoKHR pl{};
        pl.setProfiles(profile);
        vk::ImageCreateInfo ii{};
        ii.setImageType(vk::ImageType::e2D)
            .setFormat(f).setExtent({W, H, 1}).setMipLevels(1)
            .setArrayLayers(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setTiling(vk::ImageTiling::eOptimal)
            .setUsage(u)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setPNext(&pl);
        vkmini::Vk::Image o;
        o.img = vk.device->createImageUnique(ii);
        auto req = vk.device->getImageMemoryRequirements(o.img.get());
        o.mem = vk.device->allocateMemoryUnique(
            {req.size,
             vk.memoryType(req.memoryTypeBits,
                           vk::MemoryPropertyFlagBits::eDeviceLocal)});
        vk.device->bindImageMemory(o.img.get(), o.mem.get(), 0);
        vk::ImageViewCreateInfo vi{};
        vi.setImage(o.img.get())
            .setViewType(vk::ImageViewType::e2D).setFormat(f)
            .setSubresourceRange(
                {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        o.view = vk.device->createImageViewUnique(vi);
        return o;
    };
    auto dpbImg = mkVideoImage(dpbFmt,
        vk::ImageUsageFlagBits::eVideoDecodeDpbKHR);
    auto dstImg = mkVideoImage(dstFmt,
        vk::ImageUsageFlagBits::eVideoDecodeDstKHR |
        vk::ImageUsageFlagBits::eTransferSrc);

    // ---- decode ----------------------------------------------------
    auto cmds = vk.device->allocateCommandBuffersUnique(
        {decPool.get(), vk::CommandBufferLevel::ePrimary, 1});
    auto& c = cmds[0];
    c->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // video access/stage flags exist ONLY in the sync2 enums
    vk::ImageMemoryBarrier2 toVid[2];
    toVid[0]
        .setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe)
        .setSrcAccessMask(vk::AccessFlagBits2::eNone)
        .setDstStageMask(
            vk::PipelineStageFlagBits2::eVideoDecodeKHR)
        .setDstAccessMask(
            vk::AccessFlagBits2::eVideoDecodeWriteKHR)
        .setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eVideoDecodeDstKHR)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(dstImg.img.get())
        .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1,
                              0, 1});
    toVid[1] = toVid[0];
    toVid[1].setImage(dpbImg.img.get())
        .setNewLayout(vk::ImageLayout::eVideoDecodeDpbKHR);
    vk::DependencyInfo dep{};
    dep.setImageMemoryBarriers(toVid);
    c->pipelineBarrier2(dep);

    vk::VideoBeginCodingInfoKHR begin{};
    begin.setVideoSession(session)
        .setVideoSessionParameters(params);
    c->beginVideoCodingKHR(begin, dldi);

    // decoded frame becomes DPB slot 0 (reference for later frames)
    StdVideoDecodeH264ReferenceInfo refInfo{};
    refInfo.FrameNum = 0;
    refInfo.PicOrderCnt[0] = 0;
    refInfo.PicOrderCnt[1] = 0;
    vk::VideoDecodeH264DpbSlotInfoKHR dpbH264{};
    dpbH264.setPStdReferenceInfo(&refInfo);
    vk::VideoPictureResourceInfoKHR dpbRes{};
    dpbRes.setCodedOffset({0, 0}).setCodedExtent({W, H})
        .setImageViewBinding(dpbImg.view.get());
    vk::VideoReferenceSlotInfoKHR setupSlot{};
    setupSlot.setSlotIndex(0).setPPictureResource(&dpbRes)
        .setPNext(&dpbH264);

    StdVideoDecodeH264PictureInfo pic{};
    pic.flags.IdrPicFlag = st.idr;
    pic.flags.is_intra = 1;
    pic.flags.is_reference = 1;
    pic.seq_parameter_set_id = st.pps.seq_parameter_set_id;
    pic.pic_parameter_set_id = st.pps.pic_parameter_set_id;
    pic.frame_num = 0;
    pic.idr_pic_id = 0;
    pic.PicOrderCnt[0] = 0;
    pic.PicOrderCnt[1] = 0;
    vk::VideoDecodeH264PictureInfoKHR picInfo{};
    picInfo.setPStdPictureInfo(&pic)
        .setSliceCount(st.sliceOfs.size())
        .setPSliceOffsets(st.sliceOfs.data());
    vk::VideoPictureResourceInfoKHR dstRes{};
    dstRes.setCodedOffset({0, 0}).setCodedExtent({W, H})
        .setImageViewBinding(dstImg.view.get());
    vk::VideoDecodeInfoKHR dec{};
    dec.setSrcBuffer(srcBuf.buf.get())
        .setSrcBufferOffset(0)
        .setSrcBufferRange(st.slices.size())
        .setDstPictureResource(dstRes)
        .setPSetupReferenceSlot(&setupSlot)
        .setPNext(&picInfo);
    c->decodeVideoKHR(dec, dldi);
    c->endVideoCodingKHR({}, dldi);

    // ownership transfer decode->graphics for the readback
    if (decFam != gfxFam) {
        vk::ImageMemoryBarrier2 rel{};
        rel.setSrcStageMask(
               vk::PipelineStageFlagBits2::eVideoDecodeKHR)
            .setSrcAccessMask(
                vk::AccessFlagBits2::eVideoDecodeWriteKHR)
            .setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
            .setDstAccessMask(vk::AccessFlagBits2::eTransferRead)
            .setOldLayout(vk::ImageLayout::eVideoDecodeDstKHR)
            .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
            .setSrcQueueFamilyIndex(decFam)
            .setDstQueueFamilyIndex(gfxFam)
            .setImage(dstImg.img.get())
            .setSubresourceRange({vk::ImageAspectFlagBits::eColor,
                                  0, 1, 0, 1});
        vk::DependencyInfo dep2{};
        dep2.setImageMemoryBarriers(rel);
        c->pipelineBarrier2(dep2);
    }
    c->end();
    auto fence = vk.device->createFenceUnique({});
    vk::SubmitInfo si{};
    si.setCommandBuffers(c.get());
    decQueue.submit(si, fence.get());
    vk.device->waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
    printf("decoded\n");

    // ---- readback luma+chroma planes ------------------------------
    auto rb = vk.createBuffer(
        W * H * 3 / 2, vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        true);
    vk.oneTime([&](vk::CommandBuffer c2) {
        if (decFam != gfxFam) {
            vk::ImageMemoryBarrier2 acq{};
            acq.setSrcStageMask(
                   vk::PipelineStageFlagBits2::eVideoDecodeKHR)
                .setSrcAccessMask(
                    vk::AccessFlagBits2::eVideoDecodeWriteKHR)
                .setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
                .setDstAccessMask(vk::AccessFlagBits2::eTransferRead)
                .setOldLayout(vk::ImageLayout::eVideoDecodeDstKHR)
                .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
                .setSrcQueueFamilyIndex(decFam)
                .setDstQueueFamilyIndex(gfxFam)
                .setImage(dstImg.img.get())
                .setSubresourceRange(
                    {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
            vk::DependencyInfo dep3{};
            dep3.setImageMemoryBarriers(acq);
            c2.pipelineBarrier2(dep3);
        }
        vk::BufferImageCopy cy{};
        cy.setImageSubresource({vk::ImageAspectFlagBits::ePlane0, 0,
                                0, 1})
            .setImageExtent({W, H, 1});
        vk::BufferImageCopy cc{};
        cc.setImageSubresource({vk::ImageAspectFlagBits::ePlane1, 0,
                                0, 1})
            .setImageExtent({W / 2, H / 2, 1})
            .setBufferOffset(W * H);
        c2.copyImageToBuffer(dstImg.img.get(),
                             vk::ImageLayout::eTransferSrcOptimal,
                             rb.buf.get(), {cy, cc});
    });

    // NV12 -> RGB PPM
    const uint8_t* Y = (const uint8_t*)rb.mapped;
    const uint8_t* UV = Y + W * H;
    FILE* f = fopen(ppmOut, "wb");
    fprintf(f, "P6\n%u %u\n255\n", W, H);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            int yy = Y[y * W + x], u = UV[(y / 2) * W + (x / 2) * 2],
                v = UV[(y / 2) * W + (x / 2) * 2 + 1];
            int r = yy + 1.402 * (v - 128);
            int g = yy - 0.344 * (u - 128) - 0.714 * (v - 128);
            int b2 = yy + 1.772 * (u - 128);
            uint8_t out[3] = {(uint8_t)std::clamp(r, 0, 255),
                              (uint8_t)std::clamp(g, 0, 255),
                              (uint8_t)std::clamp(b2, 0, 255)};
            fwrite(out, 1, 3, f);
        }
    fclose(f);
    printf("wrote %s\n", ppmOut);
    return 0;
}
