// SPDX-License-Identifier: CC0-1.0
//
// Progressive (and baseline) JPEG DECODING in Vulkan compute shaders,
// extending p80_gpujpegdec to multi-scan streams. Four stages:
//
//   1. entropy:  one invocation per restart interval per scan decodes
//      Huffman-coded MCUs -> quantized DCT coefficients (zigzag order,
//      i16 packed pairs). Scans dispatch one after another into a
//      persistent coef buffer — progressive refinement scans read and
//      update coefficients written by earlier scans. Restart markers
//      (DRI>0) make each scan's bitstream parallel; EOBRUN and DC
//      predictors reset at markers, so intervals decode independently.
//   2. recon:    one invocation per 8x8 block: dequantize + libjpeg
//      "islow" integer IDCT -> per-component u8 sample planes.
//   3. upsample: one invocation per output pixel: libjpeg "fancy"
//      triangle upsampling + libjpeg fixed-point YCbCr->RGB -> RGBA8.
//
// Progressive scan modes (jdphuff.c mirrored exactly):
//   sequential   Ss=0..63 Ah=0  — baseline-style DC+AC per block
//   DC first     Ss=0    Ah=0  — DC coef = pred << Al
//   DC refine    Ss=0    Ah>0  — one correction bit per block
//   AC first     Ss>0    Ah=0  — band coefs = v << Al, EOBRUN runs
//   AC refine    Ss>0    Ah>0  — correction bits + new +-p1 coefs
//
// Interleaved scans (ns>1, DC or sequential only) iterate the MCU
// grid; non-interleaved scans (ns=1) iterate the component's own
// width_in_blocks x height_in_blocks raster — which can differ from
// the MCU-aligned block grid (jdinput.c: ceil(W*hs/(maxH*8))).
//
//   ./app in.jpg out.ppm           decode entirely on the GPU
//   ./app --cpu in.jpg out.ppm     libjpeg reference (for comparison)
//   ./app --verify in.jpg          GPU decode + per-stage diff report
//   ./app --bench N in.jpg         N decodes, prints MP/s
//
// Build:  g++ -std=c++23 -O2 main.cpp -o app -lvulkan -lshaderc -ljpeg
// (no GLFW needed — compute only. PIL/libjpeg used by check.py.)

#include "vkmini.hpp"
#include <jpeglib.h>

#include <chrono>
#include <cmath>
#include <map>

// Zigzag position -> natural (row-major) index inside an 8x8 block.
static constexpr uint8_t kZig[64] = {
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63};

// ============================================================
//  JFIF parser — collects every scan of a progressive stream.
// ============================================================

struct HuffTable {
    uint8_t counts[16]{};        // BITS: #codes of each length 1..16
    std::vector<uint8_t> vals;   // HUFFVAL symbols in file order
    // canonical decode tables (host-built)
    int32_t mincode[16]{}, maxcode[16]{}, valptr[16]{};
    uint32_t lut8[256]{};        // (len<<8)|sym, 0 = code longer than 8
};

struct Comp {
    int id = 0, hs = 1, vs = 1, tq = 0;
    int bw = 0, bh = 0;          // MCU-aligned block-grid dims
    int wib = 0, hib = 0;        // width/height_in_blocks (non-interleaved)
    int w = 0, h = 0;            // downsampled sample dims
};

struct Interval { uint32_t byteStart, mcuStart, nMcu; };

struct ScanComp { int comp = 0, td = 0, ta = 0; };

struct Scan {
    int ns = 0;                  // scan component count
    ScanComp sc[4];
    int ss = 0, se = 63, ah = 0, al = 0;
    int dri = 0;                 // restart interval in effect for this scan
    uint32_t dataOff = 0, dataLen = 0;
    std::vector<Interval> intervals;
    uint32_t totalMcus = 0;      // MCU count in this scan's units
    // Tables snapshot AS OF this scan — progressive files commonly
    // redefine DHT/DQT between scans (per-scan optimized tables).
    HuffTable dctab[4], actab[4];
    uint16_t qt[4][64]{};
    bool tablesChanged = false;  // differs from previous scan's set
    // DRI=0 scans get "virtual restart intervals" from the survey
    // pass: 12 st[] words (bitreader + pred + eobrun) per interval,
    // preloaded by the host instead of chained from a predecessor.
    std::vector<uint32_t> survSt;  // intervals.size()*12 when nonempty
};

struct Jpg {
    std::vector<uint8_t> data;
    int W = 0, H = 0;
    Comp comps[4];
    int nComp = 0;
    int maxH = 1, maxV = 1;
    int mcusX = 0, mcusY = 0;
    int dri = 0;                 // current DRI while scanning headers
    uint16_t qt[4][64]{};        // natural order
    HuffTable dctab[4], actab[4];
    bool haveDht = false;
    bool tablesDirty = false;    // DHT/DQT seen since the last SOS —
                                 // next scan needs a fresh upload
    std::vector<Scan> scans;
    int totalMcus = 0;
    bool progressive = false;
    int orient = 1;              // EXIF orientation tag (APP1), 1..8
};

// APP1 "Exif\0\0" TIFF -> orientation tag 0x0112 (default 1).
static int parseExifOrient(const uint8_t* s, uint32_t n) {
    if (n < 14 || memcmp(s, "Exif\0\0", 6) != 0) return 1;
    const uint8_t* t = s + 6;
    bool le = t[0] == 'I';
    if (!le && !(t[0] == 'M' && t[1] == 'M')) return 1;
    auto u16 = [&](uint32_t o) {
        return le ? t[o] | (t[o + 1] << 8) : (t[o] << 8) | t[o + 1];
    };
    auto u32 = [&](uint32_t o) {
        return le ? t[o] | (t[o + 1] << 8) | (t[o + 2] << 16) |
                        (uint32_t(t[o + 3]) << 24)
                  : (t[o] << 24) | (t[o + 1] << 16) | (t[o + 2] << 8) |
                        t[o + 3];
    };
    if (u16(2) != 42) return 1;
    uint32_t ifd = u32(4);
    if (ifd + 2 > uint32_t(n) - 6) return 1;
    uint32_t cnt = u16(ifd);
    for (uint32_t i = 0; i < cnt; ++i) {
        uint32_t e = ifd + 2 + i * 12;
        if (e + 12 > uint32_t(n) - 6) break;
        if (u16(e) == 0x0112 && u16(e + 2) == 3) {
            int v = u16(e + 8);
            return v >= 1 && v <= 8 ? v : 1;
        }
    }
    return 1;
}

static void buildCanon(HuffTable& t) {
    int code = 0, k = 0;
    for (int l = 0; l < 16; ++l) {
        if (t.counts[l]) {
            t.mincode[l] = code;
            t.valptr[l]  = k - code;          // index = valptr + code
            t.maxcode[l] = code + t.counts[l] - 1;
        } else {
            t.mincode[l] = 0; t.valptr[l] = k; t.maxcode[l] = -1;
        }
        code = (code + t.counts[l]) << 1;
        k += t.counts[l];
    }
    // 8-bit prefix LUT: every len<=8 code covers 2^(8-len) suffixes.
    code = 0; k = 0;
    for (int l = 1; l <= 8; ++l) {
        for (int i = 0; i < t.counts[l - 1]; ++i) {
            int base = code << (8 - l), span = 1 << (8 - l);
            uint32_t e = (uint32_t(l) << 8) | t.vals[k + i];
            for (int s = 0; s < span; ++s) t.lut8[base + s] = e;
            code++;
        }
        code <<= 1;
        k += t.counts[l - 1];
    }
}

// Find restart-marker boundaries inside a scan's entropy segment and
// build the interval table (byteStart, mcuStart, nMcu) in scan units.

static bool parseJpg(Jpg& j) {
    const auto& d = j.data;
    if (d.size() < 4 || d[0] != 0xFF || d[1] != 0xD8) {
        fprintf(stderr, "not a JPEG (no SOI)\n");
        return false;
    }
    size_t p = 2;
    bool haveSof = false;
    while (p + 1 < d.size()) {
        // find next marker (FF followed by non-00)
        if (d[p] != 0xFF) { p++; continue; }
        uint8_t m = d[p + 1];
        if (m == 0x00) { p += 2; continue; }   // stuffed data byte
        if (m >= 0xD0 && m <= 0xD7) { p += 2; continue; } // stray RST
        if (m == 0xD8 || m == 0x01) { p += 2; continue; } // SOI/TEM
        if (m == 0xD9) break;                              // EOI
        if (p + 3 >= d.size()) break;
        uint32_t len = (d[p + 2] << 8) | d[p + 3];
        const uint8_t* s = d.data() + p + 4;
        uint32_t slen = len - 2;
        if (p + 2 + len > d.size()) return false;          // truncated

        switch (m) {
        case 0xDB: { // DQT
            if (!j.scans.empty()) j.tablesDirty = true;
            uint32_t o = 0;
            while (o < slen) {
                int pq = s[o] >> 4, tq = s[o] & 15; o++;
                if (tq > 3 || pq > 1 || o + (pq ? 128u : 64u) > slen) return false;
                for (int i = 0; i < 64; ++i) {
                    uint16_t v = pq ? (s[o] << 8) | s[o + 1] : s[o];
                    o += pq ? 2 : 1;
                    j.qt[tq][kZig[i]] = v;   // -> natural order
                }
            }
            break;
        }
        case 0xC4: { // DHT (possibly several tables per segment)
            if (!j.scans.empty()) j.tablesDirty = true;
            uint32_t o = 0;
            while (o < slen) {
                if (o + 17 > slen) return false;
                int tc = s[o] >> 4, th = s[o] & 15; o++;
                if (th > 3 || tc > 1) return false;
                HuffTable& t = tc ? j.actab[th] : j.dctab[th];
                int n = 0;
                for (int i = 0; i < 16; ++i) { t.counts[i] = s[o + i]; n += t.counts[i]; }
                o += 16;
                if (o + uint32_t(n) > slen || n > 256) return false;
                t.vals.assign(s + o, s + o + n);
                o += n;
                j.haveDht = true;
            }
            break;
        }
        case 0xE1: // APP1: EXIF orientation
            j.orient = parseExifOrient(s, slen);
            break;
        case 0xC0: case 0xC1: case 0xC2: { // SOF0/1 sequential, SOF2 progressive
            if (s[0] != 8) {
                fprintf(stderr, "unsupported precision %d\n", s[0]);
                return false;
            }
            j.progressive = (m == 0xC2);
            j.H = (s[1] << 8) | s[2];
            j.W = (s[3] << 8) | s[4];
            j.nComp = s[5];
            if (j.nComp != 1 && j.nComp != 3) {
                fprintf(stderr, "unsupported %d components\n", j.nComp);
                return false;
            }
            for (int i = 0; i < j.nComp; ++i) {
                Comp& c = j.comps[i];
                c.id = s[6 + i * 3];
                c.hs = s[7 + i * 3] >> 4;
                c.vs = s[7 + i * 3] & 15;
                c.tq = s[8 + i * 3];
                j.maxH = std::max(j.maxH, c.hs);
                j.maxV = std::max(j.maxV, c.vs);
            }
            haveSof = true;
            break;
        }
        case 0xDD: // DRI — takes effect for all scans that follow
            j.dri = (s[0] << 8) | s[1];
            break;
        case 0xDA: { // SOS — entropy data follows the header
            Scan sc;
            sc.ns = s[0];
            if (sc.ns < 1 || sc.ns > 4 || sc.ns > j.nComp) {
                fprintf(stderr, "unsupported ns=%d\n", sc.ns);
                return false;
            }
            for (int i = 0; i < sc.ns; ++i) {
                int cs = s[1 + i * 2];
                Comp* c = nullptr;
                int ci = -1;
                for (int k = 0; k < j.nComp; ++k)
                    if (j.comps[k].id == cs) { c = &j.comps[k]; ci = k; }
                if (!c) return false;
                sc.sc[i].comp = ci;
                sc.sc[i].td = s[2 + i * 2] >> 4;
                sc.sc[i].ta = s[2 + i * 2] & 15;
                if (sc.sc[i].td > 3 || sc.sc[i].ta > 3) return false;
            }
            sc.ss = s[1 + sc.ns * 2];
            sc.se = s[2 + sc.ns * 2];
            sc.ah = s[3 + sc.ns * 2] >> 4;
            sc.al = s[3 + sc.ns * 2] & 15;
            sc.dri = j.dri;
            // snapshot tables as they stand at this scan
            for (int t = 0; t < 4; ++t) {
                sc.dctab[t] = j.dctab[t];
                sc.actab[t] = j.actab[t];
                memcpy(sc.qt[t], j.qt[t], sizeof(sc.qt[t]));
            }
            sc.tablesChanged = j.tablesDirty;
            j.tablesDirty = false;
            sc.dataOff = uint32_t(p + 4 + slen);
            // entropy data ends before the next unstuffed marker
            size_t q = sc.dataOff;
            while (q + 1 < d.size()) {
                if (d[q] == 0xFF && d[q + 1] != 0x00 &&
                    !(d[q + 1] >= 0xD0 && d[q + 1] <= 0xD7))
                    break;
                q++;
            }
            sc.dataLen = uint32_t(q - sc.dataOff);
            j.scans.push_back(sc);
            p = q;
            continue;
        }
        default: break;
        }
        p += 2 + len;
    }

    if (!haveSof || !j.haveDht || j.scans.empty()) {
        fprintf(stderr, "missing SOF/DHT/SOS\n");
        return false;
    }

    // ---- geometry ----
    j.mcusX = (j.W + 8 * j.maxH - 1) / (8 * j.maxH);
    j.mcusY = (j.H + 8 * j.maxV - 1) / (8 * j.maxV);
    j.totalMcus = j.mcusX * j.mcusY;
    for (int i = 0; i < j.nComp; ++i) {
        Comp& c = j.comps[i];
        c.bw = j.mcusX * c.hs;
        c.bh = j.mcusY * c.vs;
        c.w = (j.W * c.hs + j.maxH - 1) / j.maxH;
        c.h = (j.H * c.vs + j.maxV - 1) / j.maxV;
        // jdinput.c: width_in_blocks = ceil(W*hs/(maxH*8)) — this can
        // differ from bw (MCU-aligned); non-interleaved scans iterate
        // this grid.
        c.wib = (j.W * c.hs + j.maxH * 8 - 1) / (j.maxH * 8);
        c.hib = (j.H * c.vs + j.maxV * 8 - 1) / (j.maxV * 8);
    }

    // ---- per-scan restart intervals ----
    for (auto& sc : j.scans) {
        uint32_t total = sc.ns == 1
            ? uint32_t(j.comps[sc.sc[0].comp].wib) *
              uint32_t(j.comps[sc.sc[0].comp].hib)
            : uint32_t(j.totalMcus);
        sc.totalMcus = total;
        sc.intervals.push_back({0, 0, 0});
        const uint8_t* sd = d.data() + sc.dataOff;
        int found = 0;
        for (uint32_t i = 0; i + 1 < sc.dataLen; ++i) {
            if (sd[i] == 0xFF && sd[i + 1] >= 0xD0 && sd[i + 1] <= 0xD7) {
                found++;
                sc.intervals.back().nMcu =
                    uint32_t(found * std::max(sc.dri, 1)) -
                    sc.intervals.back().mcuStart;
                sc.intervals.push_back(
                    {i + 2, uint32_t(found) * std::max(sc.dri, 1), 0});
                i++;
            }
        }
        if (found == 0) {
            sc.intervals.resize(1);
            sc.intervals[0] = {0, 0, total};
        } else {
            sc.intervals.back().nMcu =
                total - sc.intervals.back().mcuStart;
        }
    }
    return true;
}

// ============================================================
//  GPU buffer layouts (std430, plain int32 words throughout)
// ============================================================

// Tables buffer words:
//   [0x000..0x0FF]  qtab: 4 x 64 natural-order quant values
//   per table t (8 total: 0-3 DC, 4-7 AC), base = 256 + t*816:
//     +0    mincode[16]   +16  maxcode[16]   +32  valptr[16]
//     +48   lut8[256]     +304 huffval[256]
static constexpr int kTabWords = 256 + 8 * 816;
static constexpr int kQBase = 0, kHuffBase = 256;

// Flat 16-bit "fused" Huffman LUT — one lookup yields the symbol AND,
// whenever huffLen+extLen fits in the 16-bit window, the already
// sign-extended coefficient value. Entry (u32):
//   bit31 set   -> fused: [4:0]=totalLen [8:5]=r [12:9]=s [28:13]=val+32768
//   bit31 clear -> plain: [4:0]=huffLen [15:8]=sym   (huffLen==0 -> invalid)
static constexpr int kFLutWords = 8 * 65536;

static int extendRef(int v, int s) {
    return (s > 0 && v < (1 << (s - 1))) ? v - (1 << s) + 1 : v;
}

static bool sameTable(const HuffTable& a, const HuffTable& b) {
    return !memcmp(a.counts, b.counts, sizeof(a.counts)) && a.vals == b.vals;
}

static void packFLut(const Scan& sc, std::vector<uint32_t>& fl,
                     const Scan* prev = nullptr) {
    if (!prev) fl.assign(kFLutWords, 0);
    for (int t = 0; t < 8; ++t) {
        const HuffTable& h = (t < 4) ? sc.dctab[t] : sc.actab[t - 4];
        if (prev) {   // skip rebuild when this table is unchanged
            const HuffTable& p =
                (t < 4) ? prev->dctab[t] : prev->actab[t - 4];
            if (sameTable(h, p)) continue;
            memset(fl.data() + t * 65536, 0, 65536 * 4);
        }
        for (int x = 0; x < 65536; ++x) {
            // canonical walk for the first code in this prefix
            int len = 0, sym = 0;
            for (int l = 1; l <= 16; ++l) {
                int code = x >> (16 - l);
                if (h.maxcode[l - 1] >= 0 && code <= h.maxcode[l - 1]) {
                    len = l;
                    sym = h.vals[h.valptr[l - 1] + code];
                    break;
                }
            }
            if (!len) continue;
            int r = (t < 4) ? 0 : (sym >> 4), s = sym & 15;
            if (t < 4) s = sym;                    // DC: symbol is size
            if (len + s <= 16) {
                int ext = (x >> (16 - len - s)) & ((1 << s) - 1);
                int val = extendRef(ext, s);
                fl[t * 65536 + x] = 0x80000000u | uint32_t(len + s) |
                                    uint32_t(r << 5) | uint32_t(s << 9) |
                                    uint32_t((val + 32768) << 13);
            } else {
                fl[t * 65536 + x] = uint32_t(len) | uint32_t(sym << 8);
            }
        }
    }
}

static void packTables(const Scan& sc, std::vector<int32_t>& tab) {
    tab.assign(kTabWords, 0);
    for (int t = 0; t < 4; ++t)
        for (int i = 0; i < 64; ++i)
            tab[kQBase + t * 64 + i] = sc.qt[t][i];
    for (int t = 0; t < 8; ++t) {
        const HuffTable& h = (t < 4) ? sc.dctab[t] : sc.actab[t - 4];
        int b = kHuffBase + t * 816;
        for (int i = 0; i < 16; ++i) {
            tab[b + i] = h.mincode[i];
            tab[b + 16 + i] = h.maxcode[i];
            tab[b + 32 + i] = h.valptr[i];
        }
        for (int i = 0; i < 256; ++i) {
            tab[b + 48 + i] = int32_t(h.lut8[i]);
            tab[b + 304 + i] = i < (int)h.vals.size() ? h.vals[i] : 0;
        }
    }
}

// Per-component params, 16 int32 words each (binding 4, entropy, recon
// and upsample read it):
//   0 hs  1 vs  2 bw  3 bh  4 downW  5 downH  6 qt  7 pad  8 pad
//   9 coefOfs(words)  10 planeOfs(bytes)  11 wib  12 hib  13 ss  14,15 pad
struct FrameGpu {
    int32_t comps[4][16];
    int32_t totalBlocks, planeWords, nIntervals;
};

static FrameGpu packComps(const Jpg& j, int smin = 8) {
    FrameGpu f{};
    uint32_t coefOf = 0, planeOf = 0;
    for (int i = 0; i < 4; ++i) {
        const Comp& c = j.comps[i];
        // jdmaster: double the per-comp IDCT scale while the sampling
        // grid still divides evenly — chroma gets a bigger IDCT instead
        // of upsampling (4:2:0 chroma -> ss 8 at scale 1/2, -> ss 2 at 1/8)
        int ss = smin;
        while (ss < 8 &&
               (j.maxH * smin) % (c.hs * ss * 2) == 0 &&
               (j.maxV * smin) % (c.vs * ss * 2) == 0)
            ss *= 2;
        int downW = int((int64_t(j.W) * c.hs * ss + j.maxH * 8 - 1) /
                        (j.maxH * 8));
        int downH = int((int64_t(j.H) * c.vs * ss + j.maxV * 8 - 1) /
                        (j.maxV * 8));
        f.comps[i][0] = c.hs;  f.comps[i][1] = c.vs;
        f.comps[i][2] = c.bw;  f.comps[i][3] = c.bh;
        f.comps[i][4] = downW; f.comps[i][5] = downH;
        f.comps[i][6] = c.tq;
        f.comps[i][7] = 0;
        f.comps[i][8] = 0;
        f.comps[i][9] = int32_t(coefOf);
        f.comps[i][10] = int32_t(planeOf);
        f.comps[i][11] = c.wib;
        f.comps[i][12] = c.hib;
        f.comps[i][13] = i < j.nComp ? ss : 8;
        if (i < j.nComp) {
            coefOf += uint32_t(c.bw) * c.bh * 32;      // 64 coefs / 2 per word
            // plane: ss<=2 word-per-sample, else byte-per-sample
            int64_t px = int64_t(c.bw * ss) * (c.bh * ss);
            planeOf += uint32_t(ss <= 2 ? px * 4 : px);
        }
        f.totalBlocks += c.bw * c.bh;
    }
    f.planeWords = int32_t(planeOf / 4);
    return f;
}

// Scan -> entropy mode. AC bands are always non-interleaved (ns==1);
// DC bands may be interleaved or per-component.
static int scanMode(const Scan& sc) {
    if (sc.ss == 0 && sc.se == 63 && sc.ah == 0 && sc.al == 0)
        return 0;                        // sequential full band
    if (sc.ss == 0) return sc.ah == 0 ? 1 : 2;
    return sc.ah == 0 ? 3 : 4;
}

// ============================================================
//  Survey pass: virtual restart intervals for DRI=0 scans.
//
//  A serial Huffman walk over the entropy stream that consumes
//  symbols exactly like the GPU kernel — but only tracks decoder
//  state, never coefficient storage. At every kSurveyIv MCUs it
//  snapshots (pos, buf, cnt, end, pad, pred, eobrun): those become
//  "virtual restart intervals" — preloaded state turns the serial
//  stream back into independent parallel work items. Costs a few
//  percent of a full decode; AC-refine scans additionally need the
//  coefficient buffer from prior scans (deferred survey, done after
//  those scans' GPU dispatch completes).
// ============================================================
static constexpr uint32_t kSurveyIv = 256;   // MCUs per virtual interval

struct SBit {
    const uint8_t* d; uint32_t len;
    uint32_t pos = 0, buf = 0; int cnt = 0; bool end = false; int pad = 0;

    uint8_t byteAt(uint32_t p) const { return d[p]; }
    uint32_t rdByte() {
        if (end) return 0xFFu;
        uint32_t v = byteAt(pos);
        if (v != 0xFFu) { pos++; return v; }
        uint32_t p = pos + 1, n = 0xFFu;
        while (n == 0xFFu && p < len) {
            n = byteAt(p);
            if (n == 0xFFu) p++;
        }
        if (n == 0u) { pos = p + 1; return 0xFFu; }
        end = true;
        return 0xFFu;
    }
    void fill() {
        while (cnt <= 24) {
            uint32_t v = rdByte();
            buf = (buf << 8) | v; cnt += 8;
            if (end) pad += 8;
        }
    }
    uint32_t peekN(uint32_t n) {
        fill();
        if (uint32_t(cnt) >= n) return (buf >> (cnt - int(n))) & ((1u << n) - 1u);
        uint32_t real = uint32_t(cnt);
        uint32_t v = (buf & ((1u << real) - 1u)) << (n - real);
        return v | ((1u << (n - real)) - 1u);
    }
    void eat(uint32_t n) {
        cnt = (uint32_t(cnt) >= n) ? cnt - int(n) : 0;
        if (pad > cnt) pad = cnt;
        buf &= (cnt >= 32) ? 0xFFFFFFFFu : ((1u << uint32_t(cnt)) - 1u);
    }
    uint32_t get1() { uint32_t v = peekN(1); eat(1); return v; }
    int rext(int s) {
        if (!s) return 0;
        int i = int(peekN(uint32_t(s))); eat(uint32_t(s));
        return (i < (1 << (s - 1))) ? i - (1 << s) + 1 : i;
    }
};

static int hdec(SBit& b, const HuffTable& t) {
    int code = 0;
    for (int l = 1; l <= 16; ++l) {
        code = (code << 1) | int(b.get1());
        if (t.maxcode[l - 1] >= 0 && code <= t.maxcode[l - 1])
            return t.vals[t.valptr[l - 1] + code];
    }
    return -1;                                   // corrupt stream
}

// coef access for AC-refine surveys — same zigzag-pair packing as the
// GPU coef buffer. Returns 0 when no coefficient data is available.
struct SurvCoef {
    const uint32_t* w = nullptr;
    int get(uint32_t outp, int k) const {
        if (!w) return 0;
        uint32_t w2 = w[outp + uint32_t(k >> 1)];
        int v = int((w2 >> ((k & 1) * 16)) & 0xFFFFu);
        return (v >= 32768) ? v - 65536 : v;
    }
};

// One scan's entropy stream -> virtual interval table + per-interval
// preloaded st[] state. Mirrors the shader's per-MCU symbol walk for
// all five modes; mode 4 additionally reads coefs (nonzero walk).
// Returns false on unrecoverable stream corruption.
static bool surveyScan(const Jpg& j, Scan& sc, const SurvCoef& coef,
                       const FrameGpu& fg) {
    const int mode = scanMode(sc);
    for (int t = 0; t < 4; ++t) {
        buildCanon(const_cast<HuffTable&>(sc.dctab[t]));
        buildCanon(const_cast<HuffTable&>(sc.actab[t]));
    }
    SBit b{j.data.data(), uint32_t(j.data.size())};
    b.pos = sc.dataOff;
    int32_t pred[4] = {};
    uint32_t eobrun = 0;
    sc.intervals.clear();
    sc.survSt.clear();
    const uint32_t total = sc.totalMcus;

    auto snap = [&](uint32_t mcuStart) {
        if (!sc.intervals.empty())
            sc.intervals.back().nMcu =
                mcuStart - sc.intervals.back().mcuStart;
        Interval iv{b.pos - sc.dataOff, mcuStart, 0};
        sc.intervals.push_back(iv);
        const uint32_t sw[12] = {b.pos, b.buf, uint32_t(b.cnt),
                                 b.end ? 1u : 0u,
                                 uint32_t(pred[0]), uint32_t(pred[1]),
                                 uint32_t(pred[2]), uint32_t(pred[3]),
                                 eobrun, uint32_t(b.pad), 0, 0};
        sc.survSt.insert(sc.survSt.end(), sw, sw + 12);
    };

    for (uint32_t mcu = 0; mcu < total; ++mcu) {
        if (mcu % kSurveyIv == 0) snap(mcu);
        if (!(b.end && b.cnt <= b.pad)) {
            // decode this MCU's symbols (skip only when nothing but
            // marker padding remains — mirrors the shader's check)
            if (sc.ns > 1) {
                int mx = int(mcu) % j.mcusX, my = int(mcu) / j.mcusX;
                for (int ci = 0; ci < sc.ns; ++ci) {
                    const Comp& c = j.comps[sc.sc[ci].comp];
                    const HuffTable &dt = sc.dctab[sc.sc[ci].td],
                                    &at = sc.actab[sc.sc[ci].ta];
                    uint32_t cofs = uint32_t(fg.comps[sc.sc[ci].comp][9]);
                    for (int v = 0; v < c.vs; ++v)
                    for (int h = 0; h < c.hs; ++h) {
                        int bi = (my * c.vs + v) * c.bw + mx * c.hs + h;
                        uint32_t outp = cofs + uint32_t(bi) * 32u;
                        if (mode == 0) {
                            int s = hdec(b, dt);
                            if (s < 0) return false;
                            pred[ci] += b.rext(s);
                            int k = 1;
                            while (k < 64) {
                                int sym = hdec(b, at);
                                if (sym < 0) return false;
                                int r = sym >> 4; s = sym & 15;
                                if (!s) {
                                    if (r == 15) { k += 16; continue; }
                                    break;
                                }
                                k += r;
                                if (s) b.rext(s);   // consumed even OOB
                                k++;
                            }
                        } else if (mode == 1) {
                            int s = hdec(b, dt);
                            if (s < 0) return false;
                            pred[ci] += b.rext(s);
                        } else if (mode == 2) {
                            b.get1();
                        }
                    }
                }
            } else {
                int comp = sc.sc[0].comp;
                const HuffTable &dt = sc.dctab[sc.sc[0].td],
                                &at = sc.actab[sc.sc[0].ta];
                int wib1 = j.comps[comp].wib, bw = j.comps[comp].bw;
                uint32_t cofs = uint32_t(fg.comps[comp][9]);
                int bx = int(mcu) % wib1, by = int(mcu) / wib1;
                uint32_t outp = cofs + uint32_t(by * bw + bx) * 32u;
                if (mode == 0 || mode == 1) {
                    int s = hdec(b, dt);
                    if (s < 0) return false;
                    pred[0] += b.rext(s);
                    if (mode == 0) {
                        int k = 1;
                        while (k < 64) {
                            int sym = hdec(b, at);
                            if (sym < 0) return false;
                            int r = sym >> 4; s = sym & 15;
                            if (!s) {
                                if (r == 15) { k += 16; continue; }
                                break;
                            }
                            k += r;
                            if (s) b.rext(s);       // consumed even OOB
                            k++;
                        }
                    }
                } else if (mode == 2) {
                    b.get1();
                } else if (mode == 3) {
                    if (eobrun > 0u) { eobrun--; }
                    else {
                        int k = sc.ss;
                        while (k <= sc.se) {
                            int sym = hdec(b, at);
                            if (sym < 0) return false;
                            int r = sym >> 4, s = sym & 15;
                            if (s) {
                                k += r;
                                if (s) b.rext(s);   // consumed even OOB
                                k++;
                            } else if (r == 15) {
                                k += 16;
                            } else {
                                eobrun = 1u << r;
                                if (r) eobrun += b.peekN(uint32_t(r));
                                if (r) b.eat(uint32_t(r));
                                eobrun--;
                                break;
                            }
                        }
                    }
                } else {   // mode 4 — AC refinement
                    int k = sc.ss;
                    if (eobrun == 0u) {
                        while (k <= sc.se) {
                            int sym = hdec(b, at);
                            if (sym < 0) return false;
                            int r = sym >> 4, s = sym & 15;
                            if (s == 0 && r != 15) {
                                eobrun = 1u << r;
                                if (r) eobrun += b.peekN(uint32_t(r));
                                if (r) b.eat(uint32_t(r));
                                break;
                            }
                            if (s != 0) b.get1();   // new-coef sign bit
                            do {
                                int v = coef.get(outp, k);
                                if (v != 0) {
                                    b.get1();      // correction bit
                                } else {
                                    r--;
                                    if (r < 0) break;
                                }
                                k++;
                            } while (k <= sc.se);
                            k++;
                        }
                    }
                    if (eobrun > 0u) {
                        for (; k <= sc.se; ++k) {
                            if (coef.get(outp, k) != 0) b.get1();
                        }
                        eobrun--;
                    }
                }
            }
        }
    }
    // close the last interval
    if (!sc.intervals.empty())
        sc.intervals.back().nMcu = total - sc.intervals.back().mcuStart;
    return true;
}

// ============================================================
//  libjpeg reference: coefficients (jpegtran path) + RGB pixels
// ============================================================

struct Ref {
    std::vector<uint8_t> rgb;        // W*H*3 (or gray expanded x3)
    std::vector<int16_t> coefs[4];   // per comp, bw*bh*64, natural order
    int cbw[4] = {}, cbh[4] = {};    // per-comp block dims (NOT MCU-padded)
    int W = 0, H = 0;
};

static bool refDecode(const std::string& path, Ref& r, bool wantCoefs,
                      int scaleNum = 1, int scaleDenom = 1) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    jpeg_decompress_struct cinfo{};
    jpeg_error_mgr jerr{};
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, f);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.scale_num = JDIMENSION(scaleNum);
    cinfo.scale_denom = JDIMENSION(scaleDenom);

    if (wantCoefs) {
        jvirt_barray_ptr* arr = jpeg_read_coefficients(&cinfo);
        for (int ci = 0; ci < cinfo.num_components; ++ci) {
            jpeg_component_info* c = cinfo.comp_info + ci;
            r.cbw[ci] = int(c->width_in_blocks);
            r.cbh[ci] = int(c->height_in_blocks);
            r.coefs[ci].resize(size_t(c->width_in_blocks) * c->height_in_blocks * 64);
            for (JDIMENSION y = 0; y < c->height_in_blocks; ++y) {
                JBLOCKARRAY row = (*cinfo.mem->access_virt_barray)(
                    (j_common_ptr)&cinfo, arr[ci], y, 1, TRUE);
                for (JDIMENSION x = 0; x < c->width_in_blocks; ++x)
                    memcpy(&r.coefs[ci][(y * c->width_in_blocks + x) * 64],
                           row[0][x], 64 * sizeof(JCOEF));
            }
        }
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(f);
        return true;
    }

    cinfo.out_color_space = JCS_RGB;
    cinfo.do_fancy_upsampling = TRUE;
    jpeg_start_decompress(&cinfo);
    r.W = int(cinfo.output_width);
    r.H = int(cinfo.output_height);
    r.rgb.resize(size_t(r.W) * r.H * 3);
    while (cinfo.output_scanline < cinfo.output_height) {
        uint8_t* row = r.rgb.data() + size_t(cinfo.output_scanline) * r.W * 3;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    return true;
}

// ============================================================
//  Shader 1 — entropy decode: one invocation per restart interval
//  (per scan). mode: 0 sequential, 1 DC-first, 2 DC-refine,
//  3 AC-first, 4 AC-refine (jdphuff.c, bit-exact).
// ============================================================
static const char* kEntropy = R"GLSL(
#version 460
layout(local_size_x = 32) in;

layout(set = 0, binding = 0) readonly buffer Scan { uint scan[]; };
layout(set = 0, binding = 1) readonly buffer Chunks { uvec4 ck[]; };
layout(set = 0, binding = 2) readonly buffer Tabs { uvec4 tab[]; };
#define T(i) int(tab[tabOfs + ((i) >> 2)][(i) & 3])
layout(set = 0, binding = 3) buffer State { uint st[]; };
layout(set = 0, binding = 4) uniform Comps { ivec4 comps[4][4]; };
#define CM(c, i) (comps[c][(i) >> 2][(i) & 3])
layout(set = 0, binding = 8) readonly buffer FLut { int flut[]; };
layout(set = 0, binding = 5) buffer Coef { uint coef[]; };
layout(push_constant) uniform P {
    uint W, H, mcusX, nComp, maxH, maxV, nIv, scanLen, genOfs;
    uint mode, ns, ss, se, ah, al, totalMcus;
    uvec4 scomp;        // per scan comp: comp | td<<8 | ta<<16
    uint tabOfs;        // uvec4 units into tab[] for this scan's tables
    uint flutOfs;       // int elements into flut[] for this scan's LUTs
};

const uint kTabBase = 256;          // table t base = 256 + t*816

// --- bit reader: MSB-first, byte-packed words in scan[] ---
struct BR { uint pos; uint buf; int cnt; bool end; uint wpos; uint wval;
            uint nval; int pad; };

uint byteAt(uint p) {
    return (scan[p >> 2] >> ((p & 3u) * 8u)) & 0xFFu;
}

uint rdByte(inout BR b) {
    if (b.end) return 0xFFu;                       // pad 1s
    uint wp = b.pos >> 2;
    if (wp != b.wpos) {
        // rotate in prefetched next word, prefetch the following one
        b.wval = (wp == b.wpos + 1u) ? b.nval : scan[wp];
        b.nval = (wp + 1u < ((scanLen + 3u) >> 2)) ? scan[wp + 1u] : 0u;
        b.wpos = wp;
    }
    uint v = (b.wval >> ((b.pos & 3u) * 8u)) & 0xFFu;
    if (v != 0xFFu) { b.pos++; return v; }
    // 0xFF: crawl over fill-byte run (spec allows FF+ before markers),
    // then classify the byte that follows (jdhuff.c jpeg_fill_bit_buffer)
    uint p = b.pos + 1;
    uint n = 0xFFu;
    while (n == 0xFFu && p < scanLen) {
        n = byteAt(p);
        if (n == 0xFFu) p++;
    }
    if (n == 0u) { b.pos = p + 1; b.wpos = 0xFFFFFFFEu; return 0xFFu; }
    b.end = true;                                  // marker: stop
    return 0xFFu;
}
void fill(inout BR b) {
    // pad counts the trailing buffered bits that are marker padding, not
    // real data — an MCU may only be skipped when nothing but pad remains.
    while (b.cnt <= 24) {
        uint v = rdByte(b);
        b.buf = (b.buf << 8) | v; b.cnt += 8;
        if (b.end) b.pad += 8;
    }
}
uint peekN(inout BR b, uint n) {
    fill(b);
    if (uint(b.cnt) >= n) return (b.buf >> (uint(b.cnt) - n)) & ((1u << n) - 1u);
    uint real = uint(b.cnt);
    uint v = (b.buf & ((1u << real) - 1u)) << (n - real);
    return v | ((1u << (n - real)) - 1u);
}
void eat(inout BR b, uint n) {
    b.cnt = (uint(b.cnt) >= n) ? b.cnt - int(n) : 0;
    if (b.pad > b.cnt) b.pad = b.cnt;   // real bits are consumed first
    b.buf &= (b.cnt >= 32) ? 0xFFFFFFFFu : ((1u << uint(b.cnt)) - 1u);
}
uint get1(inout BR b) { uint v = peekN(b, 1u); eat(b, 1u); return v; }

int rext(inout BR b, int s) {          // receive + extend fused
    if (s == 0) return 0;
    uint v = peekN(b, uint(s)); eat(b, uint(s));
    int i = int(v);
    int vt = 1 << (s - 1);
    return (i < vt) ? i - (1 << s) + 1 : i;
}

// coef buffer: i16 pairs per u32 word, zigzag order — 32 words/block
int getCoef(uint outp, int k) {
    uint w2 = coef[outp + uint(k >> 1)];
    int v = int((w2 >> ((k & 1) * 16)) & 0xFFFFu);
    return (v >= 32768) ? v - 65536 : v;
}
void setCoef(uint outp, int k, int v) {
    uint wi = outp + uint(k >> 1);
    uint w2 = coef[wi];
    uint sh = uint(k & 1) * 16u;
    w2 = (w2 & ~(0xFFFFu << sh)) | ((uint(v) & 0xFFFFu) << sh);
    coef[wi] = w2;
}

// ---- sequential (baseline) block: DC diff + full AC band ----
void seqBlock(inout BR b, uint outp, uint dt, uint at, inout int pred) {
    // -- DC: fused entry -> sign-extended diff in one lookup
    int e = flut[flutOfs + uint(dt) * 65536u + peekN(b, 16u)];
    bool doAc = true;
    int diff;
    if (e < 0) {
        eat(b, uint(e & 0x1F));
        diff = ((e >> 13) & 0xFFFF) - 32768;
    } else {
        int len = e & 0x1F;
        if (len == 0) { doAc = false; diff = 0; }
        else { eat(b, uint(len)); diff = rext(b, (e >> 8) & 0xFF); }
    }
    pred += diff;
    uint pw = 0u, pv = uint(pred) & 0xFFFFu;
    // -- AC: ZRL/EOB need no stores; nonzero coefs gather
    //    into the pending word
    if (doAc) {
        int k = 1;
        while (k < 64) {
            e = flut[flutOfs + uint(at) * 65536u + peekN(b, 16u)];
            int rr, ss, val;
            if (e < 0) {
                eat(b, uint(e & 0x1F));
                rr = (e >> 5) & 0xF; ss = (e >> 9) & 0xF;
                val = ((e >> 13) & 0xFFFF) - 32768;
            } else {
                int len = e & 0x1F;
                if (len == 0) break;
                eat(b, uint(len));
                int sym = (e >> 8) & 0xFF;
                rr = sym >> 4; ss = sym & 15;
                val = (ss > 0) ? rext(b, ss) : 0;
            }
            if (ss == 0) {
                if (rr == 15) { k += 16; continue; }
                break;                // EOB
            }
            k += rr;
            if (k < 64) {
                uint wv = uint(k >> 1);
                if (wv != pw) { coef[outp + pw] = pv; pw = wv; pv = 0u; }
                pv |= (uint(val) & 0xFFFFu) << ((k & 1) * 16);
            }
            k++;
        }
    }
    coef[outp + pw] = pv;
}

// ---- progressive modes (jdphuff.c) ----

// DC first: coef[0] = (pred += diff) << Al
void dcFirst(inout BR b, uint outp, uint dt, inout int pred) {
    int e = flut[flutOfs + uint(dt) * 65536u + peekN(b, 16u)];
    int diff = 0;
    if (e < 0) {
        eat(b, uint(e & 0x1F));
        diff = ((e >> 13) & 0xFFFF) - 32768;
    } else {
        int len = e & 0x1F;
        if (len > 0) { eat(b, uint(len)); diff = rext(b, (e >> 8) & 0xFF); }
    }
    pred += diff;
    setCoef(outp, 0, pred << int(al));
}

// DC refine: one correction bit per block — coef[0] |= (1 << Al)
void dcRefine(inout BR b, uint outp) {
    if (get1(b) != 0u)
        setCoef(outp, 0, getCoef(outp, 0) | (1 << int(al)));
}

// AC first: band Ss..Se, v << Al writes, EOBRUN zero-runs
void acFirst(inout BR b, uint outp, uint at, inout uint eobrun) {
    if (eobrun > 0u) { eobrun--; return; }   // EOB block: no bits read
    int k = int(ss);
    while (k <= int(se)) {
        int e = flut[flutOfs + uint(at) * 65536u + peekN(b, 16u)];
        int rr, s, val;
        if (e < 0) {
            eat(b, uint(e & 0x1F));
            rr = (e >> 5) & 0xF; s = (e >> 9) & 0xF;
            val = ((e >> 13) & 0xFFFF) - 32768;
        } else {
            int len = e & 0x1F;
            if (len == 0) break;
            eat(b, uint(len));
            int sym = (e >> 8) & 0xFF;
            rr = sym >> 4; s = sym & 15;
            val = (s > 0) ? rext(b, s) : 0;
        }
        if (s != 0) {
            k += rr;
            if (k <= int(se)) setCoef(outp, k, val << int(al));
            k++;
        } else if (rr == 15) {       // ZRL
            k += 16;
        } else {                     // EOBr: 2^r + appended bits
            eobrun = 1u << rr;
            if (rr > 0) { eobrun += peekN(b, uint(rr)); eat(b, uint(rr)); }
            eobrun--;                // this block consumed one
            break;
        }
    }
}

// AC refine: correction bits on already-nonzero coefs, new +-p1 coefs
void acRefine(inout BR b, uint outp, uint at, inout uint eobrun) {
    int p1 = 1 << int(al), m1 = -p1;
    int k = int(ss);
    if (eobrun == 0u) {
        while (k <= int(se)) {
            int e = flut[flutOfs + uint(at) * 65536u + peekN(b, 16u)];
            int r, s, newv = 0;
            if (e < 0) {
                eat(b, uint(e & 0x1F));
                r = (e >> 5) & 0xF; s = (e >> 9) & 0xF;
                if (s != 0) newv = (((e >> 13) & 0xFFFF) - 32768) << int(al);
            } else {
                int len = e & 0x1F;
                if (len == 0) break;
                eat(b, uint(len));
                int sym = (e >> 8) & 0xFF;
                r = sym >> 4; s = sym & 15;
                if (s != 0)
                    newv = (get1(b) != 0u) ? p1 : m1;
            }
            if (s == 0 && r != 15) {   // EOBr: rest of band = corrections
                eobrun = 1u << r;
                if (r > 0) { eobrun += peekN(b, uint(r)); eat(b, uint(r)); }
                break;
            }
            // walk: correction bits on nonzeros, count down r zeros
            do {
                int v = getCoef(outp, k);
                if (v != 0) {
                    if (get1(b) != 0u && (v & p1) == 0)
                        setCoef(outp, k, v + ((v >= 0) ? p1 : m1));
                } else {
                    r--;
                    if (r < 0) break;      // target zero position
                }
                k++;
            } while (k <= int(se));
            if (s != 0 && k <= int(se))
                setCoef(outp, k, newv);
            k++;
        }
    }
    if (eobrun > 0u) {
        // rest of band (or whole block under EOBRUN): correction bits
        for (; k <= int(se); ++k) {
            int v = getCoef(outp, k);
            if (v != 0 && get1(b) != 0u && (v & p1) == 0)
                setCoef(outp, k, v + ((v >= 0) ? p1 : m1));
        }
        eobrun--;
    }
}

void main() {
    if (gl_GlobalInvocationID.x >= nIv) return;   // dispatch overrun
    uint idx = genOfs + gl_GlobalInvocationID.x;  // work item = (iv,chunk) grid
    uvec4 w = ck[idx];
    if (w.z == 0u) return;                        // padded grid cell
    BR b;
    ivec4 pred;
    uint eobrun = 0u;
    if (w.w == 0u) {                              // first chunk of interval
        b.pos = w.x; b.buf = 0u; b.cnt = 0; b.end = false;
        b.wpos = 0xFFFFFFFEu; b.wval = 0u; b.nval = 0u; b.pad = 0;
        pred = ivec4(0);
    } else {
        // w.w == 1: resume predecessor generation's state at idx-nIv;
        // w.w == 2: host-surveyed "virtual restart" state at own slot
        uint s = (w.w == 1u ? idx - nIv : idx) * 12u;
        b.pos = st[s]; b.buf = st[s + 1]; b.cnt = int(st[s + 2]);
        b.end = st[s + 3] != 0u;
        b.wpos = 0xFFFFFFFEu; b.wval = 0u; b.nval = 0u;
        b.pad = int(st[s + 9]);
        pred = ivec4(int(st[s + 4]), int(st[s + 5]),
                     int(st[s + 6]), int(st[s + 7]));
        eobrun = st[s + 8];
    }

    for (uint m = 0; m < w.z; ++m) {
        int mcu = int(w.y + m);
        if (mcu >= int(totalMcus)) break;      // malformed interval table
        if (b.end && b.cnt <= b.pad) continue; // insufficient data: leave MCU
        if (ns > 1u) {                         // interleaved: MCU grid
            int mx = mcu % int(mcusX), my = mcu / int(mcusX);
            for (int ci = 0; ci < int(ns); ++ci) {
                uint scw = scomp[ci];
                int comp = int(scw & 0xFFu);
                uint dt = (scw >> 8) & 0xFFu, at = 4u + ((scw >> 16) & 0xFFu);
                int hs = CM(comp, 0), vs = CM(comp, 1);
                int bw = CM(comp, 2);
                uint cofs = uint(CM(comp, 9));
                for (int v = 0; v < vs; ++v)
                for (int h = 0; h < hs; ++h) {
                    int bi = (my * vs + v) * bw + mx * hs + h;
                    uint outp = cofs + uint(bi) * 32u;
                    switch (mode) {
                    case 0u: seqBlock(b, outp, dt, at, pred[ci]); break;
                    case 1u: dcFirst(b, outp, dt, pred[ci]); break;
                    case 2u: dcRefine(b, outp); break;
                    }
                }
            }
        } else {                               // non-interleaved: block raster
            int comp = int(scomp[0] & 0xFFu);
            uint dt = (scomp[0] >> 8) & 0xFFu, at = 4u + ((scomp[0] >> 16) & 0xFFu);
            int wib1 = CM(comp, 11);
            int bw = CM(comp, 2);
            uint cofs = uint(CM(comp, 9));
            int bx = mcu % wib1, by = mcu / wib1;
            uint outp = cofs + uint(by * bw + bx) * 32u;
            switch (mode) {
            case 0u: seqBlock(b, outp, dt, at, pred[0]); break;
            case 1u: dcFirst(b, outp, dt, pred[0]); break;
            case 2u: dcRefine(b, outp); break;
            case 3u: acFirst(b, outp, at, eobrun); break;
            case 4u: acRefine(b, outp, at, eobrun); break;
            }
        }
    }

    // hand bitstate + DC predictors + EOBRUN to the next generation
    uint s = idx * 12u;
    st[s] = b.pos; st[s + 1] = b.buf; st[s + 2] = uint(b.cnt);
    st[s + 3] = b.end ? 1u : 0u;
    st[s + 4] = uint(pred[0]); st[s + 5] = uint(pred[1]);
    st[s + 6] = uint(pred[2]); st[s + 7] = uint(pred[3]);
    st[s + 8] = eobrun; st[s + 9] = uint(b.pad);
}
)GLSL";
// ============================================================
//  Shader 2 — block reconstruction: dequant + islow int IDCT
//  (libjpeg-turbo jidctint.c, CONST_BITS=13 PASS1_BITS=2 — bit-exact)
// ============================================================
static const char* kRecon = R"GLSL(
#version 460
layout(local_size_x = 64) in;

layout(set = 0, binding = 2) readonly buffer Tabs { uvec4 tab[]; };
#define T(i) int(tab[tabOfs + ((i) >> 2)][(i) & 3])
layout(set = 0, binding = 4) uniform Comps { ivec4 comps[4][4]; };
#define CM(c, i) (comps[c][(i) >> 2][(i) & 3])
layout(set = 0, binding = 5) readonly buffer Coef { uint coef[]; };
layout(set = 0, binding = 6) writeonly buffer Planes { uint planes[]; };
layout(push_constant) uniform P { uint W, H, mcusX, nComp, maxH, maxV,
                                  nIv, scanLen, tabOfs;
                                  uint winX0, winY0, winX1, winY1; };

const int zig[64] = int[64](
    0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,
    12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,
    35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63);

int desc(int x, int n) { return (x + (1 << (n - 1))) >> n; }

// one full islow pass over 8 values (shared by both directions)
// idx selects the 8 input strided positions; returns via ws[]
void idct1d(in int inp[8], out int outp[8], int sc, int nsh) {
    int z2 = inp[2] * sc, z3 = inp[6] * sc;
    int z1 = (z2 + z3) * 4433;                    // FIX_0_541196100
    int t2 = z1 - z3 * 15137;                     // FIX_1_847759065
    int t3 = z1 + z2 * 6270;                      // FIX_0_765366865
    z2 = inp[0] * sc; z3 = inp[4] * sc;
    int t0 = (z2 + z3) << 13, t1 = (z2 - z3) << 13;
    int t10 = t0 + t3, t13 = t0 - t3;
    int t11 = t1 + t2, t12 = t1 - t2;
    t0 = inp[7] * sc; t1 = inp[5] * sc;
    t2 = inp[3] * sc; t3 = inp[1] * sc;
    z1 = t0 + t3; z2 = t1 + t2;
    int z3o = t0 + t2, z4 = t1 + t3;
    int z5 = (z3o + z4) * 9633;                   // FIX_1_175875602
    t0 *= 2446;  t1 *= 16819; t2 *= 25172; t3 *= 12299;
    z1 *= -7373; z2 *= -20995; z3o *= -16069; z4 *= -3196;
    z3o += z5; z4 += z5;
    t0 += z1 + z3o; t1 += z2 + z4;
    t2 += z2 + z3o; t3 += z1 + z4;
    outp[0] = desc(t10 + t3, nsh); outp[7] = desc(t10 - t3, nsh);
    outp[1] = desc(t11 + t2, nsh); outp[6] = desc(t11 - t2, nsh);
    outp[2] = desc(t12 + t1, nsh); outp[5] = desc(t12 - t1, nsh);
    outp[3] = desc(t13 + t0, nsh); outp[4] = desc(t13 - t0, nsh);
}

// --- scaled IDCTs (jidctred.c): reduced-size 4x4/2x2/1x1, bit-exact ---
// column pass for the 4x4 variant: reads rows 0-3,5-7 of one input
// column (already dequantized, natural order) -> ws[4] at wsptr[c].
void idct44col(in int deq[64], int col, inout int ws[32]) {
    if (deq[8 + col] == 0 && deq[16 + col] == 0 && deq[24 + col] == 0 &&
        deq[40 + col] == 0 && deq[48 + col] == 0 && deq[56 + col] == 0) {
        int dcval = deq[col] << 2;              // PASS1_BITS
        ws[col] = dcval; ws[8 + col] = dcval;
        ws[16 + col] = dcval; ws[24 + col] = dcval;
        return;
    }
    int tmp0 = deq[col] << 14;                  // CONST_BITS+1
    int z2 = deq[16 + col], z3 = deq[48 + col];
    int tmp2 = z2 * 15137 + z3 * -6270;         // 1.847759065, -0.765366865
    int tmp10 = tmp0 + tmp2, tmp12 = tmp0 - tmp2;
    int z1 = deq[56 + col]; z2 = deq[40 + col];
    z3 = deq[24 + col]; int z4 = deq[8 + col];
    tmp0 = z1 * -1730 + z2 * 11893 + z3 * -17799 + z4 * 8697;
    tmp2 = z1 * -4176 + z2 * -4926 + z3 * 7373 + z4 * 20995;
    ws[col]      = desc(tmp10 + tmp2, 12);      // CONST_BITS-PASS1_BITS+1
    ws[24 + col] = desc(tmp10 - tmp2, 12);
    ws[8 + col]  = desc(tmp12 + tmp0, 12);
    ws[16 + col] = desc(tmp12 - tmp0, 12);
}
// row pass: ws row r -> 4 u8 samples
void idct44row(in int ws[32], int r, inout uint w) {
    if (ws[r * 8 + 1] == 0 && ws[r * 8 + 2] == 0 && ws[r * 8 + 3] == 0 &&
        ws[r * 8 + 5] == 0 && ws[r * 8 + 6] == 0 && ws[r * 8 + 7] == 0) {
        int dcval = clamp(desc(ws[r * 8], 5) + 128, 0, 255);
        w = uint(dcval) * 0x01010101u;
        return;
    }
    int tmp0 = ws[r * 8] << 14;
    int tmp2 = ws[r * 8 + 2] * 15137 + ws[r * 8 + 6] * -6270;
    int tmp10 = tmp0 + tmp2, tmp12 = tmp0 - tmp2;
    int z1 = ws[r * 8 + 7], z2 = ws[r * 8 + 5];
    int z3 = ws[r * 8 + 3], z4 = ws[r * 8 + 1];
    tmp0 = z1 * -1730 + z2 * 11893 + z3 * -17799 + z4 * 8697;
    tmp2 = z1 * -4176 + z2 * -4926 + z3 * 7373 + z4 * 20995;
    w = uint(clamp(desc(tmp10 + tmp2, 19) + 128, 0, 255)) |
        (uint(clamp(desc(tmp12 + tmp0, 19) + 128, 0, 255)) << 8) |
        (uint(clamp(desc(tmp12 - tmp0, 19) + 128, 0, 255)) << 16) |
        (uint(clamp(desc(tmp10 - tmp2, 19) + 128, 0, 255)) << 24);
}

// 2x2 variant: columns 0,1,3,5,7 (skip 2,4,6) -> ws[2] per column
void idct22col(in int deq[64], int col, inout int ws[16]) {
    if (deq[8 + col] == 0 && deq[24 + col] == 0 && deq[40 + col] == 0 &&
        deq[56 + col] == 0) {
        int dcval = deq[col] << 2;
        ws[col] = dcval; ws[8 + col] = dcval;
        return;
    }
    int tmp10 = deq[col] << 15;                 // CONST_BITS+2
    int tmp0 = deq[56 + col] * -5906 +          // c7-c5+c3-c1
               deq[40 + col] * 6967 +           // -c1+c3+c5+c7
               deq[24 + col] * -10426 +         // -c1+c3-c5-c7
               deq[8 + col] * 29692;            // c1+c3+c5+c7
    ws[col]     = desc(tmp10 + tmp0, 13);       // CONST_BITS-PASS1_BITS+2
    ws[8 + col] = desc(tmp10 - tmp0, 13);
}
void idct22row(in int ws[16], int r, inout uint w) {
    if (ws[r * 8 + 1] == 0 && ws[r * 8 + 3] == 0 && ws[r * 8 + 5] == 0 &&
        ws[r * 8 + 7] == 0) {
        int dcval = clamp(desc(ws[r * 8], 5) + 128, 0, 255);
        w = uint(dcval) * 0x01010101u;
        return;
    }
    int tmp10 = ws[r * 8] << 15;
    int tmp0 = ws[r * 8 + 7] * -5906 + ws[r * 8 + 5] * 6967 +
               ws[r * 8 + 3] * -10426 + ws[r * 8 + 1] * 29692;
    w = uint(clamp(desc(tmp10 + tmp0, 20) + 128, 0, 255)) |
        (uint(clamp(desc(tmp10 - tmp0, 20) + 128, 0, 255)) << 8);
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    // find which component this block belongs to (bounded — dispatch
    // overrun invocations must not walk past the last component)
    int c = 0;
    while (c < int(nComp) - 1 && gid >= uint(CM(c, 2) * CM(c, 3))) {
        gid -= uint(CM(c, 2) * CM(c, 3)); c++;
    }
    if (c >= int(nComp) || int(gid) >= CM(c, 2) * CM(c, 3)) return;
    int bw = CM(c, 2);
    int bx = int(gid) % bw, by = int(gid) / bw;
    // crop: skip blocks outside the needed source region (block bx
    // covers src px [bx*8*maxH/hs, ..) — same window math as host)
    {
        int hs = CM(c, 0), vs = CM(c, 1);
        int sx = 8 * int(maxH), sy = 8 * int(maxV);
        int bx0 = int(winX0) * hs / sx;
        int bx1 = (int(winX1) * hs + sx - 1) / sx;
        int by0 = int(winY0) * vs / sy;
        int by1 = (int(winY1) * vs + sy - 1) / sy;
        if (bx < bx0 || bx >= bx1 || by < by0 || by >= by1) return;
    }
    int qt = CM(c, 6);
    int ss = CM(c, 13);                        // per-comp DCT_scaled_size
    uint cofs = uint(CM(c, 9));
    uint pofs = uint(CM(c, 10)) >> 2;        // bytes -> words

    // dequantize (zigzag buffer -> natural order)
    int deq[64];
    for (int i = 0; i < 64; i += 2) {
        uint w2 = coef[cofs + uint(bx + by * bw) * 32u + uint(i / 2)];
        int v0 = int(w2 & 0xFFFFu); if (v0 >= 32768) v0 -= 65536;
        int v1 = int(w2 >> 16);     if (v1 >= 32768) v1 -= 65536;
        deq[zig[i]]     = v0 * T(qt * 64 + zig[i]);
        deq[zig[i + 1]] = v1 * T(qt * 64 + zig[i + 1]);
    }

    // ss in {1,2}: plane is word-per-sample (byte packing would race
    // between neighbouring blocks sharing a word); ss in {4,8} is
    // byte-packed (block tiles stay word-aligned)
    int stride = bw * ss;
    if (ss == 1) {                             // 1x1: scaled DC average
        planes[pofs + uint(by * stride + bx)] =
            uint(clamp(desc(deq[0], 3) + 128, 0, 255));
        return;
    }
    if (ss == 2) {
        int ws[16];
        for (int col = 0; col < 8; ++col) {
            if (col == 2 || col == 4 || col == 6) continue;
            idct22col(deq, col, ws);
        }
        for (int r = 0; r < 2; ++r) {
            uint w = 0u; idct22row(ws, r, w);
            uint base = pofs + uint((by * 2 + r) * stride + bx * 2);
            planes[base] = w & 0xFFu;
            planes[base + 1] = (w >> 8) & 0xFFu;
        }
        return;
    }
    if (ss == 4) {
        int ws[32];
        for (int col = 0; col < 8; ++col) {
            if (col == 4) continue;
            idct44col(deq, col, ws);
        }
        for (int r = 0; r < 4; ++r) {
            uint w = 0u; idct44row(ws, r, w);
            planes[pofs + (uint((by * 4 + r) * stride + bx * 4) >> 2)] = w;
        }
        return;
    }

    // pass 1: columns (dequant inside the pass, like jidctint)
    int ws[64];
    for (int col = 0; col < 8; ++col) {
        int inp[8], outp[8];
        for (int r = 0; r < 8; ++r) inp[r] = deq[r * 8 + col];
        idct1d(inp, outp, 1, 11);                 // scale 1, descale 11
        for (int r = 0; r < 8; ++r) ws[r * 8 + col] = outp[r];
    }
    // pass 2: rows -> u8, pack 4 per word
    for (int r = 0; r < 8; ++r) {
        int inp[8], outp[8];
        for (int cc = 0; cc < 8; ++cc) inp[cc] = ws[r * 8 + cc];
        idct1d(inp, outp, 1, 18);                 // descale 18
        uint base = pofs + uint(by * 8 + r) * uint(bw * 2) + uint(bx * 2);
        uint w0 = 0u, w1 = 0u;
        for (int cc = 0; cc < 4; ++cc) {
            int a = clamp(outp[cc] + 128, 0, 255);
            int b = clamp(outp[cc + 4] + 128, 0, 255);
            w0 |= uint(a) << (cc * 8);
            w1 |= uint(b) << (cc * 8);
        }
        planes[base] = w0;
        planes[base + 1] = w1;
    }
}
)GLSL";

// ============================================================
//  Shader 3 — upsample + color convert (per output pixel)
//  Mirrors jdsample.c fancy upsampling (bias +1/+2, +7/+8) and
//  jdcolor.c fixed-point YCbCr->RGB — target: bit-exact vs libjpeg.
// ============================================================
static const char* kUpsample = R"GLSL(
#version 460
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 4) uniform Comps { ivec4 comps[4][4]; };
#define CM(c, i) (comps[c][(i) >> 2][(i) & 3])
layout(set = 0, binding = 6) readonly buffer Planes { uint planes[]; };
layout(set = 0, binding = 7) writeonly buffer OutB { uint outp[]; };
layout(push_constant) uniform P { uint W, H, mcusX, nComp, maxH, maxV,
                                  nIv, scanLen, smin;
                                  uint cropX, cropY, uW, uH, orient; };

// fetch one sample from component c's plane; ss in {1,2} planes are
// word-per-sample, {4,8} byte-packed — matches the recon stage
int smp(int c, int x, int y) {
    uint pofs = uint(CM(c, 10));
    int ss = CM(c, 13);
    if (ss <= 2)   // word-per-sample: pofs>>2 + word index
        return int(planes[(pofs >> 2) + uint(y * CM(c, 2) * ss + x)]);
    int stride = CM(c, 2) * ss;                // samples per plane row
    uint a = pofs + uint(y * stride + x);
    return int((planes[a >> 2] >> ((a & 3u) * 8u)) & 0xFFu);
}

int clampi(int v) { return clamp(v, 0, 255); }

// upsampled component value at output pixel (x,y)
int usamp(int c, int x, int y) {
    int hs = CM(c, 0), vs = CM(c, 1);
    int cw = CM(c, 4), ch = CM(c, 5);
    int ss = CM(c, 13);
    // jdsample group math: input group = samp*ss/min_ssize vs max_samp
    int rh = (int(maxH) * int(smin)) / (hs * ss);
    int rv = (int(maxV) * int(smin)) / (vs * ss);
    bool fancy = smin > 1u;                    // do_fancy && min>1
    if (rh == 1 && rv == 1)
        return smp(c, x, y);
    if (fancy && rh == 2 && rv == 1 && cw > 2) {  // h2v1 fancy
        int j = clamp(x >> 1, 0, cw - 1);
        if (x == 0) return smp(c, 0, y);
        if (x == cw * 2 - 1) return smp(c, cw - 1, y);
        if ((x & 1) == 0)
            return (3 * smp(c, j, y) + smp(c, j - 1, y) + 1) >> 2;
        return (3 * smp(c, j, y) + smp(c, j + 1, y) + 2) >> 2;
    }
    if (fancy && rh == 1 && rv == 2) {           // h1v2 fancy
        int i = clamp(y >> 1, 0, ch - 1);
        int adj = clamp(i + (((y & 1) == 0) ? -1 : 1), 0, ch - 1);
        int bias = ((y & 1) == 0) ? 1 : 2;
        return (3 * smp(c, x, i) + smp(c, x, adj) + bias) >> 2;
    }
    if (fancy && rh == 2 && rv == 2 && cw > 2) { // h2v2 fancy
        int i = clamp(y >> 1, 0, ch - 1);
        int i2 = clamp(i + (((y & 1) == 0) ? -1 : 1), 0, ch - 1);
        int j = clamp(x >> 1, 0, cw - 1);
        int this0 = 3 * smp(c, j, i) + smp(c, j, i2);   // colsum j
        if (x == 0) return (this0 * 4 + 8) >> 4;
        if (x == cw * 2 - 1) return (this0 * 4 + 7) >> 4;
        int j2 = clamp(j + (((x & 1) == 0) ? -1 : 1), 0, cw - 1);
        int adj0 = 3 * smp(c, j2, i) + smp(c, j2, i2);
        int bias = ((x & 1) == 0) ? 8 : 7;
        return (this0 * 3 + adj0 + bias) >> 4;
    }
    // integral factors: plain replication (also non-fancy h2v1/h2v2)
    int cx = clamp(x / rh, 0, cw - 1);
    int cy = clamp(y / rv, 0, ch - 1);
    return smp(c, cx, cy);
}

// EXIF orientation: final output (x,y) -> un-oriented coords (uW x uH)
ivec2 omap(int x, int y) {
    switch (orient) {
    case 2u: return ivec2(int(uW) - 1 - x, y);
    case 3u: return ivec2(int(uW) - 1 - x, int(uH) - 1 - y);
    case 4u: return ivec2(x, int(uH) - 1 - y);
    case 5u: return ivec2(y, x);
    case 6u: return ivec2(y, int(uH) - 1 - x);
    case 7u: return ivec2(int(uW) - 1 - y, int(uH) - 1 - x);
    case 8u: return ivec2(int(uW) - 1 - y, x);
    default: return ivec2(x, y);
    }
}

void main() {
    uint x = gl_GlobalInvocationID.x, y = gl_GlobalInvocationID.y;
    if (x >= W || y >= H) return;
    // crop selects a rect of the ORIENTED image; omap maps back to the
    // un-oriented plane coordinates the samplers index
    ivec2 uv = omap(int(x) + int(cropX), int(y) + int(cropY));
    int r, g, bl;
    if (nComp == 1u) {
        r = g = bl = usamp(0, uv.x, uv.y);
    } else {
        int yy = usamp(0, uv.x, uv.y);
        int cb = usamp(1, uv.x, uv.y) - 128;
        int cr = usamp(2, uv.x, uv.y) - 128;
        r = clampi(yy + ((91881 * cr + 32768) >> 16));
        g = clampi(yy + ((32768 - 22554 * cb - 46802 * cr) >> 16));
        bl = clampi(yy + ((116130 * cb + 32768) >> 16));
    }
    outp[y * W + x] = 0xFF000000u | uint(bl) << 16 | uint(g) << 8 | uint(r);
}
)GLSL";

// ============================================================
//  Vulkan plumbing
// ============================================================

struct Gpu {
    vkmini::Vk v;
    vk::UniqueDescriptorSetLayout dsetLayout;
    vk::UniqueDescriptorPool dpool;
    vk::DescriptorSet dset;
    vk::UniquePipelineLayout pipeLayout;
    vk::UniquePipeline pEntropy, pRecon, pUpsample;

    void init() {
        v.createInstance({});
        v.pickPhysicalDevice();
        v.createDevice();
        std::vector<vk::DescriptorSetLayoutBinding> binds;
        for (uint32_t i = 0; i < 9; ++i)
            binds.push_back({i,
                             (i == 4)
                                 ? vk::DescriptorType::eUniformBuffer
                                 : vk::DescriptorType::eStorageBuffer,
                             1, vk::ShaderStageFlagBits::eCompute});
        dsetLayout = v.device->createDescriptorSetLayoutUnique({{}, binds});
        std::vector<vk::DescriptorPoolSize> sz = {
            {vk::DescriptorType::eStorageBuffer, 9},
            {vk::DescriptorType::eUniformBuffer, 1}};
        dpool = v.device->createDescriptorPoolUnique(
            {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, sz});
        dset = v.device->allocateDescriptorSets(
            {dpool.get(), dsetLayout.get()}).front();
        vk::PushConstantRange pc{vk::ShaderStageFlagBits::eCompute, 0, 88};
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
        pEntropy = mk(kEntropy, "entropy");
        pRecon = mk(kRecon, "recon");
        pUpsample = mk(kUpsample, "upsample");
    }

    vkmini::Vk::Buffer ssbo(vk::DeviceSize bytes, bool map = false,
                            vk::BufferUsageFlags extra = {}) {
        return v.createBuffer(
            bytes, vk::BufferUsageFlagBits::eStorageBuffer | extra,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent,
            map);
    }

    void bindBuf(uint32_t binding, const vkmini::Vk::Buffer& b,
                 bool ubo = false) {
        vk::DescriptorBufferInfo bi{b.buf.get(), 0, b.size};
        vk::WriteDescriptorSet w{dset, binding, 0, 1,
                                 ubo ? vk::DescriptorType::eUniformBuffer
                                     : vk::DescriptorType::eStorageBuffer,
                                 nullptr, &bi};
        v.device->updateDescriptorSets(w, {});
    }

    void barrier(vk::CommandBuffer c) {
        vk::MemoryBarrier mb{vk::AccessFlagBits::eShaderWrite |
                                 vk::AccessFlagBits::eShaderRead,
                             vk::AccessFlagBits::eShaderRead |
                                 vk::AccessFlagBits::eShaderWrite};
        c.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                          vk::PipelineStageFlagBits::eAllCommands,
                          {}, mb, {}, {});
    }
};

// Host mirror of the shader crop+orient path: un-oriented RGB uW x uH
// -> oriented (orient transform), then crop [x,y,w,h) in oriented
// space. Returns (rgb, W, H) of the final image.
struct RgbImg { std::vector<uint8_t> px; int W = 0, H = 0; };

static RgbImg cropOrientRgb(const std::vector<uint8_t>& rgb, int uW,
                            int uH, int orient, int cx, int cy, int cw,
                            int ch) {
    auto omap = [&](int x, int y) {
        switch (orient) {
        case 2: return std::pair{uW - 1 - x, y};
        case 3: return std::pair{uW - 1 - x, uH - 1 - y};
        case 4: return std::pair{x, uH - 1 - y};
        case 5: return std::pair{y, x};
        case 6: return std::pair{y, uH - 1 - x};
        case 7: return std::pair{uW - 1 - y, uH - 1 - x};
        case 8: return std::pair{uW - 1 - y, x};
        default: return std::pair{x, y};
        }
    };
    RgbImg o;
    o.W = cw; o.H = ch;
    o.px.resize(size_t(cw) * ch * 3);
    for (int y = 0; y < ch; ++y)
        for (int x = 0; x < cw; ++x) {
            auto [ux, uy] = omap(x + cx, y + cy);
            for (int k = 0; k < 3; ++k)
                o.px[(size_t(y) * cw + x) * 3 + k] =
                    rgb[(size_t(uy) * uW + ux) * 3 + k];
        }
    return o;
}

static void savePpm(const char* path, const uint8_t* rgba, int W, int H) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";
    for (int i = 0; i < W * H; ++i) {
        char px[3] = {char(rgba[i * 4]), char(rgba[i * 4 + 1]),
                      char(rgba[i * 4 + 2])};
        f.write(px, 3);
    }
}

int main(int argc, char** argv) {
    bool cpuMode = false, verify = false;
    int bench = 0;
    int scaleNum = 1, scaleDenom = 1;
    int cropW = -1, cropH = 0, cropX = 0, cropY = 0; // -1 => full image
    std::string orientArg = "none";                  // none|auto|1..8
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cpu") cpuMode = true;
        else if (a == "--verify") verify = true;
        else if (a == "--bench") bench = atoi(argv[++i]);
        else if (a == "--scale") {
            if (++i >= argc ||
                sscanf(argv[i], "%d/%d", &scaleNum, &scaleDenom) != 2) {
                fprintf(stderr, "--scale needs M/N (like djpeg -scale)\n");
                return 2;
            }
        }
        else if (a == "--crop") {
            if (++i >= argc ||
                sscanf(argv[i], "%dx%d+%d+%d", &cropW, &cropH,
                       &cropX, &cropY) != 4) {
                fprintf(stderr, "--crop needs WxH+X+Y\n");
                return 2;
            }
        }
        else if (a == "--orient") {
            if (++i >= argc) { fprintf(stderr, "--orient needs arg\n"); return 2; }
            orientArg = argv[i];
        }
        else pos.push_back(a);
    }
    // djpeg -scale M/N picks the smallest DCT_scaled_size N with
    // scale_num*8 <= scale_denom*N (jdmaster.c); we implement the
    // classic reduced IDCTs for N in {1,2,4,8} (1/8, 1/4, 1/2, 1/1)
    int smin = scaleDenom > 0
                   ? (scaleNum * 8 + scaleDenom - 1) / scaleDenom : 8;
    if (smin < 1) smin = 1;
    if (smin > 8) smin = 8;
    if (smin != 1 && smin != 2 && smin != 4 && smin != 8) {
        fprintf(stderr, "unsupported DCT_scaled_size %d "
                "(only 1/8, 1/4, 1/2 scales implemented)\n", smin);
        return 2;
    }
    if (pos.empty() || (!cpuMode && !verify && !bench && pos.size() < 2)) {
        fprintf(stderr,
            "usage: app [--cpu|--verify|--bench N|--scale M/N]\n"
            "           [--crop WxH+X+Y] [--orient none|auto|1-8] "
            "in.jpg [out.ppm]\n");
        return 2;
    }
    const std::string inPath = pos[0];

    if (cpuMode) {
        // need the parsed header for --orient auto's EXIF value
        int o2 = 1;
        if (orientArg != "none") {
            Jpg h;
            std::ifstream f(inPath, std::ios::binary);
            h.data.assign(std::istreambuf_iterator<char>(f), {});
            if (!parseJpg(h)) return 1;
            o2 = orientArg == "auto" ? h.orient : atoi(orientArg.c_str());
            if (o2 < 1 || o2 > 8) o2 = 1;
        }
        Ref r;
        if (!refDecode(inPath, r, false, scaleNum, scaleDenom)) return 1;
        int oW2 = o2 >= 5 ? r.H : r.W, oH2 = o2 >= 5 ? r.W : r.H;
        int cx2 = std::clamp(cropX, 0, oW2), cy2 = std::clamp(cropY, 0, oH2);
        int cw2 = cropW < 0 ? oW2 : std::min(cropW, oW2 - cx2);
        int ch2 = cropW < 0 ? oH2 : std::min(cropH, oH2 - cy2);
        RgbImg t = cropOrientRgb(r.rgb, r.W, r.H, o2, cx2, cy2, cw2, ch2);
        savePpm(pos[1].c_str(), [&] {
            static std::vector<uint8_t> rgba;
            rgba.resize(size_t(t.W) * t.H * 4);
            for (size_t i = 0; i < size_t(t.W) * t.H; ++i) {
                rgba[i * 4] = t.px[i * 3];
                rgba[i * 4 + 1] = t.px[i * 3 + 1];
                rgba[i * 4 + 2] = t.px[i * 3 + 2];
                rgba[i * 4 + 3] = 255;
            }
            return rgba.data();
        }(), t.W, t.H);
        printf("cpu: wrote %s (%dx%d)\n", pos[1].c_str(), t.W, t.H);
        return 0;
    }

    // ---- parse on host ----
    Jpg j;
    {
        std::ifstream f(inPath, std::ios::binary);
        j.data.assign(std::istreambuf_iterator<char>(f), {});
    }
    if (!parseJpg(j)) return 1;
    fprintf(stderr, "%dx%d %d comp  mcu %dx%d  dri %d  %zu scans %s\n",
            j.W, j.H, j.nComp, j.mcusX, j.mcusY, j.dri,
            j.scans.size(), j.progressive ? "(progressive)" : "");
    FrameGpu fg = packComps(j, smin);
    // output dims like jdmaster: ceil(image * min_ssize / 8)
    const uint32_t outW = uint32_t(int64_t(j.W) * smin + 7) / 8;
    const uint32_t outH = uint32_t(int64_t(j.H) * smin + 7) / 8;

    // ---- orientation + crop -----------------------------------------
    int orient = 1;
    if (orientArg == "auto") orient = j.orient;
    else if (orientArg != "none") orient = atoi(orientArg.c_str());
    if (orient < 1 || orient > 8) orient = 1;
    const uint32_t uW = outW, uH = outH;            // un-oriented dims
    const uint32_t oW = orient >= 5 ? uH : uW;      // oriented dims
    const uint32_t oH = orient >= 5 ? uW : uH;
    // crop rect in oriented space (default = whole image)
    uint32_t cw = cropW < 0 ? oW : uint32_t(cropW);
    uint32_t ch = cropW < 0 ? oH : uint32_t(cropH);
    cropX = std::clamp(cropX, 0, int(oW));
    cropY = std::clamp(cropY, 0, int(oH));
    cw = std::min(cw, oW - uint32_t(cropX));
    ch = std::min(ch, oH - uint32_t(cropY));
    const uint32_t fW = cw, fH = ch;                // final output dims
    // needed region in un-oriented output space: omap applied to the
    // crop rect (each axis maps monotonically -> corners bound it)
    auto corner = [&](uint32_t x, uint32_t y) {
        switch (orient) {
        case 2: return std::pair{uW - 1 - x, y};
        case 3: return std::pair{uW - 1 - x, uH - 1 - y};
        case 4: return std::pair{x, uH - 1 - y};
        case 5: return std::pair{y, x};
        case 6: return std::pair{y, uH - 1 - x};
        case 7: return std::pair{uW - 1 - y, uH - 1 - x};
        case 8: return std::pair{uW - 1 - y, x};
        default: return std::pair{x, y};
        }
    };
    uint32_t ux0, uy0, ux1, uy1;
    {
        auto [ax, ay] = corner(uint32_t(cropX), uint32_t(cropY));
        auto [bx, by] =
            corner(uint32_t(cropX) + cw - 1, uint32_t(cropY) + ch - 1);
        ux0 = std::min(ax, bx); ux1 = std::max(ax, bx) + 1;
        uy0 = std::min(ay, by); uy1 = std::max(ay, by) + 1;
    }
    // needed region in source pixels (map back by 8/smin, pad by one
    // MCU — covers fancy-upsample's +-1 comp-sample reads)
    const int64_t marg = 8LL * std::max(j.maxH, j.maxV);
    const int64_t srcX0 = std::clamp<int64_t>(
        int64_t(ux0) * 8 / smin - marg, 0, j.W);
    const int64_t srcX1 = std::clamp<int64_t>(
        int64_t(ux1) * 8 / smin + marg, 0, j.W);
    const int64_t srcY0 = std::clamp<int64_t>(
        int64_t(uy0) * 8 / smin - marg, 0, j.H);
    const int64_t srcY1 = std::clamp<int64_t>(
        int64_t(uy1) * 8 / smin + marg, 0, j.H);
    // AC-refine surveys (deferred) read prior coefs — and refine
    // decode inside the shader reads coefs too. Any comp with a
    // deferred scan therefore needs EVERY scan touching it decoded in
    // full, else the survey walks desync on zero coefs.
    bool defComp[4] = {};
    for (const auto& sc : j.scans)
        if (sc.dri == 0 && sc.totalMcus > kSurveyIv &&
            scanMode(sc) == 4)
            defComp[sc.sc[0].comp] = true;
    auto scanFull = [&](const Scan& sc) {
        for (int i = 0; i < sc.ns; ++i)
            if (defComp[sc.sc[i].comp]) return true;
        return false;
    };
    // does interval [ms,ms+n) of this scan overlap the needed rect?
    // intervals are skipped WHOLE (a partial range can't be entered
    // mid-stream anyway); the crop window gets an MCU of margin
    auto ivKeep = [&](const Scan& sc, const Interval& iv) {
        if (scanFull(sc)) return true;
        if (srcX0 >= srcX1 || srcY0 >= srcY1) return false;
        if (sc.ns == 1) {
            const Comp& c = j.comps[sc.sc[0].comp];
            // block b covers src px [b*8*maxH/hs, +8*maxH/hs)
            const int64_t sx = 8 * j.maxH, sy = 8 * j.maxV;
            int bx0 = int(srcX0 * c.hs / sx);
            int bx1 = int((srcX1 * c.hs + sx - 1) / sx);
            int by0 = int(srcY0 * c.vs / sy);
            int by1 = int((srcY1 * c.vs + sy - 1) / sy);
            int w = c.wib;
            int r0 = std::max(by0, int(iv.mcuStart) / w);
            int r1 = std::min(by1, int(iv.mcuStart + iv.nMcu - 1) / w + 1);
            for (int r = r0; r < r1; ++r) {
                int64_t lo = std::max<int64_t>(iv.mcuStart,
                                               int64_t(r) * w + bx0);
                int64_t hi = std::min<int64_t>(
                    iv.mcuStart + iv.nMcu, int64_t(r) * w + bx1);
                if (lo < hi) return true;
            }
            return false;
        }
        int mx0 = int(srcX0 / (8 * j.maxH));
        int mx1 = int((srcX1 + 8 * j.maxH - 1) / (8 * j.maxH));
        int my0 = int(srcY0 / (8 * j.maxV));
        int my1 = int((srcY1 + 8 * j.maxV - 1) / (8 * j.maxV));
        int w = j.mcusX;
        int r0 = std::max(my0, int(iv.mcuStart) / w);
        int r1 = std::min(my1, int(iv.mcuStart + iv.nMcu - 1) / w + 1);
        for (int r = r0; r < r1; ++r) {
            int64_t lo = std::max<int64_t>(iv.mcuStart,
                                           int64_t(r) * w + mx0);
            int64_t hi = std::min<int64_t>(iv.mcuStart + iv.nMcu,
                                           int64_t(r) * w + mx1);
            if (lo < hi) return true;
        }
        return false;
    };

    // ---- DRI=0 survey: virtual restart intervals --------------------
    // A serial scan bigger than kSurveyIv MCUs gets split into virtual
    // intervals by a cheap CPU symbol-walk (see surveyScan). AC-refine
    // scans additionally need the coef state of prior scans — their
    // survey is DEFERRED until those scans' GPU dispatch completed.
    const uint32_t nScans = uint32_t(j.scans.size());
    auto needsSurvey = [](const Scan& sc) {
        return sc.dri == 0 && sc.totalMcus > kSurveyIv;
    };
    auto deferredSurvey = [&](const Scan& sc) {
        return needsSurvey(sc) && scanMode(sc) == 4;
    };
    uint32_t nSurveyed = 0, nDeferred = 0;
    {
        SurvCoef noCoef{};
        for (auto& sc : j.scans) {
            if (!needsSurvey(sc) || deferredSurvey(sc)) continue;
            if (!surveyScan(j, sc, noCoef, fg)) {
                fprintf(stderr, "survey failed — serial decode\n");
                sc.survSt.clear();
                sc.intervals.assign(1, Interval{0, 0, sc.totalMcus});
                continue;
            }
            nSurveyed++;
        }
    }

    // ---- upload ----
    Gpu g;
    g.init();

    // ---- one-time upload: scan data, chunk grid, state slots, tables ----
    // Whole file as scan buffer: interval byteStarts are file-absolute.
    auto scanBuf = g.ssbo((j.data.size() + 3) & ~3u, true);
    memcpy(scanBuf.mapped, j.data.data(), j.data.size());

    const uint32_t kChunkMcus = 2048;

    // per-scan chunk entries concatenated into one buffer.
    // Surveyed scans emit flag=2 chunks (host-preloaded st[]);
    // deferred-survey scans emit the same mechanical MCU partition
    // now — the survey only fills st[] at dispatch time.
    std::vector<uint32_t> ckAll, ckBase(nScans), nIvV(nScans), nGenV(nScans);
    uint32_t totalSlots = 0;
    for (uint32_t si = 0; si < nScans; ++si) {
        const Scan& sc = j.scans[si];
        ckBase[si] = totalSlots;
        if (deferredSurvey(sc)) {
            // reserve + emit the fixed kSurveyIv partition now
            const uint32_t nIv = (sc.totalMcus + kSurveyIv - 1) / kSurveyIv;
            nIvV[si] = nIv; nGenV[si] = 1;
            for (uint32_t i = 0; i < nIv; ++i) {
                uint32_t nm = std::min(kSurveyIv, sc.totalMcus - i * kSurveyIv);
                Interval viv{0, i * kSurveyIv, nm};
                ckAll.push_back(0);
                ckAll.push_back(i * kSurveyIv);
                ckAll.push_back(ivKeep(sc, viv) ? nm : 0u);
                ckAll.push_back(2u);
            }
            totalSlots += nIv;
            nDeferred++;
            continue;
        }
        const uint32_t nIv = uint32_t(sc.intervals.size());
        uint32_t nGen = 1;
        for (const auto& iv : sc.intervals)
            nGen = std::max(nGen, (iv.nMcu + kChunkMcus - 1) / kChunkMcus);
        nIvV[si] = nIv;
        nGenV[si] = nGen;
        const bool surveyed = !sc.survSt.empty();
        for (uint32_t gI = 0; gI < nGen; ++gI)
            for (uint32_t i = 0; i < nIv; ++i) {
                const Interval& iv = sc.intervals[i];
                uint32_t ms = iv.mcuStart + gI * kChunkMcus;
                int64_t nm = int64_t(iv.mcuStart) + iv.nMcu - int64_t(ms);
                nm = std::clamp<int64_t>(nm, 0, kChunkMcus);
                if (!ivKeep(sc, iv)) nm = 0;   // crop: skip whole iv
                ckAll.push_back(gI == 0 ? sc.dataOff + iv.byteStart : 0);
                ckAll.push_back(ms);
                ckAll.push_back(uint32_t(nm));
                ckAll.push_back(surveyed ? 2u : (gI > 0 ? 1u : 0u));
            }
        totalSlots += nIv * nGen;
    }
    auto ivBuf = g.ssbo(ckAll.size() * 4, true);
    memcpy(ivBuf.mapped, ckAll.data(), ckAll.size() * 4);
    auto stBuf = g.ssbo(size_t(totalSlots) * 12 * 4, true);
    // preload host-surveyed interval states (flag=2 chunks)
    {
        uint32_t* stw = static_cast<uint32_t*>(stBuf.mapped);
        for (uint32_t si = 0; si < nScans; ++si) {
            const Scan& sc = j.scans[si];
            if (sc.survSt.empty()) continue;
            memcpy(stw + size_t(ckBase[si]) * 12, sc.survSt.data(),
                   sc.survSt.size() * 4);
        }
    }

    // per-scan Huffman/quant tables at region si*kTabWords, plus one
    // recon region (final quant state) at nScans*kTabWords
    auto tabBuf = g.ssbo(size_t(nScans + 1) * kTabWords * 4, true);
    for (auto& sc : j.scans) {
        for (auto& t : sc.dctab) buildCanon(t);
        for (auto& t : sc.actab) buildCanon(t);
    }
    {
        std::vector<int32_t> tab;
        for (uint32_t si = 0; si < nScans; ++si) {
            tab.clear();
            packTables(j.scans[si], tab);
            memcpy(static_cast<int32_t*>(tabBuf.mapped) + size_t(si) * kTabWords,
                   tab.data(), tab.size() * 4);
        }
        // recon region: only the quant tables are used (final state)
        std::vector<int32_t> qtReg(kTabWords, 0);
        for (int t = 0; t < 4; ++t)
            for (int i = 0; i < 64; ++i)
                qtReg[kQBase + t * 64 + i] = j.qt[t][i];
        memcpy(static_cast<int32_t*>(tabBuf.mapped) +
                   size_t(nScans) * kTabWords,
               qtReg.data(), qtReg.size() * 4);
    }

    // per-scan fast LUTs at region si*kFLutWords (dedup table slots
    // identical to the previous scan's — cjpeg re-emits some tables)
    auto flutBuf = g.ssbo(size_t(nScans) * kFLutWords * 4, true);
    {
        std::vector<uint32_t> fl(kFLutWords), flPrev;
        const Scan* prev = nullptr;
        for (uint32_t si = 0; si < nScans; ++si) {
            const Scan& sc = j.scans[si];
            for (int t = 0; t < 8; ++t) {
                const HuffTable& h = (t < 4) ? sc.dctab[t] : sc.actab[t - 4];
                if (prev &&
                    sameTable(h, (t < 4) ? prev->dctab[t] : prev->actab[t - 4])) {
                    memcpy(fl.data() + t * 65536, flPrev.data() + t * 65536,
                           65536 * 4);
                    continue;
                }
                memset(fl.data() + t * 65536, 0, 65536 * 4);
                // canonical walk for the first code in each 16-bit prefix
                for (int x = 0; x < 65536; ++x) {
                    int len = 0, sym = 0;
                    for (int l = 1; l <= 16; ++l) {
                        int code = x >> (16 - l);
                        if (h.maxcode[l - 1] >= 0 && code <= h.maxcode[l - 1]) {
                            len = l;
                            sym = h.vals[h.valptr[l - 1] + code];
                            break;
                        }
                    }
                    if (!len) continue;
                    int r = (t < 4) ? 0 : (sym >> 4), s = sym & 15;
                    if (t < 4) s = sym;
                    if (len + s <= 16) {
                        int ext = (x >> (16 - len - s)) & ((1 << s) - 1);
                        int val = extendRef(ext, s);
                        fl[t * 65536 + x] =
                            0x80000000u | uint32_t(len + s) |
                            uint32_t(r << 5) | uint32_t(s << 9) |
                            uint32_t((val + 32768) << 13);
                    } else {
                        fl[t * 65536 + x] = uint32_t(len) | uint32_t(sym << 8);
                    }
                }
            }
            memcpy(static_cast<uint32_t*>(flutBuf.mapped) +
                       size_t(si) * kFLutWords,
                   fl.data(), fl.size() * 4);
            flPrev = fl;
            prev = &sc;
        }
    }

    auto compBuf = g.ssbo(sizeof(fg.comps), true,
                          vk::BufferUsageFlagBits::eUniformBuffer);
    memcpy(compBuf.mapped, fg.comps, sizeof(fg.comps));

    // coefBuf is host-mapped: AC-refine surveys read prior scans'
    // coefficients directly out of it between submits.
    auto coefBuf = g.ssbo(size_t(fg.totalBlocks) * 32 * 4, true,
                          vk::BufferUsageFlagBits::eTransferDst);
    auto planeBuf = g.ssbo(size_t(fg.planeWords) * 4);
    auto outBuf = g.ssbo(size_t(fW) * fH * 4, true);

    g.bindBuf(0, scanBuf);
    g.bindBuf(1, ivBuf);
    g.bindBuf(2, tabBuf);
    g.bindBuf(3, stBuf);
    g.bindBuf(4, compBuf, true);
    g.bindBuf(5, coefBuf);
    g.bindBuf(6, planeBuf);
    g.bindBuf(7, outBuf);
    g.bindBuf(8, flutBuf);

    // ---- dispatch grouping: a single multi-second submit can get killed
    // by the driver's job watchdog on some GPUs — group (scan,generation)
    // dispatches into submits bounded by total MCU work. Files that fit
    // the budget stay single-submit.
    const uint32_t kSubmitMcus = getenv("SUBMITMCUS")
        ? uint32_t(atoi(getenv("SUBMITMCUS"))) : 32768;
    struct Job { uint32_t si, gI, mcus; };
    std::vector<Job> jobs;
    for (uint32_t si = 0; si < nScans; ++si) {
        const Scan& sc = j.scans[si];
        for (uint32_t gI = 0; gI < nGenV[si]; ++gI) {
            uint64_t w = 0;
            for (const auto& iv : sc.intervals) {
                int64_t nm = int64_t(iv.mcuStart) + iv.nMcu -
                             int64_t(iv.mcuStart + gI * kChunkMcus);
                w += uint64_t(std::clamp<int64_t>(nm, 0, kChunkMcus));
            }
            jobs.push_back({si, gI, uint32_t(w)});
        }
    }
    // greedy grouping into ranges whose total MCUs stay under budget;
    // a deferred-survey (AC-refine, DRI=0) scan must START a submit —
    // its st[] state is written only after all prior scans completed.
    std::vector<std::pair<uint32_t, uint32_t>> submits;
    for (uint32_t i = 0, acc = 0, start = 0; i <= uint32_t(jobs.size()); ++i) {
        if (i == uint32_t(jobs.size()) ||
            (acc > 0 && (acc + jobs[i].mcus > kSubmitMcus ||
                         deferredSurvey(j.scans[jobs[i].si])))) {
            submits.push_back({start, i});
            start = i; acc = 0;
            if (i == uint32_t(jobs.size())) break;
        }
        acc += jobs[i].mcus;
    }

    uint32_t pc[22] = {uint32_t(j.W),     uint32_t(j.H),
                       uint32_t(j.mcusX), uint32_t(j.nComp),
                       uint32_t(j.maxH),  uint32_t(j.maxV),
                       0,                 uint32_t(j.data.size()),
                       0u,                0u,
                       0u,                0u,
                       0u,                0u,
                       0u,                0u,
                       0u,                0u,
                       0u,                0u};
    auto recordEntropy = [&](vk::CommandBuffer c, uint32_t j0, uint32_t j1) {
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             g.pipeLayout.get(), 0, g.dset, {});
        c.bindPipeline(vk::PipelineBindPoint::eCompute, g.pEntropy.get());
        uint32_t lastSi = ~0u;
        for (uint32_t ji = j0; ji < j1; ++ji) {
            const Job& jb = jobs[ji];
            const uint32_t si = jb.si;
            const Scan& sc = j.scans[si];
            const uint32_t nIv = nIvV[si];
            if (si != lastSi) {           // scan-level fields once per scan
                pc[6] = nIv;
                pc[9] = uint32_t(scanMode(sc));
                pc[10] = uint32_t(sc.ns);
                pc[11] = uint32_t(sc.ss);
                pc[12] = uint32_t(sc.se);
                pc[13] = uint32_t(sc.ah);
                pc[14] = uint32_t(sc.al);
                pc[15] = sc.totalMcus;
                for (int i = 0; i < sc.ns; ++i)
                    pc[16 + i] = uint32_t(sc.sc[i].comp) |
                                 (uint32_t(sc.sc[i].td) << 8) |
                                 (uint32_t(sc.sc[i].ta) << 16);
                pc[20] = si * kTabWords / 4; // tabOfs in uvec4 units
                pc[21] = si * kFLutWords;    // flutOfs in int units
                lastSi = si;
            }
            pc[8] = ckBase[si] + jb.gI * nIv;
            c.pushConstants(g.pipeLayout.get(),
                            vk::ShaderStageFlagBits::eCompute, 0, 88, pc);
            g.barrier(c);
            c.dispatch((nIv + 31) / 32, 1, 1);
        }
    };
    auto decodeOnce = [&] {
        bool stagetimes = getenv("STAGE") != nullptr;
        auto t0 = std::chrono::steady_clock::now();
        int scanLim = getenv("SCANLIMIT") ? atoi(getenv("SCANLIMIT")) : int(nScans);
        const uint32_t scanMin = getenv("SCANMIN")
            ? uint32_t(atoi(getenv("SCANMIN"))) : 0;
        bool first = true;
        for (auto [j0, j1] : submits) {
            if (jobs[j0].si < scanMin || jobs[j0].si >= uint32_t(scanLim)) continue;
            uint32_t je = j1;   // clamp range end to scanLim
            while (je > j0 && jobs[je - 1].si >= uint32_t(scanLim)) --je;
            if (je <= j0) continue;
            Scan& sc0 = j.scans[jobs[j0].si];
            if (deferredSurvey(sc0) && sc0.survSt.empty()) {
                // all prior scans have completed (oneTime waits) —
                // coefBuf.mapped now holds their coefficients, which
                // the AC-refine survey needs for its nonzero walk.
                if (first)   // fillBuffer hasn't run yet (scanMin skip)
                    memset(coefBuf.mapped, 0, coefBuf.size);
                SurvCoef cc{static_cast<const uint32_t*>(coefBuf.mapped)};
                if (surveyScan(j, sc0, cc, fg)) {
                    memcpy(static_cast<uint32_t*>(stBuf.mapped) +
                               size_t(ckBase[jobs[j0].si]) * 12,
                           sc0.survSt.data(), sc0.survSt.size() * 4);
                } else {
                    // degrade to a single serial chunk: entry 0 covers
                    // the whole scan from its data offset (flag 0 =
                    // fresh state); remaining slots get nMcu=0.
                    fprintf(stderr,
                            "deferred survey failed (scan %u) — serial\n",
                            jobs[j0].si);
                    const uint32_t si = jobs[j0].si;
                    uint32_t* ck = static_cast<uint32_t*>(ivBuf.mapped) +
                                   size_t(ckBase[si]) * 4;
                    memset(ck, 0, nIvV[si] * 4 * sizeof(uint32_t));
                    ck[0] = sc0.dataOff;
                    ck[2] = sc0.totalMcus;
                }
            }
            g.v.oneTime([&](vk::CommandBuffer c) {
                if (first) {   // zero coefficient buffer once, up front
                    c.fillBuffer(coefBuf.buf.get(), 0, coefBuf.size, 0);
                    vk::MemoryBarrier mb0{
                        vk::AccessFlagBits::eTransferWrite,
                        vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eShaderWrite};
                    c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eComputeShader,
                                      {}, mb0, {}, {});
                }
                recordEntropy(c, j0, je);
            });
            first = false;
        }
        if (getenv("DBG")) fprintf(stderr, "submits=%zu, recon submit\n",
                                   submits.size());
        g.v.oneTime([&](vk::CommandBuffer c) {
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                 g.pipeLayout.get(), 0, g.dset, {});
            pc[8] = nScans * kTabWords / 4;  // tabOfs -> recon region
            pc[9] = uint32_t(srcX0); pc[10] = uint32_t(srcY0);
            pc[11] = uint32_t(srcX1); pc[12] = uint32_t(srcY1);
            c.pushConstants(g.pipeLayout.get(),
                            vk::ShaderStageFlagBits::eCompute, 0, 88, pc);
            c.bindPipeline(vk::PipelineBindPoint::eCompute, g.pRecon.get());
            g.barrier(c);
            c.dispatch((fg.totalBlocks + 63) / 64, 1, 1);
            g.barrier(c);
            // upsample: output dims + min scale + crop/orient fields
            // (different push layout — indices 9..13 get new meaning)
            pc[0] = fW; pc[1] = fH; pc[8] = uint32_t(smin);
            pc[9] = uint32_t(cropX); pc[10] = uint32_t(cropY);
            pc[11] = uW; pc[12] = uH; pc[13] = uint32_t(orient);
            c.pushConstants(g.pipeLayout.get(),
                            vk::ShaderStageFlagBits::eCompute, 0, 88, pc);
            c.bindPipeline(vk::PipelineBindPoint::eCompute,
                           g.pUpsample.get());
            c.dispatch((fW + 15) / 16, (fH + 15) / 16, 1);
        });
        if (stagetimes)
            fprintf(stderr, "decode (all stages): %.1f ms\n",
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count());
    };

    if (bench > 0) {
        decodeOnce();  // warm-up
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < bench; ++i) decodeOnce();
        double s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        double mp = double(j.W) * j.H * bench / 1e6;
        printf("%d decodes in %.2fs -> %.1f MP/s (%.1f MPix/decode)\n",
               bench, s, mp / s, double(j.W) * j.H / 1e6);
        return 0;
    }

    decodeOnce();

    if (verify || getenv("DBG")) {
        auto countNz = [&](vkmini::Vk::Buffer& b, const char* name) {
            const void* m = b.mapped ? b.mapped
                                     : g.v.device->mapMemory(b.mem.get(), 0, b.size);
            const auto* w = static_cast<const uint32_t*>(m);
            size_t nz = 0;
            for (size_t i = 0; i < b.size / 4; ++i) nz += w[i] != 0;
            fprintf(stderr, "%s: %zu/%zu nonzero words\n", name, nz,
                    b.size / 4);
            if (!b.mapped) g.v.device->unmapMemory(b.mem.get());
        };
        countNz(coefBuf, "coef");
        countNz(planeBuf, "planes");
        countNz(outBuf, "out");
    }

    if (pos.size() >= 2) {
        savePpm(pos[1].c_str(),
                static_cast<const uint8_t*>(outBuf.mapped), fW, fH);
        printf("wrote %s\n", pos[1].c_str());
    }

    if (verify) {
        // coef stage: GPU zigzag-packed i16 vs libjpeg natural JCOEF
        // (only meaningful uncropped: skipped intervals leave coef = 0)
        if (fW == oW && fH == oH) {
        Ref ref;
        if (!refDecode(inPath, ref, true)) return 1;
        const auto* cw = static_cast<const uint32_t*>([&] {
            void* m = g.v.device->mapMemory(coefBuf.mem.get(), 0,
                                            coefBuf.size);
            return m;
        }());
        int bad = 0, checked = 0, printed = 0;
        for (int c = 0; c < j.nComp; ++c) {
            const Comp& cc = j.comps[c];
            const auto& rc = ref.coefs[c];
            uint32_t base = uint32_t(fg.comps[c][9]);
            for (int by = 0; by < ref.cbh[c]; ++by)
            for (int bx = 0; bx < ref.cbw[c]; ++bx) {
                int b = by * cc.bw + bx;      // GPU layout is MCU-padded
                for (int z = 0; z < 64; ++z) {
                    uint32_t w = cw[base + b * 32 + z / 2];
                    int v = int((w >> ((z & 1) * 16)) & 0xFFFF);
                    if (v >= 32768) v -= 65536;
                    checked++;
                    int rv = rc[size_t(by * ref.cbw[c] + bx) * 64 + kZig[z]];
                    if (v != rv) {
                        bad++;
                        if (printed++ < 30)
                            printf("  c%d blk%d (bx%d by%d) z%d: gpu %d ref %d\n",
                                   c, b, bx, by, z, v, rv);
                    }
                }
            }
        }
        printf("coefs: %d/%d mismatch\n", bad, checked);
        } else {
            printf("coefs: skipped (crop)\n");
        }

        // final RGB vs libjpeg (apply the same orient+crop on the ref)
        Ref rr;
        refDecode(inPath, rr, false, scaleNum, scaleDenom);
        RgbImg rc = cropOrientRgb(rr.rgb, int(outW), int(outH), orient,
                                 cropX, cropY, int(fW), int(fH));
        const auto* px = static_cast<const uint8_t*>(outBuf.mapped);
        int md = 0; double sum = 0;
        for (uint32_t i = 0; i < fW * fH * 3; ++i) {
            int d = abs(int(px[(i / 3) * 4 + i % 3]) - int(rc.px[i]));
            md = std::max(md, d); sum += d;
        }
        printf("rgb: max |diff| %d  mean %.4f\n", md, sum / (fW * fH * 3));
    }
    return 0;
}
