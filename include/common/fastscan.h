#pragma once
// FastScan engine: 4-bit sub-codes of 32 vectors scored with in-register
// table lookups (PSHUFB / TBL), the batch layout of FAISS PQ4 FastScan and
// RaBitQ (SIGMOD 2024, §3.3.1).
//
// A code is a sequence of G 4-bit sub-codes. For a query, each group g has a
// 16-entry float LUT_g; the score of a code is sum_g LUT_g[subcode_g].
//
// Packed layout (n vectors, Gp = G rounded up to even, nb = ceil(n / 32)):
//   data[((b * Gp) + g) * 16 + j], j < 16:
//     low nibble  = subcode_g of vector 32b + j
//     high nibble = subcode_g of vector 32b + 16 + j
//   Padding vectors / groups hold 0 and are ignored by the caller.
//
// LUT quantization (per query): Delta = max_g (max LUT_g - min LUT_g) / 255,
//   lut_u8[g][k] = round((LUT_g[k] - min LUT_g) / Delta),
//   score ~ Delta * sum_g lut_u8[g][subcode_g] + sum_g min LUT_g.
// The rounding error per group is <= Delta / 2 (std Delta / sqrt(12)).
//
// Integer sums accumulate in uint16 lanes and are flushed to uint32 before
// they can overflow (at most 255 * 256 = 65280 per lane between flushes).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "config.h"

namespace vsq::common {
namespace fastscan {

constexpr size_t kBlock = 32;

struct PackedCodes {
    size_t n = 0;        // vectors
    size_t groups = 0;   // G
    size_t gpad = 0;     // G rounded up to even
    size_t blocks = 0;   // ceil(n / 32)
    std::vector<uint8_t> data;  // blocks * gpad * 16 bytes

    // subcode(i, g) for i < n, g < G returns a value in [0, 16).
    template <class SubcodeFn>
    static PackedCodes pack(size_t n, size_t G, SubcodeFn subcode) {
        if (G == 0) throw std::invalid_argument("FastScan: zero groups");
        PackedCodes p;
        p.n = n;
        p.groups = G;
        p.gpad = (G + 1) & ~size_t{1};
        p.blocks = (n + kBlock - 1) / kBlock;
        p.data.assign(p.blocks * p.gpad * 16, 0);
        for (size_t i = 0; i < n; ++i) {
            const size_t b = i / kBlock, j = i % kBlock;
            for (size_t g = 0; g < G; ++g) {
                const uint8_t s = static_cast<uint8_t>(subcode(i, g) & 0x0F);
                uint8_t &byte = p.data[(b * p.gpad + g) * 16 + (j & 15)];
                byte = static_cast<uint8_t>(j < 16 ? (byte | s) : (byte | (s << 4)));
            }
        }
        return p;
    }

    // Sub-code of vector i, group g (inverse of pack; used for exact rerank).
    uint8_t subcode(size_t i, size_t g) const {
        const size_t b = i / kBlock, j = i % kBlock;
        const uint8_t byte = data[(b * gpad + g) * 16 + (j & 15)];
        return j < 16 ? (byte & 0x0F) : (byte >> 4);
    }
};

// Quantized LUT for one query. lut has gpad * 16 bytes (padding groups 0).
struct QuantLut {
    std::vector<uint8_t> lut;
    float delta = 1.0f;
    float bias = 0.0f;  // sum_g min LUT_g

    // flut: G * 16 floats (LUT_g contiguous).
    void build(const float *flut, size_t G, size_t gpad) {
        lut.assign(gpad * 16, 0);
        float span = 0.0f;
        bias = 0.0f;
        for (size_t g = 0; g < G; ++g) {
            const float *t = flut + g * 16;
            const auto mm = std::minmax_element(t, t + 16);
            span = std::max(span, *mm.second - *mm.first);
            bias += *mm.first;
        }
        delta = span > 0.0f ? span / 255.0f : 1.0f;
        const float inv = 1.0f / delta;
        for (size_t g = 0; g < G; ++g) {
            const float *t = flut + g * 16;
            const float lo = *std::min_element(t, t + 16);
            for (size_t k = 0; k < 16; ++k) {
                const float v = std::nearbyint((t[k] - lo) * inv);
                lut[g * 16 + k] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, v)));
            }
        }
    }
    float score(uint32_t sum) const { return delta * static_cast<float>(sum) + bias; }
};

namespace detail {

// sums: 32 uint32 per block (blocks * 32 entries), overwritten.
inline void scanScalar(const PackedCodes &p, const uint8_t *lut, uint32_t *sums) {
    for (size_t b = 0; b < p.blocks; ++b) {
        uint32_t acc[kBlock] = {};
        const uint8_t *blk = p.data.data() + b * p.gpad * 16;
        for (size_t g = 0; g < p.gpad; ++g) {
            const uint8_t *t = lut + g * 16;
            const uint8_t *row = blk + g * 16;
            for (size_t j = 0; j < 16; ++j) {
                acc[j] += t[row[j] & 0x0F];
                acc[16 + j] += t[row[j] >> 4];
            }
        }
        std::memcpy(sums + b * kBlock, acc, sizeof(acc));
    }
}

#if defined(VSQ_NEON)
inline void scanNeon(const PackedCodes &p, const uint8_t *lut, uint32_t *sums) {
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    for (size_t b = 0; b < p.blocks; ++b) {
        uint32_t tot[kBlock] = {};
        const uint8_t *blk = p.data.data() + b * p.gpad * 16;
        size_t g = 0;
        while (g < p.gpad) {
            const size_t end = std::min(p.gpad, g + 256);
            uint16x8_t a0 = vdupq_n_u16(0), a1 = vdupq_n_u16(0);
            uint16x8_t a2 = vdupq_n_u16(0), a3 = vdupq_n_u16(0);
            for (; g < end; ++g) {
                const uint8x16_t v = vld1q_u8(blk + g * 16);
                const uint8x16_t t = vld1q_u8(lut + g * 16);
                const uint8x16_t rl = vqtbl1q_u8(t, vandq_u8(v, mask));  // vectors 0..15
                const uint8x16_t rh = vqtbl1q_u8(t, vshrq_n_u8(v, 4));   // vectors 16..31
                a0 = vaddw_u8(a0, vget_low_u8(rl));
                a1 = vaddw_high_u8(a1, rl);
                a2 = vaddw_u8(a2, vget_low_u8(rh));
                a3 = vaddw_high_u8(a3, rh);
            }
            uint16_t s[32];
            vst1q_u16(s, a0);
            vst1q_u16(s + 8, a1);
            vst1q_u16(s + 16, a2);
            vst1q_u16(s + 24, a3);
            for (size_t j = 0; j < kBlock; ++j) tot[j] += s[j];
        }
        std::memcpy(sums + b * kBlock, tot, sizeof(tot));
    }
}
#endif

#if defined(VSQ_HAVE_AVX2_KERNELS)
// Two groups per YMM register: lane 0 = group g, lane 1 = group g + 1.
VSQ_TARGET_AVX2 inline void scanAvx2(const PackedCodes &p, const uint8_t *lut,
                                            uint32_t *sums) {
    const __m256i mask = _mm256_set1_epi8(0x0F);
    const __m256i lo8 = _mm256_set1_epi16(0x00FF);
    for (size_t b = 0; b < p.blocks; ++b) {
        uint32_t tot[kBlock] = {};
        const uint8_t *blk = p.data.data() + b * p.gpad * 16;
        size_t g = 0;
        while (g < p.gpad) {
            const size_t end = std::min(p.gpad, g + 256);  // 128 pairs per flush
            __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
            __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
            for (; g < end; g += 2) {
                const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk + g * 16));
                const __m256i t = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lut + g * 16));
                const __m256i rl = _mm256_shuffle_epi8(t, _mm256_and_si256(v, mask));
                const __m256i rh =
                    _mm256_shuffle_epi8(t, _mm256_and_si256(_mm256_srli_epi16(v, 4), mask));
                a0 = _mm256_add_epi16(a0, _mm256_and_si256(rl, lo8));  // even vectors 0..14
                a1 = _mm256_add_epi16(a1, _mm256_srli_epi16(rl, 8));   // odd vectors 1..15
                a2 = _mm256_add_epi16(a2, _mm256_and_si256(rh, lo8));  // even vectors 16..30
                a3 = _mm256_add_epi16(a3, _mm256_srli_epi16(rh, 8));   // odd vectors 17..31
            }
            alignas(32) uint16_t s[4][16];
            _mm256_store_si256(reinterpret_cast<__m256i *>(s[0]), a0);
            _mm256_store_si256(reinterpret_cast<__m256i *>(s[1]), a1);
            _mm256_store_si256(reinterpret_cast<__m256i *>(s[2]), a2);
            _mm256_store_si256(reinterpret_cast<__m256i *>(s[3]), a3);
            for (size_t e = 0; e < 16; ++e) {  // u16 element e: lane e / 8, byte pair e % 8
                const size_t v2 = 2 * (e & 7);
                tot[v2] += s[0][e];
                tot[v2 + 1] += s[1][e];
                tot[16 + v2] += s[2][e];
                tot[17 + v2] += s[3][e];
            }
        }
        std::memcpy(sums + b * kBlock, tot, sizeof(tot));
    }
}
#endif

}  // namespace detail

// sums: p.blocks * 32 uint32 (entries >= p.n are padding).
inline void scan(const PackedCodes &p, const QuantLut &q, uint32_t *sums, Isa isa) {
#if defined(VSQ_HAVE_AVX2_KERNELS)
    if (isa == Isa::Avx2) return detail::scanAvx2(p, q.lut.data(), sums);
#endif
#if defined(VSQ_NEON)
    if (isa == Isa::Neon) return detail::scanNeon(p, q.lut.data(), sums);
#endif
    (void)isa;
    detail::scanScalar(p, q.lut.data(), sums);
}

// Indices of the `k` smallest values (ascending by value; ties by index).
inline std::vector<uint32_t> smallestK(const float *v, size_t n, size_t k) {
    k = std::min(k, n);
    std::vector<uint32_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = static_cast<uint32_t>(i);
    auto less = [v](uint32_t a, uint32_t b) { return v[a] < v[b] || (v[a] == v[b] && a < b); };
    if (k < n) std::nth_element(idx.begin(), idx.begin() + static_cast<long>(k), idx.end(), less);
    idx.resize(k);
    std::sort(idx.begin(), idx.end(), less);
    return idx;
}

}  // namespace fastscan
}  // namespace vsq::common
