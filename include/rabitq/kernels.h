#pragma once
// RaBitQ inner-product kernels for float queries (scalar / NEON / AVX2) and
// the integer helpers of the Extended RaBitQ encoder. Included by
// rabitq/space.h; AVX2 specialisations carry target attributes and are
// selected at runtime.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "../common/config.h"

namespace vsq::rabitq {
namespace detail {

// Which inner-product kernel a distance function instantiates.
// The constructor picks one ISA for the whole translation unit.
enum class DotIsa { Scalar, Neon, Avx2 };

// Bit i set => coordinate contributes +q_i. Bit clear => -q_i.
// Row b, lane k is 0 when bit k of b is set, else the float sign bit,
// so XOR with the query flips the sign only for a clear bit.
// Each row is 32 bytes and 32-byte aligned for an AVX2 aligned load.
inline const uint32_t *signXorMask(unsigned bits) {
#if defined(VSQ_NEON) || defined(VSQ_HAVE_AVX2_KERNELS)
    struct Table {
        alignas(32) uint32_t row[256][8];
        Table() {
            for (int b = 0; b < 256; ++b) {
                for (int k = 0; k < 8; ++k)
                    row[b][k] = (b & (1 << k)) ? 0u : 0x80000000u;
            }
        }
    };
    static const Table table;
    return table.row[bits & 255u];
#else
    (void)bits;
    return nullptr;
#endif
}

inline float dot1From(const float *q, const uint8_t *signs, size_t begin,
                      size_t n, float inv_sqrt_d) {
    float acc = 0.0f;
    for (size_t i = begin; i < n; ++i) {
        const unsigned bit = (signs[i >> 3] >> (i & 7u)) & 1u;
        const float s = bit ? inv_sqrt_d : -inv_sqrt_d;
        acc += s * q[i];
    }
    return acc;
}

inline float dot4From(const float *q, const uint8_t *packed, size_t begin,
                      size_t n) {
    float acc = 0.0f;
    for (size_t i = begin; i < n; ++i) {
        const uint8_t byte = packed[i >> 1];
        const uint8_t nib = (i & 1u) ? static_cast<uint8_t>(byte >> 4)
                                     : static_cast<uint8_t>(byte & 0x0Fu);
        acc += static_cast<float>(nib) * q[i];
    }
    return acc;
}

// IEEE-754 order, so a larger finite float gets a larger uint32.
inline uint32_t sortableFloat(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

// Magnitude on the extended grid. ex_bits = bits - 1, so max_code is
// 7 at 4 bits and 127 at 8 bits. q = code + 0.5 matches |grid - center|
// at centers 7.5 and 127.5. Thresholds are the integers k / magnitude,
// the same events Algorithm 1 sorts. magnitude >= 0, t >= 0, max_code >= 0.
inline int magnitudeAtScale(double magnitude, double t, int max_code) {
    if (!(magnitude > 0.0) || !(t > 0.0) || max_code <= 0)
        return 0;
    int code = static_cast<int>(
        std::min(t * magnitude, static_cast<double>(max_code)));
    if (code < 0)
        code = 0;
    if (code < max_code &&
        (static_cast<double>(code) + 1.0) / magnitude <= t)
        ++code;
    else if (code > 0 && static_cast<double>(code) / magnitude > t)
        --code;
    if (code < 0 || code > max_code)
        throw std::logic_error("RaBitQ magnitudeAtScale left the grid");
    return code;
}

// Fraction of [0, t_end] skipped before the per-vector search.
// Index is ex_bits. 3 -> 4-bit codes (0.52), 7 -> 8-bit codes (0.77).
// Values are the RaBitQ-Library table kTightStart.
inline float tightStart(int ex_bits) {
    static constexpr float kStart[9] = {
        0.00f, 0.15f, 0.20f, 0.52f, 0.59f, 0.71f, 0.75f, 0.77f, 0.81f};
    if (ex_bits < 0 || ex_bits >= 9)
        throw std::invalid_argument("RaBitQ: ex_bits out of the tight-start table");
    return kStart[ex_bits];
}

// Stable 8-bit LSD radix on the high 32 bits of (threshold << 32 | coordinate)
// keys: 4 passes instead of 8. Keys are generated in ascending coordinate
// order, so a stable sort by threshold alone yields (threshold, coordinate)
// order, the same result as sorting the full 64-bit key.
inline void radixSortU64(std::vector<uint64_t> &keys) {
    if (keys.size() < 2)
        return;
    std::vector<uint64_t> scratch(keys.size());
    for (int shift = 32; shift < 64; shift += 8) {
        size_t count[256] = {};
        for (uint64_t key : keys)
            ++count[(key >> shift) & 255u];
        size_t sum = 0;
        for (size_t &bin : count) {
            const size_t n = bin;
            bin = sum;
            sum += n;
        }
        for (uint64_t key : keys)
            scratch[count[(key >> shift) & 255u]++] = key;
        keys.swap(scratch);
    }
}

inline float dot8From(const float *q, const uint8_t *packed, size_t begin,
                      size_t n) {
    float acc = 0.0f;
    for (size_t i = begin; i < n; ++i)
        acc += static_cast<float>(packed[i]) * q[i];
    return acc;
}

// q has length n. signs/packed is the slot payload. n is padded_dim.
template <DotIsa Isa>
float dot1(const float *q, const uint8_t *signs, size_t n, float inv_sqrt_d);

template <DotIsa Isa>
float dot4(const float *q, const uint8_t *packed, size_t n);

template <DotIsa Isa>
float dot8(const float *q, const uint8_t *packed, size_t n);

template <>
inline float dot1<DotIsa::Scalar>(const float *q, const uint8_t *signs,
                                  size_t n, float inv_sqrt_d) {
    return dot1From(q, signs, 0, n, inv_sqrt_d);
}

template <>
inline float dot4<DotIsa::Scalar>(const float *q, const uint8_t *packed,
                                  size_t n) {
    return dot4From(q, packed, 0, n);
}

template <>
inline float dot8<DotIsa::Scalar>(const float *q, const uint8_t *packed,
                                  size_t n) {
    return dot8From(q, packed, 0, n);
}

#if defined(VSQ_NEON)
template <>
inline float dot1<DotIsa::Neon>(const float *q, const uint8_t *signs,
                                size_t n, float inv_sqrt_d) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint32_t *mask = signXorMask(signs[i >> 3]);
        float32x4_t q0 = vld1q_f32(q + i);
        float32x4_t q1 = vld1q_f32(q + i + 4);
        q0 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q0),
                                             vld1q_u32(mask)));
        q1 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q1),
                                             vld1q_u32(mask + 4)));
        acc0 = vaddq_f32(acc0, q0);
        acc1 = vaddq_f32(acc1, q1);
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1)) * inv_sqrt_d +
           dot1From(q, signs, i, n, inv_sqrt_d);
}

// 16 coordinates per iteration: 8 packed bytes, low nibble = even index.
template <>
inline float dot4<DotIsa::Neon>(const float *q, const uint8_t *packed,
                                size_t n) {
    const uint8x8_t mask = vdup_n_u8(0x0F);
    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f);
    float32x4_t s3 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const uint8x8_t bytes = vld1_u8(packed + (i >> 1));
        const uint8x8_t lo = vand_u8(bytes, mask);
        const uint8x8_t hi = vshr_n_u8(bytes, 4);
        const uint8x8x2_t z = vzip_u8(lo, hi);
        const uint16x8_t w0 = vmovl_u8(z.val[0]);
        const uint16x8_t w1 = vmovl_u8(z.val[1]);
        const float32x4_t f0 =
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(w0)));
        const float32x4_t f1 =
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(w0)));
        const float32x4_t f2 =
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(w1)));
        const float32x4_t f3 =
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(w1)));
        s0 = vmlaq_f32(s0, vld1q_f32(q + i), f0);
        s1 = vmlaq_f32(s1, vld1q_f32(q + i + 4), f1);
        s2 = vmlaq_f32(s2, vld1q_f32(q + i + 8), f2);
        s3 = vmlaq_f32(s3, vld1q_f32(q + i + 12), f3);
    }
    const float acc =
        vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    return acc + dot4From(q, packed, i, n);
}

template <>
inline float dot8<DotIsa::Neon>(const float *q, const uint8_t *packed,
                                size_t n) {
    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint8x8_t b = vld1_u8(packed + i);
        const uint16x8_t w = vmovl_u8(b);
        const float32x4_t f0 =
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
        const float32x4_t f1 =
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
        s0 = vmlaq_f32(s0, vld1q_f32(q + i), f0);
        s1 = vmlaq_f32(s1, vld1q_f32(q + i + 4), f1);
    }
    return vaddvq_f32(vaddq_f32(s0, s1)) + dot8From(q, packed, i, n);
}
#endif

#if defined(VSQ_HAVE_AVX2_KERNELS)
VSQ_AVX2_INLINE float hsum256(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    const __m128 s4 = _mm_add_ps(lo, hi);
    const __m128 dup = _mm_movehdup_ps(s4);
    const __m128 sums = _mm_add_ps(s4, dup);
    const __m128 high = _mm_movehl_ps(sums, sums);
    return _mm_cvtss_f32(_mm_add_ss(sums, high));
}

template <>
VSQ_TARGET_AVX2 inline float dot1<DotIsa::Avx2>(const float *q, const uint8_t *signs,
                                size_t n, float inv_sqrt_d) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i mask = _mm256_load_si256(
            reinterpret_cast<const __m256i *>(signXorMask(signs[i >> 3])));
        __m256 qv = _mm256_loadu_ps(q + i);
        qv = _mm256_xor_ps(qv, _mm256_castsi256_ps(mask));
        acc = _mm256_add_ps(acc, qv);
    }
    return hsum256(acc) * inv_sqrt_d + dot1From(q, signs, i, n, inv_sqrt_d);
}

// 16 coordinates per iteration. unpacklo(low nibble, high nibble) is the
// same even/odd order as NEON vzip_u8.
template <>
VSQ_TARGET_AVX2 inline float dot4<DotIsa::Avx2>(const float *q, const uint8_t *packed,
                                size_t n) {
    const __m128i nibble = _mm_set1_epi8(0x0F);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m128i bytes =
            _mm_loadl_epi64(reinterpret_cast<const __m128i *>(packed + (i >> 1)));
        const __m128i lo = _mm_and_si128(bytes, nibble);
        const __m128i hi =
            _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble);
        const __m128i codes = _mm_unpacklo_epi8(lo, hi);
        const __m128i c0 = _mm_cvtepu8_epi32(codes);
        const __m128i c1 = _mm_cvtepu8_epi32(_mm_srli_si128(codes, 4));
        const __m128i c2 = _mm_cvtepu8_epi32(_mm_srli_si128(codes, 8));
        const __m128i c3 = _mm_cvtepu8_epi32(_mm_srli_si128(codes, 12));
        const __m256 f0 = _mm256_cvtepi32_ps(_mm256_set_m128i(c1, c0));
        const __m256 f1 = _mm256_cvtepi32_ps(_mm256_set_m128i(c3, c2));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(q + i), f0));
        acc1 = _mm256_add_ps(acc1,
                             _mm256_mul_ps(_mm256_loadu_ps(q + i + 8), f1));
    }
    return hsum256(_mm256_add_ps(acc0, acc1)) + dot4From(q, packed, i, n);
}

template <>
VSQ_TARGET_AVX2 inline float dot8<DotIsa::Avx2>(const float *q, const uint8_t *packed,
                                size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m128i bytes =
            _mm_loadl_epi64(reinterpret_cast<const __m128i *>(packed + i));
        const __m128i c0 = _mm_cvtepu8_epi32(bytes);
        const __m128i c1 = _mm_cvtepu8_epi32(_mm_srli_si128(bytes, 4));
        const __m256 f = _mm256_cvtepi32_ps(_mm256_set_m128i(c1, c0));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(q + i), f));
    }
    return hsum256(acc) + dot8From(q, packed, i, n);
}
#endif

}  // namespace detail

}  // namespace vsq::rabitq
