#pragma once

// RaBitQ (Gao & Long, SIGMOD 2024) and Extended RaBitQ
// (Gao, Gou, Xu, Yang, Liu, Long; arXiv:2409.09913). Header-only.
// Rotation is srht.h, included beside this file. A copy into HNSWLIB
// takes both headers and keeps them in the same directory.
// Asymmetric squared L2. Not a code-to-code score, so it must not be used
// as the HNSW link-construction metric.
//
// bits is 1, 4, or 8. D = padded_dim = roundUpPow2(max(input_dim, 4)).
// Coordinates [input_dim, D) are 0 before the rotation. The rotation is an
// SRHT (srht.h): splitmix64 signs, then a Walsh-Hadamard scaled by 1/sqrt(D).
//
// Slot, little-endian. The two float32 tails are copied with memcpy because
// the payload length is not always a multiple of 4.
//   bits 1: uint8 signs[(D + 7) / 8]
//           bit i set => coordinate +1/sqrt(D), else -1/sqrt(D)
//           This layout, including dot_factor's scale, is the original
//           1-bit code. Do not retarget it at the ±1/2 grid below.
//   bits 4: uint8 nibbles[D / 2]
//           low nibble = even index, high nibble = odd index, value 0..15
//   bits 8: uint8 codes[D], value 0..255
//   float32 norm_to_centroid     ||x - c|| in the original coordinates
//   float32 dot_factor           <y, o'>
//           1-bit: y_i = ±1/sqrt(D)
//           4/8-bit: y_i = code_i - (2^bits - 1) / 2
//                    (the centered grid of eq. 7; ||y|| cancels in the ratio
//                    and is not stored)
//
// Prepared query, (D + 2) float32:
//   rotated unit residual o' or q' [D], ||q - c||, sum of the rotated units.
//   1-bit distance reads only the norm. 4/8-bit distance uses the sum so the
//   inner product is <code, q'> - center * sum(q'), which is eq. 12.
//
// get_dist_func() is one kernel, chosen in the constructor from `bits` and
// the ISA of this translation unit: AVX2, else NEON, else scalar. Every
// kernel returns the same squared L2. Only the inner-product sum differs.
// distancePreparedScalar keeps the scalar sum callable for tests.
// 4-bit SIMD splits nibbles the same way as TurboQuant (low = even index)
// and widens the RaBitQ index. It does not use Lloyd-Max centroids or the
// QJL sign bit.
//
// 4/8-bit encode_mode. The constructor default is -1, resolved from bits:
// 1-bit stays algorithm1; 4-bit and 8-bit use fixed_scale.
//   0 algorithm1      sort every threshold. Bit-exact Extended RaBitQ
//                     Algorithm 1. Pass 0 to keep that code at 4 or 8 bits.
//   1 fixed_scale     one scale t frozen for the whole space, then O(1)
//                     per coordinate. t is the mean Algorithm 1 plateau
//                     edge over 100 N(0,1) residuals at seed 42 (not
//                     rot_seed). This is the 4/8-bit default. At 8 bits
//                     and dim 128 that single t does not concentrate, so
//                     recall@10 can drop by more than 0.01 versus the sweep.
//   2 windowed_scale  one scale per vector, inside the RaBitQ-Library
//                     tight interval, chosen by a min-heap of the next
//                     magnitude event. This is not the library interval
//                     pruner. No inner thread pool: encode uses only
//                     automatic storage, so one space may be called from
//                     several outer threads.
// fixed_scale and windowed_scale require bits 4 or 8. Passing either with
// bits 1 throws. t for fixed_scale is a positive finite double, stored
// once; the other modes leave it at 0.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "srht.h"

#if defined(__aarch64__) || defined(__ARM_NEON)
#  define TURBOQUANT_RABITQ_NEON 1
#  include <arm_neon.h>
#endif

#if defined(__AVX2__)
#  define TURBOQUANT_RABITQ_AVX2 1
#  include <immintrin.h>
#endif

// RaBitQ is not nested in turboquant. A nested scope would also see
// SRHT names, and TurboQuant names in a translation unit that includes
// both spaces. Shared rotation is called as turboquant::.
// turboquant::RaBitQSpace is an alias of rabitq::RaBitQSpace.
namespace rabitq {
namespace detail {

// Which inner-product kernel a distance function instantiates.
// The constructor picks one ISA for the whole translation unit.
enum class DotIsa { Scalar, Neon, Avx2 };

// Bit i set => coordinate contributes +q_i. Bit clear => -q_i.
// Row b, lane k is 0 when bit k of b is set, else the float sign bit,
// so XOR with the query flips the sign only for a clear bit.
// Each row is 32 bytes and 32-byte aligned for an AVX2 aligned load.
inline const uint32_t *signXorMask(unsigned bits) {
#if defined(TURBOQUANT_RABITQ_NEON) || defined(TURBOQUANT_RABITQ_AVX2)
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

// Stable 8-bit LSD radix. Used for (threshold, coordinate) keys.
inline void radixSortU64(std::vector<uint64_t> &keys) {
    if (keys.size() < 2)
        return;
    std::vector<uint64_t> scratch(keys.size());
    for (int shift = 0; shift < 64; shift += 8) {
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

#if defined(TURBOQUANT_RABITQ_NEON)
template <>
inline float dot1<DotIsa::Neon>(const float *q, const uint8_t *signs,
                                size_t n, float inv_sqrt_d) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    const float32x4_t scale = vdupq_n_f32(inv_sqrt_d);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint32_t *mask = signXorMask(signs[i >> 3]);
        float32x4_t q0 = vld1q_f32(q + i);
        float32x4_t q1 = vld1q_f32(q + i + 4);
        q0 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q0),
                                             vld1q_u32(mask)));
        q1 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q1),
                                             vld1q_u32(mask + 4)));
        acc0 = vaddq_f32(acc0, vmulq_f32(q0, scale));
        acc1 = vaddq_f32(acc1, vmulq_f32(q1, scale));
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1)) +
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

#if defined(TURBOQUANT_RABITQ_AVX2)
inline float hsum256(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    const __m128 s4 = _mm_add_ps(lo, hi);
    const __m128 dup = _mm_movehdup_ps(s4);
    const __m128 sums = _mm_add_ps(s4, dup);
    const __m128 high = _mm_movehl_ps(sums, sums);
    return _mm_cvtss_f32(_mm_add_ss(sums, high));
}

template <>
inline float dot1<DotIsa::Avx2>(const float *q, const uint8_t *signs,
                                size_t n, float inv_sqrt_d) {
    __m256 acc = _mm256_setzero_ps();
    const __m256 scale = _mm256_set1_ps(inv_sqrt_d);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i mask = _mm256_load_si256(
            reinterpret_cast<const __m256i *>(signXorMask(signs[i >> 3])));
        __m256 qv = _mm256_loadu_ps(q + i);
        qv = _mm256_xor_ps(qv, _mm256_castsi256_ps(mask));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(qv, scale));
    }
    return hsum256(acc) + dot1From(q, signs, i, n, inv_sqrt_d);
}

// 16 coordinates per iteration. unpacklo(low nibble, high nibble) is the
// same even/odd order as NEON vzip_u8.
template <>
inline float dot4<DotIsa::Avx2>(const float *q, const uint8_t *packed,
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
inline float dot8<DotIsa::Avx2>(const float *q, const uint8_t *packed,
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

// One space: fixed centroid, fixed rotation seed, fixed bit width, fixed D.
class RaBitQSpace {
public:
    // centroid == nullptr means the zero vector of length `dim`.
    // The pointer is copied; it is not retained.
    // bits is appended so existing (dim, seed, centroid) calls stay 1-bit.
    // encode_mode -1/0/1/2: default, algorithm1, fixed_scale, windowed_scale.
    // -1 resolves from bits: 1 -> algorithm1, 4 or 8 -> fixed_scale.
    RaBitQSpace(size_t dim, uint64_t rot_seed, const float *centroid,
                int bits = 1, int encode_mode = -1)
        : dim_(dim),
          padded_(turboquant::roundUpPow2AtLeast4(dim)),
          bits_(bits),
          rot_seed_(rot_seed),
          centroid_(dim, 0.0f),
          inv_sqrt_d_(1.0f / std::sqrt(static_cast<float>(padded_))),
          center_(bits >= 8 ? 127.5f : (bits >= 4 ? 7.5f : 0.5f)),
          encode_mode_(encode_mode),
          t_fixed_(0.0) {
        if (dim_ == 0)
            throw std::invalid_argument("RaBitQ: dim must be positive");
        if (bits_ != 1 && bits_ != 4 && bits_ != 8)
            throw std::invalid_argument("RaBitQ: bits must be 1, 4, or 8");
        if (encode_mode_ < -1 || encode_mode_ > 2)
            throw std::invalid_argument(
                "RaBitQ: encode_mode must be -1, 0, 1, or 2");
        if (encode_mode_ < 0)
            encode_mode_ = bits_ == 1 ? 0 : 1;
        if (encode_mode_ != 0 && bits_ == 1)
            throw std::invalid_argument(
                "RaBitQ: fixed_scale and windowed_scale require bits 4 or 8");
        if (centroid != nullptr) {
            for (size_t i = 0; i < dim_; ++i)
                centroid_[i] = centroid[i];
        }
        signs_ = turboquant::generateSigns(padded_, rot_seed_);
        dist_func_ = selectDist(bits_);
        if (encode_mode_ == 1)
            t_fixed_ = calibrateFixedScale();
    }

    size_t dim() const { return dim_; }
    size_t paddedDim() const { return padded_; }
    int bits() const { return bits_; }
    uint64_t rotSeed() const { return rot_seed_; }
    // "algorithm1", "fixed_scale", or "windowed_scale".
    const char *encodeModeName() const {
        if (encode_mode_ == 1)
            return "fixed_scale";
        if (encode_mode_ == 2)
            return "windowed_scale";
        return "algorithm1";
    }
    // Frozen scale of fixed_scale. 0 for algorithm1 and windowed_scale.
    double fixedScale() const { return t_fixed_; }

    // Bytes of one data slot.
    size_t codeSizeBytes() const { return payloadBytes() + 2 * sizeof(float); }

    // Bytes of one prepared query (rotated unit, norm, sum of rotated units).
    size_t querySizeBytes() const { return (padded_ + 2) * sizeof(float); }

    // HNSWLIB SpaceInterface shape. These three are non-virtual on purpose:
    // the header does not include hnswlib. A copy placed next to space_l2.h
    // can inherit SpaceInterface<float> and forward to them.
    // get_dist_func() is the kernel selected for this bits/ISA pair.
    size_t get_data_size() { return codeSizeBytes(); }

    using DistFunc = float (*)(const void *, const void *, const void *);

    DistFunc get_dist_func() { return dist_func_; }

    void *get_dist_func_param() { return this; }

    // Writes one slot. Throws std::invalid_argument if ||x - c|| == 0 or
    // the estimator denominator is 0. The message includes D.
    void encode(const float *x, void *slot) const {
        if (x == nullptr || slot == nullptr)
            throw std::invalid_argument("RaBitQ encode: null pointer");
        std::vector<float> rotated(padded_, 0.0f);
        float norm = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            const float v = x[i] - centroid_[i];
            rotated[i] = v;
            norm += v * v;
        }
        norm = std::sqrt(norm);
        if (!(norm > 0.0f)) {
            throw std::invalid_argument(
                "RaBitQ encode: ||x - c|| is 0, padded_dim=" +
                std::to_string(padded_));
        }
        const float inv = 1.0f / norm;
        for (size_t i = 0; i < dim_; ++i)
            rotated[i] *= inv;
        rotateUnit(rotated.data());

        auto *bytes = static_cast<uint8_t *>(slot);
        if (bits_ == 1) {
            const float dot = dotWithCube(rotated.data());
            if (!(dot > 0.0f) && !(dot < 0.0f)) {
                throw std::invalid_argument(
                    "RaBitQ encode: dot_factor is 0, padded_dim=" +
                    std::to_string(padded_));
            }
            std::memset(bytes, 0, payloadBytes());
            for (size_t i = 0; i < padded_; ++i) {
                if (rotated[i] >= 0.0f)
                    bytes[i >> 3] = static_cast<uint8_t>(
                        bytes[i >> 3] | (1u << (i & 7u)));
            }
            std::memcpy(bytes + payloadBytes(), &norm, sizeof(float));
            std::memcpy(bytes + payloadBytes() + sizeof(float), &dot,
                        sizeof(float));
            return;
        }

        std::vector<uint8_t> codes(padded_);
        float dot = 0.0f;
        if (encode_mode_ == 1)
            dot = quantizeFixedScale(rotated.data(), codes.data());
        else if (encode_mode_ == 2)
            dot = quantizeWindowedScale(rotated.data(), codes.data());
        else
            dot = quantizeExtended(rotated.data(), codes.data());
        packCodes(codes.data(), norm, dot, bytes);
    }

    // Row-major [n, dim] into n packed slots. One serial pass, same encode
    // as the single-vector call. No inner thread pool: the caller is already
    // threaded, and a throw stays on this thread.
    void encodeBatch(const float *raws, size_t n, void *out) const {
        if (out == nullptr || (n > 0 && raws == nullptr))
            throw std::invalid_argument("RaBitQ encodeBatch: null pointer");
        auto *bytes = static_cast<uint8_t *>(out);
        const size_t stride = codeSizeBytes();
        for (size_t i = 0; i < n; ++i)
            encode(raws + i * dim_, bytes + i * stride);
    }

    // `out` must hold querySizeBytes(). q has length dim().
    void prepareQuery(const float *q, void *out) const {
        if (q == nullptr || out == nullptr)
            throw std::invalid_argument("RaBitQ prepareQuery: null pointer");
        std::vector<float> rotated(padded_, 0.0f);
        float norm = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            const float v = q[i] - centroid_[i];
            rotated[i] = v;
            norm += v * v;
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            const float inv = 1.0f / norm;
            for (size_t i = 0; i < dim_; ++i)
                rotated[i] *= inv;
            rotateUnit(rotated.data());
        }
        float sum_q = 0.0f;
        for (size_t i = 0; i < padded_; ++i)
            sum_q += rotated[i];
        auto *dst = static_cast<float *>(out);
        std::memcpy(dst, rotated.data(), padded_ * sizeof(float));
        std::memcpy(dst + padded_, &norm, sizeof(float));
        std::memcpy(dst + padded_ + 1, &sum_q, sizeof(float));
    }

    // First pointer: prepared query. Second: code slot. Third: this space.
    float distancePrepared(const void *prepared, const void *slot) const {
        return dist_func_(prepared, slot, this);
    }

    // Raw query of length dim(). Uses get_dist_func(), so the HNSW pointer
    // and this path cannot drift.
    float distanceRaw(const float *q, const void *slot) const {
        std::vector<float> prepared(querySizeBytes() / sizeof(float), 0.0f);
        prepareQuery(q, prepared.data());
        return get_dist_func_const()(prepared.data(), slot, this);
    }

    // Same epilogue as get_dist_func(), accumulated in scalar order.
    float distancePreparedScalar(const void *prepared, const void *slot) const {
        if (bits_ == 1)
            return distKernel<1, detail::DotIsa::Scalar>(prepared, slot,
                                                               this);
        if (bits_ == 4)
            return distKernel<4, detail::DotIsa::Scalar>(prepared, slot,
                                                               this);
        return distKernel<8, detail::DotIsa::Scalar>(prepared, slot, this);
    }

    float distanceRawScalar(const float *q, const void *slot) const {
        std::vector<float> prepared(querySizeBytes() / sizeof(float), 0.0f);
        prepareQuery(q, prepared.data());
        return distancePreparedScalar(prepared.data(), slot);
    }

    // 1-to-N asymmetric search. `query` has length dim(). `codes` is n slots
    // packed back to back, each codeSizeBytes() long. `out` has length n.
    // The query is rotated once; each slot then uses the selected kernel.
    void distanceBatch1ToN(const float *query, const void *codes, size_t n,
                           float *out) const {
        if (query == nullptr || out == nullptr)
            throw std::invalid_argument("RaBitQ distanceBatch1ToN: null pointer");
        if (n > 0 && codes == nullptr)
            throw std::invalid_argument("RaBitQ distanceBatch1ToN: null codes");
        std::vector<float> prepared(querySizeBytes() / sizeof(float));
        prepareQuery(query, prepared.data());
        const auto *base = static_cast<const char *>(codes);
        const size_t stride = codeSizeBytes();
        const DistFunc fn = dist_func_;
        for (size_t i = 0; i < n; ++i)
            out[i] = fn(prepared.data(), base + i * stride, this);
    }

    // M-to-N asymmetric search. `queries` is row-major [m, dim()].
    // `codes` is n packed slots. `out` is row-major [m, n].
    // Each query is rotated once, then dotted with every slot.
    void distanceBatchMToN(const float *queries, size_t m, const void *codes,
                           size_t n, float *out) const {
        if (out == nullptr)
            throw std::invalid_argument("RaBitQ distanceBatchMToN: null out");
        if (m > 0 && queries == nullptr)
            throw std::invalid_argument("RaBitQ distanceBatchMToN: null queries");
        if (n > 0 && codes == nullptr)
            throw std::invalid_argument("RaBitQ distanceBatchMToN: null codes");
        std::vector<float> prepared(querySizeBytes() / sizeof(float));
        const auto *base = static_cast<const char *>(codes);
        const size_t stride = codeSizeBytes();
        const DistFunc fn = dist_func_;
        for (size_t qi = 0; qi < m; ++qi) {
            prepareQuery(queries + qi * dim_, prepared.data());
            float *row = out + qi * n;
            for (size_t i = 0; i < n; ++i)
                row[i] = fn(prepared.data(), base + i * stride, this);
        }
    }

    // "1-neon", "4-avx2", "8-scalar", and the other bits/ISA pairs.
    const char *distanceKernel() const {
#if defined(TURBOQUANT_RABITQ_AVX2)
        if (bits_ == 1)
            return "1-avx2";
        if (bits_ == 4)
            return "4-avx2";
        return "8-avx2";
#elif defined(TURBOQUANT_RABITQ_NEON)
        if (bits_ == 1)
            return "1-neon";
        if (bits_ == 4)
            return "4-neon";
        return "8-neon";
#else
        if (bits_ == 1)
            return "1-scalar";
        if (bits_ == 4)
            return "4-scalar";
        return "8-scalar";
#endif
    }

private:
    size_t payloadBytes() const {
        if (bits_ == 1)
            return (padded_ + 7) / 8;
        if (bits_ == 4)
            return padded_ / 2;
        return padded_;
    }

    void rotateUnit(float *unit) const {
        turboquant::randomizedHadamard(unit, signs_.data(), padded_);
    }

    // <ō, o> where o is the rotated unit vector and ō_i = ±1/sqrt(D).
    float dotWithCube(const float *rotated_unit) const {
        float acc = 0.0f;
        for (size_t i = 0; i < padded_; ++i) {
            const float s = (rotated_unit[i] >= 0.0f) ? inv_sqrt_d_
                                                       : -inv_sqrt_d_;
            acc += s * rotated_unit[i];
        }
        return acc;
    }

    // Extended RaBitQ Algorithm 1. `rotated_unit` is o'. Writes one unsigned
    // grid index per coordinate and returns <y, o'> in float32.
    //
    // Grid coordinates are the half-integers
    //   -(2^B-1)/2, -(2^B-1)/2+1, ..., (2^B-1)/2.
    // As t → 0+ the nearest in-orthant point is +1/2 where o' >= 0 and
    // -1/2 where o' < 0. Later critical values move one step further into
    // that orthant. The initial rounding is scored: it is the nearest
    // neighbour on (0, t_first).
    float quantizeExtended(const float *rotated_unit, uint8_t *codes) const {
        const int levels = 1 << bits_;
        const int start_pos = levels >> 1;       // round(center) = 2^{B-1}
        const int start_neg = start_pos - 1;     // floor(center)
        const double center = static_cast<double>(center_);

        std::vector<int> code(padded_);
        double dot = 0.0;
        double normsq = 0.0;
        for (size_t i = 0; i < padded_; ++i) {
            const int c = (rotated_unit[i] >= 0.0f) ? start_pos : start_neg;
            code[i] = c;
            const double v = static_cast<double>(c) - center;
            const double oi = static_cast<double>(rotated_unit[i]);
            dot += v * oi;
            normsq += v * v;
        }
        const std::vector<int> initial = code;

        // One key per critical value: high 32 bits are the sortable
        // threshold, low 32 bits are the coordinate. Radix order matches
        // sorting by (t, index). Below a few thousand keys a comparison
        // sort is faster, so that path stays.
        const size_t steps_per_coord =
            static_cast<size_t>(std::max(start_pos - 1, 0));
        std::vector<uint64_t> keys;
        keys.reserve(padded_ * steps_per_coord);
        for (size_t i = 0; i < padded_; ++i) {
            const float oi = rotated_unit[i];
            if (oi > 0.0f) {
                for (int k = start_pos + 1; k <= levels - 1; ++k) {
                    const float t =
                        (static_cast<float>(k) - 0.5f - center_) / oi;
                    const uint64_t key =
                        (static_cast<uint64_t>(detail::sortableFloat(t))
                         << 32) |
                        static_cast<uint32_t>(i);
                    keys.push_back(key);
                }
            } else if (oi < 0.0f) {
                for (int m = start_neg - 1; m >= 0; --m) {
                    const float t =
                        (static_cast<float>(m) + 0.5f - center_) / oi;
                    const uint64_t key =
                        (static_cast<uint64_t>(detail::sortableFloat(t))
                         << 32) |
                        static_cast<uint32_t>(i);
                    keys.push_back(key);
                }
            }
        }
        if (keys.size() >= 4096)
            detail::radixSortU64(keys);
        else
            std::sort(keys.begin(), keys.end());

        // Squared cosine is monotone with the cosine for a positive dot,
        // and it avoids a sqrt in the comparison.
        auto consider = [](double d, double nsq) {
            return (d * d) / nsq;
        };
        double best_score = -1.0;
        long long best_step = -1;
        bool have = false;
        if (dot > 0.0 && normsq > 0.0) {
            best_score = consider(dot, normsq);
            have = true;
        }
        for (size_t s = 0; s < keys.size(); ++s) {
            const size_t i = static_cast<size_t>(keys[s] & 0xFFFFFFFFu);
            const int delta = (rotated_unit[i] > 0.0f) ? 1 : -1;
            const int neu = code[i] + delta;
            if (neu < 0 || neu >= levels) {
                throw std::logic_error(
                    "RaBitQ quantize: code left the grid, padded_dim=" +
                    std::to_string(padded_));
            }
            const double v = static_cast<double>(code[i]) - center;
            const double dv = static_cast<double>(delta);
            dot += dv * static_cast<double>(rotated_unit[i]);
            normsq += 2.0 * v * dv + dv * dv;
            code[i] = neu;
            if (dot > 0.0 && normsq > 0.0) {
                const double score = consider(dot, normsq);
                if (!have || score > best_score) {
                    best_score = score;
                    best_step = static_cast<long long>(s);
                    have = true;
                }
            }
        }
        if (!have) {
            throw std::invalid_argument(
                "RaBitQ encode: dot_factor is 0, padded_dim=" +
                std::to_string(padded_));
        }

        if (best_step >= 0) {
            code = initial;
            for (long long s = 0; s <= best_step; ++s) {
                const size_t i =
                    static_cast<size_t>(keys[static_cast<size_t>(s)] & 0xFFFFFFFFu);
                const int delta = (rotated_unit[i] > 0.0f) ? 1 : -1;
                code[i] += delta;
            }
        } else {
            code = initial;
        }

        float acc = 0.0f;
        for (size_t i = 0; i < padded_; ++i) {
            codes[i] = static_cast<uint8_t>(code[i]);
            acc += (static_cast<float>(code[i]) - center_) * rotated_unit[i];
        }
        if (!(acc > 0.0f) && !(acc < 0.0f)) {
            throw std::invalid_argument(
                "RaBitQ encode: dot_factor is 0, padded_dim=" +
                std::to_string(padded_));
        }
        return acc;
    }

    // Grid index from a positive scale. o >= 0 maps to [2^{B-1}, 2^B).
    // o < 0 maps to [0, 2^{B-1}). |code - center| = magnitude + 0.5.
    float emitGrid(const float *rotated_unit, uint8_t *codes, double t) const {
        const int levels = 1 << bits_;
        const int max_mag = (levels >> 1) - 1;
        const int start_pos = levels >> 1;
        const int start_neg = start_pos - 1;
        if (!(t > 0.0) || !std::isfinite(t))
            throw std::invalid_argument(
                "RaBitQ encode: scale is not positive, padded_dim=" +
                std::to_string(padded_));
        float acc = 0.0f;
        for (size_t i = 0; i < padded_; ++i) {
            const float oi = rotated_unit[i];
            const int mag = detail::magnitudeAtScale(
                std::fabs(static_cast<double>(oi)), t, max_mag);
            const int c = (oi >= 0.0f) ? start_pos + mag : start_neg - mag;
            if (c < 0 || c >= levels)
                throw std::logic_error(
                    "RaBitQ quantize: code left the grid, padded_dim=" +
                    std::to_string(padded_));
            codes[i] = static_cast<uint8_t>(c);
            acc += (static_cast<float>(c) - center_) * oi;
        }
        if (!(acc > 0.0f) && !(acc < 0.0f))
            throw std::invalid_argument(
                "RaBitQ encode: dot_factor is 0, padded_dim=" +
                std::to_string(padded_));
        return acc;
    }

    float quantizeFixedScale(const float *rotated_unit, uint8_t *codes) const {
        return emitGrid(rotated_unit, codes, t_fixed_);
    }

    // Left edge of the Algorithm 1 plateau that produced `codes`.
    // The last applied threshold is the max t among coordinates that left
    // the initial rounding. If none did, the edge is half the next threshold.
    double scaleOfCodes(const float *rotated_unit,
                        const uint8_t *codes) const {
        const int levels = 1 << bits_;
        const int start_pos = levels >> 1;
        const int start_neg = start_pos - 1;
        float max_t = 0.0f;
        bool any = false;
        float next_t = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < padded_; ++i) {
            const float oi = rotated_unit[i];
            const int c = static_cast<int>(codes[i]);
            if (oi > 0.0f) {
                if (c < start_pos || c >= levels)
                    throw std::logic_error(
                        "RaBitQ fixed scale: positive code left the ray");
                if (c > start_pos) {
                    const float t =
                        (static_cast<float>(c) - 0.5f - center_) / oi;
                    if (!any || t > max_t)
                        max_t = t;
                    any = true;
                }
                if (c + 1 < levels) {
                    const float t =
                        (static_cast<float>(c + 1) - 0.5f - center_) / oi;
                    if (t < next_t)
                        next_t = t;
                }
            } else if (oi < 0.0f) {
                if (c < 0 || c > start_neg)
                    throw std::logic_error(
                        "RaBitQ fixed scale: negative code left the ray");
                if (c < start_neg) {
                    const float t =
                        (static_cast<float>(c) + 0.5f - center_) / oi;
                    if (!any || t > max_t)
                        max_t = t;
                    any = true;
                }
                if (c > 0) {
                    const float t =
                        (static_cast<float>(c - 1) + 0.5f - center_) / oi;
                    if (t < next_t)
                        next_t = t;
                }
            }
        }
        if (any)
            return static_cast<double>(max_t);
        if (std::isfinite(next_t) && next_t > 0.0f)
            return 0.5 * static_cast<double>(next_t);
        throw std::invalid_argument(
            "RaBitQ fixed scale: no threshold, padded_dim=" +
            std::to_string(padded_));
    }

    // Mean Algorithm 1 scale on 100 standard-normal residuals. Seed 42 is
    // the library's calibration seed and does not depend on rot_seed.
    // The residual is padded, normalized, and rotated the same way encode is.
    double calibrateFixedScale() const {
        constexpr int kCount = 100;
        std::mt19937_64 rng(42);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> rotated(padded_, 0.0f);
        std::vector<uint8_t> codes(padded_);
        double sum = 0.0;
        int used = 0;
        for (int n = 0; n < kCount; ++n) {
            std::fill(rotated.begin(), rotated.end(), 0.0f);
            float norm = 0.0f;
            for (size_t i = 0; i < dim_; ++i) {
                const float v = normal(rng);
                rotated[i] = v;
                norm += v * v;
            }
            norm = std::sqrt(norm);
            if (!(norm > 0.0f))
                continue;
            const float inv = 1.0f / norm;
            for (size_t i = 0; i < dim_; ++i)
                rotated[i] *= inv;
            rotateUnit(rotated.data());
            try {
                quantizeExtended(rotated.data(), codes.data());
            } catch (const std::invalid_argument &) {
                continue;
            }
            sum += scaleOfCodes(rotated.data(), codes.data());
            ++used;
        }
        if (used == 0)
            throw std::invalid_argument(
                "RaBitQ fixed scale: calibration produced no vector, padded_dim=" +
                std::to_string(padded_));
        const double t = sum / static_cast<double>(used);
        if (!(t > 0.0) || !std::isfinite(t))
            throw std::invalid_argument(
                "RaBitQ fixed scale: t_fixed is not positive, padded_dim=" +
                std::to_string(padded_));
        return t;
    }

    // windowed_scale: min-heap of the next magnitude step inside [t_start, t_end).
    // Score is N/sqrt(S) with q_i = magnitude_i + 0.5, the same argmax as
    // Algorithm 1's squared cosine. One live event per coordinate.
    double bestWindowedScale(const float *rotated_unit) const {
        const int ex_bits = bits_ - 1;
        const int max_code = (1 << ex_bits) - 1;
        double max_o = 0.0;
        for (size_t i = 0; i < padded_; ++i)
            max_o = std::max(max_o, std::fabs(static_cast<double>(rotated_unit[i])));
        if (!(max_o > 0.0))
            throw std::invalid_argument(
                "RaBitQ encode: rotated residual is 0, padded_dim=" +
                std::to_string(padded_));
        const double t_end =
            static_cast<double>(max_code + 10) / max_o;
        const double t_start =
            t_end * static_cast<double>(detail::tightStart(ex_bits));

        using Event = std::pair<double, size_t>;
        std::vector<Event> next_t;
        next_t.reserve(padded_);
        std::vector<int> cur(padded_);
        double sqr = static_cast<double>(padded_) * 0.25;
        double num = 0.0;
        for (size_t i = 0; i < padded_; ++i) {
            const double magnitude =
                std::fabs(static_cast<double>(rotated_unit[i]));
            const int c = detail::magnitudeAtScale(
                magnitude, t_start, max_code);
            cur[i] = c;
            sqr += static_cast<double>(c) * static_cast<double>(c) + c;
            num += (static_cast<double>(c) + 0.5) * magnitude;
            if (magnitude > 0.0 && c < max_code) {
                const double nxt = static_cast<double>(c + 1) / magnitude;
                if (nxt < t_end)
                    next_t.emplace_back(nxt, i);
            }
        }
        std::make_heap(next_t.begin(), next_t.end(), std::greater<Event>{});
        double max_ip = num / std::sqrt(sqr);
        double best_t = t_start;
        while (!next_t.empty()) {
            const double cur_t = next_t.front().first;
            do {
                const size_t i = next_t.front().second;
                ++cur[i];
                sqr += 2.0 * static_cast<double>(cur[i]);
                const double magnitude =
                    std::fabs(static_cast<double>(rotated_unit[i]));
                num += magnitude;
                Event next{t_end, i};
                if (cur[i] < max_code)
                    next.first = static_cast<double>(cur[i] + 1) / magnitude;
                if (next.first >= t_end) {
                    next = next_t.back();
                    next_t.pop_back();
                }
                if (!next_t.empty()) {
                    size_t parent = 0;
                    size_t child = 1;
                    while (child < next_t.size()) {
                        if (child + 1 < next_t.size() &&
                            next_t[child + 1] < next_t[child])
                            ++child;
                        if (!(next_t[child] < next))
                            break;
                        next_t[parent] = next_t[child];
                        parent = child;
                        child = 2 * parent + 1;
                    }
                    next_t[parent] = next;
                }
            } while (!next_t.empty() && next_t.front().first == cur_t);
            const double cur_ip = num / std::sqrt(sqr);
            if (cur_ip > max_ip) {
                max_ip = cur_ip;
                best_t = cur_t;
            }
        }
        return best_t;
    }

    float quantizeWindowedScale(const float *rotated_unit, uint8_t *codes) const {
        return emitGrid(rotated_unit, codes, bestWindowedScale(rotated_unit));
    }

    void packCodes(const uint8_t *codes, float norm, float dot,
                   uint8_t *slot) const {
        const size_t n = payloadBytes();
        std::memset(slot, 0, n);
        if (bits_ == 8) {
            std::memcpy(slot, codes, padded_);
        } else {
            for (size_t i = 0; i < padded_; ++i) {
                const unsigned shift = (i & 1u) ? 4u : 0u;
                slot[i >> 1] = static_cast<uint8_t>(slot[i >> 1] |
                                                     (codes[i] << shift));
            }
        }
        std::memcpy(slot + n, &norm, sizeof(float));
        std::memcpy(slot + n + sizeof(float), &dot, sizeof(float));
    }

    template <int Bits, detail::DotIsa Isa>
    static float distKernel(const void *prepared, const void *slot,
                            const void *param) {
        const auto *self = static_cast<const RaBitQSpace *>(param);
        if (prepared == nullptr || slot == nullptr)
            throw std::invalid_argument("RaBitQ distance: null pointer");
        assert(self->bits_ == Bits);
        const auto *qrot = static_cast<const float *>(prepared);
        float qnorm = 0.0f;
        std::memcpy(&qnorm, qrot + self->padded_, sizeof(float));

        const auto *bytes = static_cast<const uint8_t *>(slot);
        const size_t payload = self->payloadBytes();
        float xnorm = 0.0f;
        float dot_factor = 0.0f;
        std::memcpy(&xnorm, bytes + payload, sizeof(float));
        std::memcpy(&dot_factor, bytes + payload + sizeof(float),
                    sizeof(float));
        if (!(dot_factor > 0.0f) && !(dot_factor < 0.0f)) {
            throw std::invalid_argument(
                "RaBitQ distance: dot_factor is 0, padded_dim=" +
                std::to_string(self->padded_));
        }

        float ip = 0.0f;
        if (qnorm > 0.0f) {
            if constexpr (Bits == 1) {
                const float acc = detail::dot1<Isa>(
                    qrot, bytes, self->padded_, self->inv_sqrt_d_);
                ip = acc / dot_factor;
            } else {
                float sum_q = 0.0f;
                std::memcpy(&sum_q, qrot + self->padded_ + 1, sizeof(float));
                const float acc =
                    (Bits == 4)
                        ? detail::dot4<Isa>(qrot, bytes, self->padded_)
                        : detail::dot8<Isa>(qrot, bytes, self->padded_);
                ip = (acc - self->center_ * sum_q) / dot_factor;
            }
        }
        return xnorm * xnorm + qnorm * qnorm - 2.0f * xnorm * qnorm * ip;
    }

    static DistFunc selectDist(int bits) {
#if defined(TURBOQUANT_RABITQ_AVX2)
        if (bits == 1)
            return &RaBitQSpace::distKernel<1, detail::DotIsa::Avx2>;
        if (bits == 4)
            return &RaBitQSpace::distKernel<4, detail::DotIsa::Avx2>;
        return &RaBitQSpace::distKernel<8, detail::DotIsa::Avx2>;
#elif defined(TURBOQUANT_RABITQ_NEON)
        if (bits == 1)
            return &RaBitQSpace::distKernel<1, detail::DotIsa::Neon>;
        if (bits == 4)
            return &RaBitQSpace::distKernel<4, detail::DotIsa::Neon>;
        return &RaBitQSpace::distKernel<8, detail::DotIsa::Neon>;
#else
        if (bits == 1)
            return &RaBitQSpace::distKernel<1, detail::DotIsa::Scalar>;
        if (bits == 4)
            return &RaBitQSpace::distKernel<4, detail::DotIsa::Scalar>;
        return &RaBitQSpace::distKernel<8, detail::DotIsa::Scalar>;
#endif
    }

    DistFunc get_dist_func_const() const { return dist_func_; }

    size_t dim_;
    size_t padded_;
    int bits_;
    uint64_t rot_seed_;
    std::vector<float> signs_;
    std::vector<float> centroid_;
    float inv_sqrt_d_;
    float center_;
    int encode_mode_;
    double t_fixed_;
    DistFunc dist_func_ = nullptr;
};

}  // namespace rabitq

namespace turboquant {
using RaBitQSpace = rabitq::RaBitQSpace;
}  // namespace turboquant
