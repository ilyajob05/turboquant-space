#pragma once
// RaBitQ 1-bit distance with a quantized query (Gao & Long, SIGMOD 2024, §3.3).
//
// The data code is x_b in {0,1}^D (bit set => coordinate +1/sqrt(D)). The
// rotated unit query q' (D floats) is quantized once per query to B_q bits:
//
//   v_l = min_i q'_i,  Delta = (max_i q'_i - v_l) / (2^B_q - 1)
//   q_u[i] = floor((q'_i - v_l) / Delta + u_i),  u_i ~ U[0,1)   (unbiased rounding)
//   q_bar  = Delta * q_u + v_l
//
// and stored as B_q bit-planes of D/64 uint64 words. Then
//
//   <x_b, q_u>   = sum_j 2^j * popcount(x_b & plane_j)
//   <x_b, q_bar> = Delta <x_b, q_u> + v_l * popcount(x_b)
//   <o_bar, q'>  ~ (2 <x_b, q_bar> - sum_i q_bar_i) / sqrt(D)
//
// i.e. (B_q + 1) * D/64 popcounts per code instead of D float FMAs. The
// dither u_i comes from splitmix64 with a fixed seed, so a query is prepared
// identically on every call (results are reproducible).
//
// D must be a multiple of 64. Code bits are LSB-first in bytes, so a
// little-endian uint64 word w holds coordinates [64w, 64w + 64).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../common/config.h"
#include "../common/srht.h"

namespace vsq::rabitq {
namespace bitwise {

constexpr int kMaxQueryBits = 8;

// Scalars written in front of the bit-planes of a prepared query.
struct QueryHeader {
    float qnorm;   // ||q - c||
    float delta;   // Delta
    float vl;      // v_l
    float sum_qb;  // sum_i q_bar_i = Delta * sum_i q_u[i] + D * v_l
};

inline size_t planeWords(size_t D) { return D / 64; }

// Bytes of a prepared query: header + B_q planes.
inline size_t preparedBytes(size_t D, int qbits) {
    return sizeof(QueryHeader) + static_cast<size_t>(qbits) * planeWords(D) * sizeof(uint64_t);
}

// q: rotated unit query (D floats). out: preparedBytes(D, qbits) bytes.
inline void quantizeQueryBitplanes(const float *q, size_t D, int qbits, float qnorm,
                                   uint64_t seed, void *out) {
    float lo = q[0], hi = q[0];
    for (size_t i = 1; i < D; ++i) {
        lo = q[i] < lo ? q[i] : lo;
        hi = q[i] > hi ? q[i] : hi;
    }
    const uint32_t qmax = (1u << qbits) - 1u;
    const float delta = hi > lo ? (hi - lo) / static_cast<float>(qmax) : 1.0f;
    const float inv = 1.0f / delta;
    auto *bytes = static_cast<uint8_t *>(out);
    uint8_t *planes = bytes + sizeof(QueryHeader);
    const size_t W = planeWords(D);
    std::memset(planes, 0, static_cast<size_t>(qbits) * W * sizeof(uint64_t));
    common::RndGen64 rng(seed);
    uint64_t sum_qu = 0;
    for (size_t w = 0; w < W; ++w) {
        uint64_t word[kMaxQueryBits] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t b = 0; b < 64; ++b) {
            const size_t i = 64 * w + b;
            const float u = static_cast<float>(rng.next() >> 40) * 0x1.0p-24f;  // [0,1)
            float v = std::floor((q[i] - lo) * inv + u);
            v = v < 0.f ? 0.f : v;
            uint32_t qu = static_cast<uint32_t>(v);
            qu = qu > qmax ? qmax : qu;
            sum_qu += qu;
            for (int j = 0; j < qbits; ++j) word[j] |= static_cast<uint64_t>((qu >> j) & 1u) << b;
        }
        for (int j = 0; j < qbits; ++j)
            common::storeUnaligned<uint64_t>(planes + 8 * (static_cast<size_t>(j) * W + w), word[j]);
    }
    const QueryHeader h{qnorm, delta, lo,
                        delta * static_cast<float>(sum_qu) + static_cast<float>(D) * lo};
    std::memcpy(bytes, &h, sizeof(h));
}

// <x_b, q_u> (popcount-weighted) and popcount(x_b).
VSQ_ALWAYS_INLINE void bitwiseBody(const uint8_t *code, const uint8_t *planes, size_t W,
                                          int qbits, uint64_t *ip_out, uint64_t *pop_out) {
    uint64_t pop = 0;
    uint64_t acc[kMaxQueryBits] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t w = 0; w < W; ++w) {
        const uint64_t x = common::loadUnaligned<uint64_t>(code + 8 * w);
        pop += static_cast<uint64_t>(common::popcount64(x));
        for (int j = 0; j < qbits; ++j) {
            const uint64_t pl = common::loadUnaligned<uint64_t>(planes + 8 * (static_cast<size_t>(j) * W + w));
            acc[j] += static_cast<uint64_t>(common::popcount64(x & pl));
        }
    }
    uint64_t ip = 0;
    for (int j = 0; j < qbits; ++j) ip += acc[j] << j;
    *ip_out = ip;
    *pop_out = pop;
}

inline void bitwiseScalar(const uint8_t *code, const uint8_t *planes, size_t W, int qbits,
                          uint64_t *ip, uint64_t *pop) {
    bitwiseBody(code, planes, W, qbits, ip, pop);
}

#if defined(VSQ_HAVE_AVX2_KERNELS)
// Same body with the popcnt instruction enabled (the x86-64 baseline lacks it).
VSQ_TARGET_AVX2 inline void bitwiseAvx2(const uint8_t *code, const uint8_t *planes,
                                               size_t W, int qbits, uint64_t *ip,
                                               uint64_t *pop) {
    bitwiseBody(code, planes, W, qbits, ip, pop);
}
#endif

using BitwiseFn = void (*)(const uint8_t *, const uint8_t *, size_t, int, uint64_t *, uint64_t *);

inline BitwiseFn selectBitwise(common::Isa isa) {
#if defined(VSQ_HAVE_AVX2_KERNELS)
    if (isa == common::Isa::Avx2) return &bitwiseAvx2;
#endif
    (void)isa;
    return &bitwiseScalar;
}

// <o_bar, q'> estimate from the popcounts. D = padded dimension.
VSQ_ALWAYS_INLINE float estimateDot(const QueryHeader &h, uint64_t ip, uint64_t pop,
                                           float inv_sqrt_d) {
    const float xq = h.delta * static_cast<float>(ip) + h.vl * static_cast<float>(pop);
    return (2.0f * xq - h.sum_qb) * inv_sqrt_d;
}

}  // namespace bitwise
}  // namespace vsq::rabitq
