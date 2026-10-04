#pragma once
// AVX2 + FMA kernels for TurboQuant v2 payloads (runtime-dispatched).
//
// Same four kernels as kernels_scalar.h. Decoders turn 8 coordinates into one
// __m256 without gathers where possible:
//   Nibble  16-entry float table held in two YMM registers; two
//           vpermps lookups + one blend on bit 3 of the index
//   Byte    256-entry table: eight scalar loads (L1-resident, 1 KiB)
//   Half    vcvtph2ps (F16C)
// D is a multiple of 64; loops step 16 coordinates with two accumulators.

#include <cstddef>
#include <cstdint>

#include "../common/config.h"

#if defined(VSQ_HAVE_AVX2_KERNELS)

namespace vsq::turboquant {
namespace kernels {
namespace avx2 {

struct Nibble {
    __m256 lo, hi;  // table[0..8), table[8..16)
    VSQ_AVX2_INLINE explicit Nibble(const float *t)
        : lo(_mm256_loadu_ps(t)), hi(_mm256_loadu_ps(t + 8)) {}
    // Coordinates i..i+7 (i % 8 == 0): 4 payload bytes, low nibble first.
    VSQ_AVX2_INLINE __m256 load8(const uint8_t *p, size_t i) const {
        const __m128i b = _mm_cvtsi32_si128(common::loadUnaligned<int32_t>(p + (i >> 1)));
        const __m128i m = _mm_set1_epi8(0x0F);
        const __m128i n_lo = _mm_and_si128(b, m);
        const __m128i n_hi = _mm_and_si128(_mm_srli_epi16(b, 4), m);
        const __m256i idx = _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(n_lo, n_hi));
        const __m256 from_lo = _mm256_permutevar8x32_ps(lo, idx);  // uses idx & 7
        const __m256 from_hi = _mm256_permutevar8x32_ps(hi, idx);
        // blendv picks by the sign bit: move index bit 3 there.
        return _mm256_blendv_ps(from_lo, from_hi,
                                _mm256_castsi256_ps(_mm256_slli_epi32(idx, 28)));
    }
};

// 256-entry table (1 KiB, L1-resident): eight scalar loads, which cost about
// the same as vgatherdps on Intel and are faster on pre-Zen4 AMD. (Wrong gather
// lanes once seen under qemu-user 7.2 were a qemu bug — VSIB index ymm4 decoded
// as "no index" — not a defect of the gather variant; see docker/test_amd64.sh.)
struct Byte {
    const float *t;
    VSQ_AVX2_INLINE explicit Byte(const float *table) : t(table) {}
    VSQ_AVX2_INLINE __m256 load8(const uint8_t *p, size_t i) const {
        const uint8_t *c = p + i;
        return _mm256_setr_ps(t[c[0]], t[c[1]], t[c[2]], t[c[3]], t[c[4]], t[c[5]], t[c[6]], t[c[7]]);
    }
};

struct Half {
    VSQ_AVX2_INLINE explicit Half(const float *) {}
    VSQ_AVX2_INLINE __m256 load8(const uint8_t *p, size_t i) const {
        return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 2 * i)));
    }
};

VSQ_AVX2_INLINE float hsum(__m256 v) {
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 h = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(h, _mm_movehdup_ps(h)));
}

template <class Dec>
VSQ_TARGET_AVX2 float dot1(const float *q, const uint8_t *p, const float *table,
                                  size_t D) {
    const Dec d(table);
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    for (size_t i = 0; i < D; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), d.load8(p, i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i + 8), d.load8(p, i + 8), s1);
    }
    return hsum(_mm256_add_ps(s0, s1));
}

template <class Dec>
VSQ_TARGET_AVX2 void dot2(const float *q1, const float *q2, const uint8_t *p,
                                 const float *tA, const float *tB, size_t D, float *out) {
    const Dec a(tA), b(tB);
    __m256 s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps();
    for (size_t i = 0; i < D; i += 8) {
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + i), a.load8(p, i), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(q2 + i), b.load8(p, i), s2);
    }
    out[0] = hsum(s1);
    out[1] = hsum(s2);
}

template <class Dec>
VSQ_TARGET_AVX2 float sym(const uint8_t *a, const uint8_t *b, const float *table,
                                 size_t D) {
    const Dec d(table);
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    for (size_t i = 0; i < D; i += 16) {
        s0 = _mm256_fmadd_ps(d.load8(a, i), d.load8(b, i), s0);
        s1 = _mm256_fmadd_ps(d.load8(a, i + 8), d.load8(b, i + 8), s1);
    }
    return hsum(_mm256_add_ps(s0, s1));
}

template <class Dec>
VSQ_TARGET_AVX2 void symCross(const uint8_t *a, const uint8_t *b, const float *ya,
                                     const float *yb, const float *table, size_t D,
                                     float *out) {
    const Dec d(table);
    __m256 sab = _mm256_setzero_ps(), sba = _mm256_setzero_ps(), sab2 = _mm256_setzero_ps();
    for (size_t i = 0; i < D; i += 8) {
        const __m256 da = d.load8(a, i), db = d.load8(b, i);
        sab = _mm256_fmadd_ps(da, db, sab);
        sba = _mm256_fmadd_ps(_mm256_loadu_ps(yb + i), da, sba);
        sab2 = _mm256_fmadd_ps(_mm256_loadu_ps(ya + i), db, sab2);
    }
    out[0] = hsum(sab);
    out[1] = hsum(sba);
    out[2] = hsum(sab2);
}

}  // namespace avx2
}  // namespace kernels
}  // namespace vsq::turboquant

#endif  // VSQ_HAVE_AVX2_KERNELS
