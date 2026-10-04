#pragma once
// NEON (aarch64 baseline) kernels for TurboQuant v2 payloads.
//
// Same four kernels as kernels_scalar.h. Decoders produce 8 coordinates as
// two float32x4_t:
//   Nibble  16-entry float table as a 64-byte TBL4 lookup (byte indices
//           4*idx + {0,1,2,3}), no scalar loads
//   Byte    256-entry table: scalar loads (AArch64 has no float gather)
//   Half    fcvtl / fcvtl2

#include <cstddef>
#include <cstdint>

#include "../common/config.h"

#if defined(VSQ_NEON)

namespace vsq::turboquant {
namespace kernels {
namespace neon {

struct F8 {
    float32x4_t lo, hi;
};

struct Nibble {
    uint8x16x4_t lut;
    uint8x16_t sel_lo, sel_hi, offs;
    explicit Nibble(const float *t) {
        const uint8_t *b = reinterpret_cast<const uint8_t *>(t);
        lut.val[0] = vld1q_u8(b);
        lut.val[1] = vld1q_u8(b + 16);
        lut.val[2] = vld1q_u8(b + 32);
        lut.val[3] = vld1q_u8(b + 48);
        static const uint8_t k_sel_lo[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
        static const uint8_t k_sel_hi[16] = {4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7};
        static const uint8_t k_offs[16] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
        sel_lo = vld1q_u8(k_sel_lo);
        sel_hi = vld1q_u8(k_sel_hi);
        offs = vld1q_u8(k_offs);
    }
    // Coordinates i..i+7 (i % 8 == 0): 4 payload bytes, low nibble first.
    VSQ_ALWAYS_INLINE F8 load8(const uint8_t *p, size_t i) const {
        const uint8x8_t b =
            vcreate_u8(static_cast<uint64_t>(common::loadUnaligned<uint32_t>(p + (i >> 1))));
        const uint8x8_t n = vzip1_u8(vand_u8(b, vdup_n_u8(0x0F)), vshr_n_u8(b, 4));  // n0..n7
        const uint8x16_t nn = vcombine_u8(n, n);
        const uint8x16_t ilo = vaddq_u8(vshlq_n_u8(vqtbl1q_u8(nn, sel_lo), 2), offs);
        const uint8x16_t ihi = vaddq_u8(vshlq_n_u8(vqtbl1q_u8(nn, sel_hi), 2), offs);
        return {vreinterpretq_f32_u8(vqtbl4q_u8(lut, ilo)),
                vreinterpretq_f32_u8(vqtbl4q_u8(lut, ihi))};
    }
};

struct Byte {
    const float *t;
    explicit Byte(const float *table) : t(table) {}
    VSQ_ALWAYS_INLINE F8 load8(const uint8_t *p, size_t i) const {
        const float lo[4] = {t[p[i]], t[p[i + 1]], t[p[i + 2]], t[p[i + 3]]};
        const float hi[4] = {t[p[i + 4]], t[p[i + 5]], t[p[i + 6]], t[p[i + 7]]};
        return {vld1q_f32(lo), vld1q_f32(hi)};
    }
};

struct Half {
    explicit Half(const float *) {}
    VSQ_ALWAYS_INLINE F8 load8(const uint8_t *p, size_t i) const {
        const float16x8_t h = vreinterpretq_f16_u8(vld1q_u8(p + 2 * i));
        return {vcvt_f32_f16(vget_low_f16(h)), vcvt_high_f32_f16(h)};
    }
};

template <class Dec>
float dot1(const float *q, const uint8_t *p, const float *table, size_t D) {
    const Dec d(table);
    float32x4_t s0 = vdupq_n_f32(0.f), s1 = vdupq_n_f32(0.f);
    for (size_t i = 0; i < D; i += 8) {
        const F8 v = d.load8(p, i);
        s0 = vfmaq_f32(s0, vld1q_f32(q + i), v.lo);
        s1 = vfmaq_f32(s1, vld1q_f32(q + i + 4), v.hi);
    }
    return vaddvq_f32(vaddq_f32(s0, s1));
}

template <class Dec>
void dot2(const float *q1, const float *q2, const uint8_t *p, const float *tA,
          const float *tB, size_t D, float *out) {
    const Dec a(tA), b(tB);
    float32x4_t s1 = vdupq_n_f32(0.f), s2 = vdupq_n_f32(0.f);
    for (size_t i = 0; i < D; i += 8) {
        const F8 va = a.load8(p, i), vb = b.load8(p, i);
        s1 = vfmaq_f32(s1, vld1q_f32(q1 + i), va.lo);
        s1 = vfmaq_f32(s1, vld1q_f32(q1 + i + 4), va.hi);
        s2 = vfmaq_f32(s2, vld1q_f32(q2 + i), vb.lo);
        s2 = vfmaq_f32(s2, vld1q_f32(q2 + i + 4), vb.hi);
    }
    out[0] = vaddvq_f32(s1);
    out[1] = vaddvq_f32(s2);
}

template <class Dec>
float sym(const uint8_t *a, const uint8_t *b, const float *table, size_t D) {
    const Dec d(table);
    float32x4_t s0 = vdupq_n_f32(0.f), s1 = vdupq_n_f32(0.f);
    for (size_t i = 0; i < D; i += 8) {
        const F8 va = d.load8(a, i), vb = d.load8(b, i);
        s0 = vfmaq_f32(s0, va.lo, vb.lo);
        s1 = vfmaq_f32(s1, va.hi, vb.hi);
    }
    return vaddvq_f32(vaddq_f32(s0, s1));
}

template <class Dec>
void symCross(const uint8_t *a, const uint8_t *b, const float *ya, const float *yb,
              const float *table, size_t D, float *out) {
    const Dec d(table);
    float32x4_t sab = vdupq_n_f32(0.f), sba = vdupq_n_f32(0.f), sab2 = vdupq_n_f32(0.f);
    for (size_t i = 0; i < D; i += 8) {
        const F8 va = d.load8(a, i), vb = d.load8(b, i);
        sab = vfmaq_f32(vfmaq_f32(sab, va.lo, vb.lo), va.hi, vb.hi);
        sba = vfmaq_f32(vfmaq_f32(sba, vld1q_f32(yb + i), va.lo), vld1q_f32(yb + i + 4), va.hi);
        sab2 = vfmaq_f32(vfmaq_f32(sab2, vld1q_f32(ya + i), vb.lo), vld1q_f32(ya + i + 4), vb.hi);
    }
    out[0] = vaddvq_f32(sab);
    out[1] = vaddvq_f32(sba);
    out[2] = vaddvq_f32(sab2);
}

}  // namespace neon
}  // namespace kernels
}  // namespace vsq::turboquant

#endif  // VSQ_NEON
