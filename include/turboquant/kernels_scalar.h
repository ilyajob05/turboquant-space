#pragma once
// Scalar reference kernels for TurboQuant v2 payloads.
//
// A payload of D coordinates (D % 8 == 0) decodes to D floats through a
// Decoder (value at coordinate i). The kernels are written once over the
// Decoder concept; every SIMD file implements the same four kernels and must
// agree with these up to float reassociation.
//
//   dot1     sum_i q[i] * dec(p, i)
//   dot2     sum_i q1[i] * decA(p, i),  sum_i q2[i] * decB(p, i)
//   sym      sum_i dec(a, i) * dec(b, i)
//   symCross sum_i dec(a,i) dec(b,i),  sum_i yb[i] dec(a,i),  sum_i ya[i] dec(b,i)

#include <cstddef>
#include <cstdint>

#include "../common/config.h"
#include "../common/fp16.h"

namespace vsq::turboquant {
namespace kernels {
namespace scalar {

// 4-bit index per coordinate, low nibble = even coordinate; 16-entry table.
struct Nibble {
    const float *t;
    float at(const uint8_t *p, size_t i) const {
        const uint8_t b = p[i >> 1];
        return t[(i & 1) ? (b >> 4) : (b & 0x0F)];
    }
};

// 8-bit index per coordinate; 256-entry table.
struct Byte {
    const float *t;
    float at(const uint8_t *p, size_t i) const { return t[p[i]]; }
};

// IEEE binary16 per coordinate (little-endian); no table.
struct Half {
    const float *t;  // unused
    float at(const uint8_t *p, size_t i) const {
        return common::halfToFloat(common::loadUnaligned<uint16_t>(p + 2 * i));
    }
};

template <class Dec>
float dot1(const float *q, const uint8_t *p, const float *table, size_t D) {
    const Dec d{table};
    float s0 = 0.f, s1 = 0.f;
    for (size_t i = 0; i < D; i += 2) {
        s0 += q[i] * d.at(p, i);
        s1 += q[i + 1] * d.at(p, i + 1);
    }
    return s0 + s1;
}

template <class Dec>
void dot2(const float *q1, const float *q2, const uint8_t *p, const float *tA,
          const float *tB, size_t D, float *out) {
    const Dec a{tA}, b{tB};
    float s1 = 0.f, s2 = 0.f;
    for (size_t i = 0; i < D; ++i) {
        s1 += q1[i] * a.at(p, i);
        s2 += q2[i] * b.at(p, i);
    }
    out[0] = s1;
    out[1] = s2;
}

template <class Dec>
float sym(const uint8_t *a, const uint8_t *b, const float *table, size_t D) {
    const Dec d{table};
    float s0 = 0.f, s1 = 0.f;
    for (size_t i = 0; i < D; i += 2) {
        s0 += d.at(a, i) * d.at(b, i);
        s1 += d.at(a, i + 1) * d.at(b, i + 1);
    }
    return s0 + s1;
}

template <class Dec>
void symCross(const uint8_t *a, const uint8_t *b, const float *ya, const float *yb,
              const float *table, size_t D, float *out) {
    const Dec d{table};
    float sab = 0.f, sba = 0.f, sab2 = 0.f;
    for (size_t i = 0; i < D; ++i) {
        const float da = d.at(a, i), db = d.at(b, i);
        sab += da * db;
        sba += yb[i] * da;
        sab2 += ya[i] * db;
    }
    out[0] = sab;
    out[1] = sba;
    out[2] = sab2;
}

}  // namespace scalar
}  // namespace kernels
}  // namespace vsq::turboquant
