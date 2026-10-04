#pragma once
// Dense float32 vector primitives (dot, squared L2) with an AVX2 clone.
//
// Used outside the per-code hot path: query preparation, k-means, centroid
// tables. The bodies are plain loops; with -O3 (and -ffast-math for the
// reassociation) the compiler vectorises them, 8-wide in the AVX2 clone.

#include <cstddef>

#include "config.h"

namespace vsq::common {
namespace detail {

VSQ_ALWAYS_INLINE float dotBody(const float *VSQ_RESTRICT a,
                                       const float *VSQ_RESTRICT b, size_t n) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

VSQ_ALWAYS_INLINE float l2sqBody(const float *VSQ_RESTRICT a,
                                        const float *VSQ_RESTRICT b, size_t n) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float d0 = a[i] - b[i], d1 = a[i + 1] - b[i + 1];
        const float d2 = a[i + 2] - b[i + 2], d3 = a[i + 3] - b[i + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    for (; i < n; ++i) {
        const float d = a[i] - b[i];
        s0 += d * d;
    }
    return (s0 + s1) + (s2 + s3);
}

inline float dotScalar(const float *a, const float *b, size_t n) { return dotBody(a, b, n); }
inline float l2sqScalar(const float *a, const float *b, size_t n) { return l2sqBody(a, b, n); }

#if defined(VSQ_HAVE_AVX2_KERNELS)
VSQ_TARGET_AVX2 inline float dotAvx2(const float *a, const float *b, size_t n) {
    return dotBody(a, b, n);
}
VSQ_TARGET_AVX2 inline float l2sqAvx2(const float *a, const float *b, size_t n) {
    return l2sqBody(a, b, n);
}
#endif

}  // namespace detail

// a, b: n floats each.
inline float dotF32(const float *a, const float *b, size_t n, Isa isa = detectIsa()) {
#if defined(VSQ_HAVE_AVX2_KERNELS)
    if (isa == Isa::Avx2) return detail::dotAvx2(a, b, n);
#endif
    (void)isa;
    return detail::dotScalar(a, b, n);
}

inline float l2sqF32(const float *a, const float *b, size_t n, Isa isa = detectIsa()) {
#if defined(VSQ_HAVE_AVX2_KERNELS)
    if (isa == Isa::Avx2) return detail::l2sqAvx2(a, b, n);
#endif
    (void)isa;
    return detail::l2sqScalar(a, b, n);
}

inline float normSqF32(const float *a, size_t n, Isa isa = detectIsa()) {
    return dotF32(a, a, n, isa);
}

}  // namespace vsq::common
