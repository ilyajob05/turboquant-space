#pragma once
// Kernel table for TurboQuant v2: one set of four function pointers per
// (payload kind, ISA), chosen once when a space is constructed.

#include <cstddef>
#include <cstdint>

#include "../common/config.h"
#include "kernels_avx2.h"
#include "kernels_neon.h"
#include "kernels_scalar.h"

namespace vsq::turboquant {
namespace kernels {

enum class Payload : uint8_t { Nibble = 0, Byte = 1, Half = 2 };

// q, q1, q2, ya, yb: D floats. p, a, b: payload bytes. table(s): decode tables
// (16 or 256 floats; unused for Half). out: 2 or 3 floats.
using Dot1Fn = float (*)(const float *q, const uint8_t *p, const float *table, size_t D);
using Dot2Fn = void (*)(const float *q1, const float *q2, const uint8_t *p, const float *tA,
                        const float *tB, size_t D, float *out);
using SymFn = float (*)(const uint8_t *a, const uint8_t *b, const float *table, size_t D);
using SymCrossFn = void (*)(const uint8_t *a, const uint8_t *b, const float *ya,
                            const float *yb, const float *table, size_t D, float *out);

struct Kernels {
    Dot1Fn dot1;
    Dot2Fn dot2;
    SymFn sym;
    SymCrossFn symCross;
    common::Isa isa;
};

namespace detail {
template <class Dec>
Kernels scalarSet() {
    return {&scalar::dot1<Dec>, &scalar::dot2<Dec>, &scalar::sym<Dec>, &scalar::symCross<Dec>,
            common::Isa::Scalar};
}
#if defined(VSQ_NEON)
template <class Dec>
Kernels neonSet() {
    return {&neon::dot1<Dec>, &neon::dot2<Dec>, &neon::sym<Dec>, &neon::symCross<Dec>,
            common::Isa::Neon};
}
#endif
#if defined(VSQ_HAVE_AVX2_KERNELS)
template <class Dec>
Kernels avx2Set() {
    return {&avx2::dot1<Dec>, &avx2::dot2<Dec>, &avx2::sym<Dec>, &avx2::symCross<Dec>,
            common::Isa::Avx2};
}
#endif
}  // namespace detail

// `isa` is clamped to what the CPU supports (resolveIsa).
inline Kernels selectKernels(Payload payload, common::Isa isa) {
    const common::Isa use = common::resolveIsa(isa);
#if defined(VSQ_HAVE_AVX2_KERNELS)
    if (use == common::Isa::Avx2) {
        switch (payload) {
        case Payload::Nibble: return detail::avx2Set<avx2::Nibble>();
        case Payload::Byte: return detail::avx2Set<avx2::Byte>();
        case Payload::Half: return detail::avx2Set<avx2::Half>();
        }
    }
#endif
#if defined(VSQ_NEON)
    if (use == common::Isa::Neon) {
        switch (payload) {
        case Payload::Nibble: return detail::neonSet<neon::Nibble>();
        case Payload::Byte: return detail::neonSet<neon::Byte>();
        case Payload::Half: return detail::neonSet<neon::Half>();
        }
    }
#endif
    switch (payload) {
    case Payload::Nibble: return detail::scalarSet<scalar::Nibble>();
    case Payload::Byte: return detail::scalarSet<scalar::Byte>();
    default: return detail::scalarSet<scalar::Half>();
    }
}

}  // namespace kernels
}  // namespace vsq::turboquant
