#pragma once
// rotation.h — random orthogonal rotation R : R^dim -> R^D (zero-padded input).
//
// Two kinds:
//
//   BlockKac (default, format v2)
//     D = dim rounded up to a multiple of 64 (no power-of-two padding:
//     768, 1536, 3072 stay as they are). P = largest power of two <= D.
//     One round r = 0..rounds-1:
//         x *= signs_r                        (Rademacher diagonal)
//         x[0, P)   <- H_P x[0, P)            (orthonormal Walsh–Hadamard)
//         if P < D:
//           Kac step: (x_i, x_{i+D/2}) <- ((x_i + x_{i+D/2}) / sqrt2,
//                                          (x_i - x_{i+D/2}) / sqrt2)
//           x[D-P, D) <- H_P x[D-P, D)
//     The two overlapping blocks plus the Kac step mix every coordinate with
//     every other one within a round; >= 3 rounds make structured inputs
//     (sparse, non-negative, Hadamard-aligned) look Gaussian. This follows the
//     FHT-Kac rotator of RaBitQ-Library.
//
//   LegacySrht (RaBitQ codes of turboquant-space 0.1.x)
//     D = roundUpPow2(max(dim, 4)), one round, srht.h signs and butterfly —
//     bit-identical to common::randomizedHadamard.
//
// Every step is a symmetric orthogonal matrix, so R preserves inner products
// and R^T is the same steps in reverse order (applyTransposePadded).
//
// Signs: rounds * D values from splitmix64 (RndGen64), +1 when the low bit is
// set. Changing the generator, the step order or the scaling changes codes.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.h"
#include "srht.h"

namespace vsq::common {

enum class RotationKind : uint8_t { BlockKac = 0, LegacySrht = 1 };

namespace detail {

// Unnormalised in-place Walsh–Hadamard butterfly on a power-of-two block,
// followed by the 1/sqrt(n) scale. Body shared by the scalar and AVX2 clones.
VSQ_ALWAYS_INLINE void fwhtBody(float *VSQ_RESTRICT x, size_t n,
                                       float scale) {
    for (size_t step = 1; step < n; step <<= 1) {
        const size_t jump = step << 1;
        for (size_t i = 0; i < n; i += jump) {
            float *VSQ_RESTRICT lo = x + i;
            float *VSQ_RESTRICT hi = x + i + step;
            for (size_t j = 0; j < step; ++j) {
                const float a = lo[j];
                const float b = hi[j];
                lo[j] = a + b;
                hi[j] = a - b;
            }
        }
    }
    for (size_t i = 0; i < n; ++i) x[i] *= scale;
}

VSQ_ALWAYS_INLINE void kacBody(float *VSQ_RESTRICT x, size_t d) {
    const size_t h = d / 2;
    const float s = static_cast<float>(1.0 / std::sqrt(2.0));
    float *VSQ_RESTRICT lo = x;
    float *VSQ_RESTRICT hi = x + h;
    for (size_t i = 0; i < h; ++i) {
        const float a = lo[i];
        const float b = hi[i];
        lo[i] = (a + b) * s;
        hi[i] = (a - b) * s;
    }
}

VSQ_ALWAYS_INLINE void mulBody(float *VSQ_RESTRICT x,
                                      const float *VSQ_RESTRICT s, size_t n) {
    for (size_t i = 0; i < n; ++i) x[i] *= s[i];
}

inline void fwhtScalar(float *x, size_t n, float scale) { fwhtBody(x, n, scale); }
inline void kacScalar(float *x, size_t d) { kacBody(x, d); }
inline void mulScalar(float *x, const float *s, size_t n) { mulBody(x, s, n); }

#if defined(VSQ_HAVE_AVX2_KERNELS)
// Same source, compiled for AVX2 so the loops vectorise 8-wide.
VSQ_TARGET_AVX2 inline void fwhtAvx2(float *x, size_t n, float scale) {
    fwhtBody(x, n, scale);
}
VSQ_TARGET_AVX2 inline void kacAvx2(float *x, size_t d) { kacBody(x, d); }
VSQ_TARGET_AVX2 inline void mulAvx2(float *x, const float *s, size_t n) {
    mulBody(x, s, n);
}
#endif

inline size_t largestPow2AtMost(size_t n) {
    size_t p = 1;
    while ((p << 1) <= n) p <<= 1;
    return p;
}

}  // namespace detail

class Rotation {
public:
    static constexpr size_t kAlign = 64;  // BlockKac D is a multiple of this
    static constexpr int kMaxRounds = 16;

    // Padded dimension D for an input dimension. dim >= 1.
    static size_t paddedDimFor(size_t dim, RotationKind kind) {
        if (dim == 0) throw std::invalid_argument("Rotation: dim must be >= 1");
        if (kind == RotationKind::LegacySrht) return roundUpPow2AtLeast4(dim);
        return (dim + kAlign - 1) / kAlign * kAlign;
    }

    // rounds is ignored for LegacySrht (always 1).
    Rotation(size_t dim, uint64_t seed, int rounds = 3,
             RotationKind kind = RotationKind::BlockKac, Isa isa = detectIsa())
        : dim_(dim),
          padded_(paddedDimFor(dim, kind)),
          block_(kind == RotationKind::LegacySrht ? padded_
                                                  : detail::largestPow2AtMost(padded_)),
          rounds_(kind == RotationKind::LegacySrht ? 1 : rounds),
          kind_(kind),
          seed_(seed),
          isa_(resolveIsa(isa)) {
        if (kind == RotationKind::BlockKac && (rounds < 1 || rounds > kMaxRounds))
            throw std::invalid_argument("Rotation: rounds must be in [1, " +
                                        std::to_string(kMaxRounds) + "], got " +
                                        std::to_string(rounds));
        if (kind == RotationKind::LegacySrht) {
            signs_ = generateSigns(padded_, seed);
        } else {
            signs_.resize(static_cast<size_t>(rounds_) * padded_);
            RndGen64 rng(seed);
            for (float &s : signs_) s = (rng.next() & 1ULL) ? 1.0f : -1.0f;
        }
    }

    size_t dim() const { return dim_; }
    size_t paddedDim() const { return padded_; }
    size_t blockSize() const { return block_; }
    int rounds() const { return rounds_; }
    RotationKind kind() const { return kind_; }
    uint64_t seed() const { return seed_; }

    // out[0, D) = R [x; 0]. x has dim() floats; out has paddedDim() floats.
    // x and out may not alias.
    void apply(const float *x, float *out) const {
        std::memcpy(out, x, dim_ * sizeof(float));
        std::memset(out + dim_, 0, (padded_ - dim_) * sizeof(float));
        applyPadded(out);
    }

    // buf[0, D) <- R buf. buf already holds a zero-padded vector.
    void applyPadded(float *buf) const {
        if (kind_ == RotationKind::LegacySrht) {
            randomizedHadamard(buf, signs_.data(), padded_);
            return;
        }
        for (int r = 0; r < rounds_; ++r) round(buf, r, /*transpose=*/false);
    }

    // buf[0, D) <- R^T buf (the inverse rotation).
    void applyTransposePadded(float *buf) const {
        if (kind_ == RotationKind::LegacySrht) {
            const float scale = 1.0f / std::sqrt(static_cast<float>(padded_));
            whtInplaceScalar(buf, padded_);
            for (size_t i = 0; i < padded_; ++i) buf[i] *= scale * signs_[i];
            return;
        }
        for (int r = rounds_ - 1; r >= 0; --r) round(buf, r, /*transpose=*/true);
    }

private:
    void fwht(float *x) const {
        const float scale = 1.0f / std::sqrt(static_cast<float>(block_));
#if defined(VSQ_HAVE_AVX2_KERNELS)
        if (isa_ == Isa::Avx2) return detail::fwhtAvx2(x, block_, scale);
#endif
        detail::fwhtScalar(x, block_, scale);
    }
    void kac(float *x) const {
#if defined(VSQ_HAVE_AVX2_KERNELS)
        if (isa_ == Isa::Avx2) return detail::kacAvx2(x, padded_);
#endif
        detail::kacScalar(x, padded_);
    }
    void mulSigns(float *x, int r) const {
        const float *s = signs_.data() + static_cast<size_t>(r) * padded_;
#if defined(VSQ_HAVE_AVX2_KERNELS)
        if (isa_ == Isa::Avx2) return detail::mulAvx2(x, s, padded_);
#endif
        detail::mulScalar(x, s, padded_);
    }

    // Forward: signs, H(front), [Kac, H(back)]. Transpose: reverse order.
    void round(float *x, int r, bool transpose) const {
        const bool split = block_ < padded_;
        float *back = x + (padded_ - block_);
        if (!transpose) {
            mulSigns(x, r);
            fwht(x);
            if (split) {
                kac(x);
                fwht(back);
            }
        } else {
            if (split) {
                fwht(back);
                kac(x);
            }
            fwht(x);
            mulSigns(x, r);
        }
    }

    size_t dim_;
    size_t padded_;
    size_t block_;
    int rounds_;
    RotationKind kind_;
    uint64_t seed_;
    Isa isa_;
    std::vector<float> signs_;  // rounds_ * padded_
};

}  // namespace vsq::common
