#pragma once

// Shared SRHT used by TurboQuant and RaBitQ.
//
//   signs[i]     = +1 or -1 from the low bit of splitmix64 (RndGen64)
//   data         = signs ∘ data, then an in-place Walsh–Hadamard
//   scale        = 1/sqrt(d) applied after the butterfly
//   padded dim   = roundUpPow2AtLeast4(input_dim)
//
// d is a positive power of two. Coordinates past the original dim are already
// zero. Both quantizers depend on this exact generator and this exact scale;
// a different butterfly or a different sign bit changes stored codes.
//
// The header has no TurboQuant or RaBitQ types, so a consumer can include it
// without either code layout.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace turboquant {

// splitmix64. The sequence is the sign source; do not change the constants.
class RndGen64 {
  uint64_t state_;

public:
  explicit RndGen64(uint64_t const seed) : state_(seed) {}

  uint64_t next() {
    uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
};

// Unnormalized Walsh–Hadamard butterfly. d must be a positive power of two.
inline void whtInplaceScalar(float *data, size_t const d) {
  for (size_t step = 1; step < d; step <<= 1) {
    const size_t jump = step << 1;
    for (size_t i = 0; i < d; i += jump) {
      float *__restrict__ low = &data[i];
      float *__restrict__ high = &data[i + step];
      for (size_t j = 0; j < step; ++j) {
        float a = low[j];
        float b = high[j];
        low[j] = a + b;
        high[j] = a - b;
      }
    }
  }
}

// Walsh–Hadamard, then multiply every coordinate by 1/sqrt(d).
// data has length d. d must be a positive power of two.
inline void whtInplace(float *data, size_t const d) {
  if (data == nullptr)
    throw std::invalid_argument("whtInplace: data is null");
  if (d == 0 || (d & (d - 1)) != 0)
    throw std::invalid_argument("whtInplace: d must be a positive power of 2");
  whtInplaceScalar(data, d);

  float const norm = 1.0f / std::sqrt(static_cast<float>(d));
  for (size_t i = 0; i < d; ++i)
    data[i] *= norm;
}

// Length d. signs[i] is +1 when the low bit of splitmix64 is set, else -1.
inline std::vector<float> generateSigns(size_t const d, uint64_t const seed) {
  std::vector<float> signs(d);
  RndGen64 rng(seed);
  for (size_t i = 0; i < d; ++i) {
    uint64_t const bits = rng.next();
    signs[i] = (bits & 1ULL) ? 1.0f : -1.0f;
  }
  return signs;
}

// Elementwise multiply by signs, then whtInplace.
// data and signs each have length d.
inline void randomizedHadamard(float *data,
                               const float *const __restrict__ signs,
                               size_t const d) {
  if (data == nullptr || signs == nullptr)
    throw std::invalid_argument("randomizedHadamard: null pointer");
  for (size_t i = 0; i < d; ++i)
    data[i] *= signs[i];
  whtInplace(data, d);
}

// Smallest power of two that is >= n.
// n == 0 and n == 1 return 1. Throws if n does not fit in a size_t power of two.
inline size_t roundUpPow2(size_t n) {
  if (n <= 1)
    return 1;
  constexpr size_t kMaxPow2 = size_t{1} << (sizeof(size_t) * 8 - 1);
  if (n > kMaxPow2)
    throw std::invalid_argument("roundUpPow2: value does not fit in size_t");
  size_t p = 1;
  while (p < n)
    p <<= 1;
  return p;
}

// Power of two >= max(n, 4). Both spaces use this as padded_dim.
// n == 0 yields 4; the space still rejects a zero input dimension.
inline size_t roundUpPow2AtLeast4(size_t n) {
  return roundUpPow2(n < 4 ? 4 : n);
}

}  // namespace turboquant
