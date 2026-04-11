#pragma once
/// turbo_quant.h — TurboQuant (ICLR 2026, arXiv:2504.19874), Algorithm 2
///
/// Data structures and math utilities for TurboQuant quantization.
/// All encoding/distance/search logic lives in space_turbo_quant.h.
///
///   byte[i] = (sq_idx << 1) | qjl_bit, 1 byte/coord
///   meta: [norm, gamma, sigma] = 3 x float32 immediately after packed bytes

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace turboquant {

// ===========================================================================
// RndGen64 — splitmix64 PRNG for deterministic sign generation
// ===========================================================================

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

// ===========================================================================
// Walsh-Hadamard Transform + Randomized Hadamard Transform
// ===========================================================================

// Walsh-Hadamard Transform (WHT) — scalar fallback
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

// Walsh-Hadamard Transform (WHT)
inline void whtInplace(float *data, size_t const d) {
  assert(d > 0 && (d & (d - 1)) == 0 &&
         "whtInplace: d must be a positive power of 2");
  whtInplaceScalar(data, d);

  // Normalize
  float norm = 1.0f / std::sqrt(static_cast<float>(d));
  for (size_t i = 0; i < d; ++i)
    data[i] *= norm;
}

inline std::vector<float> generateSigns(size_t const d, uint64_t const seed) {
  std::vector<float> signs(d);
  RndGen64 rng(seed);
  for (size_t i = 0; i < d; ++i) {
    uint64_t bits = rng.next();
    if (bits & 1) {
      signs[i] = 1.0f;
    } else {
        signs[i] = -1.0f;
    }
  }
  return signs;
}

// Randomized Walsh-Hadamard Transform. Elementwise multiply + WHT.
inline void randomizedHadamard(float *data,
                               const float *const __restrict__ signs,
                               size_t const d) {
  assert(d > 0 && (d & (d - 1)) == 0 &&
         "randomizedHadamard: d must be a positive power of 2");

  for (size_t i = 0; i < d; ++i) {
    data[i] *= signs[i];
  }
  whtInplace(data, d);
}

// ===========================================================================
//
// Layout depends on bits_per_coord:
//   b >= 5: 1 byte/coord, packed[i] = (sq_idx << 1) | qjl_bit
//           total = dim + 12 bytes
//   b == 4 (3+1): 1 nibble/coord, 2 coords/byte
//           byte[i] = (nibble[2i+1] << 4) | nibble[2i],
//           nibble = (sq_idx << 1) | qjl_bit  (sq_idx in 0..7)
//           total = dim/2 + 12 bytes
//
// Does NOT own the buffer — caller manages lifetime.
// ===========================================================================

class TurboQuantCode {
public:
  uint8_t *sq_packed_;
  float *meta_;

  TurboQuantCode() : sq_packed_(nullptr), meta_(nullptr) {}

  /// Wrap an existing  buffer slot.
  TurboQuantCode(void *buf, size_t dim, int bits_per_coord = 8)
      : sq_packed_(reinterpret_cast<uint8_t *>(buf)),
        meta_(reinterpret_cast<float *>(
            static_cast<char *>(buf) + packedBytes(dim, bits_per_coord))) {}

  /// Const version for read-only access.
  TurboQuantCode(const void *buf, size_t dim, int bits_per_coord = 8)
      : sq_packed_(const_cast<uint8_t *>(
            reinterpret_cast<const uint8_t *>(buf))),
        meta_(const_cast<float *>(
            reinterpret_cast<const float *>(
                static_cast<const char *>(buf) +
                packedBytes(dim, bits_per_coord)))) {}

  // -- Packed unit accessors ------------------------------------------------
  //
  // These work for the full-byte layout (b>=5). For packed-nibble layout
  // (b<=4), use the variants that take bits_per_coord or call the
  // space-level helpers directly.

  inline uint8_t sqIndex(size_t i) const { return sq_packed_[i] >> 1; }
  inline bool qjlSign(size_t i) const { return sq_packed_[i] & 1; }
  inline void set(size_t i, uint8_t sq_idx, bool qjl_positive) {
    sq_packed_[i] = static_cast<uint8_t>((sq_idx << 1) | (qjl_positive ? 1u : 0u));
  }

  /// Layout-aware unit accessor. Returns the 4- or 8-bit packed unit for
  /// coordinate i: (sq_idx << 1) | qjl_bit.
  inline uint8_t unit(size_t i, int bits_per_coord) const {
    if (bits_per_coord <= 4) {
      uint8_t byte = sq_packed_[i >> 1];
      return (i & 1) ? (byte >> 4) : (byte & 0x0F);
    }
    return sq_packed_[i];
  }
  inline uint8_t sqIndex(size_t i, int bits_per_coord) const {
    return unit(i, bits_per_coord) >> 1;
  }
  inline bool qjlSign(size_t i, int bits_per_coord) const {
    return unit(i, bits_per_coord) & 1;
  }

  // -- Meta accessors -------------------------------------------------------

  float norm() const { return meta_[0]; }
  float gamma() const { return meta_[1]; }
  float sigma() const { return meta_[2]; }

  void setNorm(float v) { meta_[0] = v; }
  void setGamma(float v) { meta_[1] = v; }
  void setSigma(float v) { meta_[2] = v; }

  // -- Size -----------------------------------------------------------------

  /// Bytes used by the packed region for given dim/bits.
  static size_t packedBytes(size_t dim, int bits_per_coord) {
    return (bits_per_coord <= 4) ? (dim + 1) / 2 : dim;
  }

  ///  buffer size in bytes for a given dimension and bit budget.
  static size_t codeSizeBytes(size_t dim, int bits_per_coord = 8) {
    return packedBytes(dim, bits_per_coord) + sizeof(float) * 3;
  }
};

} // namespace turboquant
