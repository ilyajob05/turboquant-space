#pragma once
/// turboquant.h — TurboQuant (ICLR 2026, arXiv:2504.19874), Algorithm 2
///
/// Packed code layout (TurboQuantCode).
/// The shared SRHT lives in srht.h.
/// Encoding and distance live in space_turboquant.h.
///
///   byte[i] = (sq_idx << 1) | qjl_bit, 1 byte/coord
///   meta: [norm, gamma, sigma] = 3 x float32 immediately after packed bytes

#include <cstddef>
#include <cstdint>

#include "srht.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace turboquant {

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
