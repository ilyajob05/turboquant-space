#pragma once
// IEEE-754 binary16 <-> binary32 conversion, used by the 16-bit payload.
//
// Scalar versions are exact (round-to-nearest-even, subnormals, inf/nan) and
// define the stored bits; SIMD paths (F16C on x86, vcvt on aarch64) produce the
// same results, so codes are identical whichever ISA encoded them.

#include <cstdint>
#include <cstring>

namespace vsq::common {

inline uint16_t floatToHalf(float value) {
    uint32_t f;
    std::memcpy(&f, &value, sizeof(f));
    const uint32_t sign = (f >> 16) & 0x8000u;
    const uint32_t abs = f & 0x7FFFFFFFu;
    if (abs >= 0x7F800000u)  // inf / nan (keep a quiet-nan payload bit)
        return static_cast<uint16_t>(sign | 0x7C00u | (abs > 0x7F800000u ? 0x200u : 0u));
    if (abs >= 0x477FF000u)  // rounds to >= 65520 -> inf
        return static_cast<uint16_t>(sign | 0x7C00u);
    if (abs < 0x38800000u) {  // result is subnormal or zero
        if (abs < 0x33000000u) return static_cast<uint16_t>(sign);  // < 2^-25
        const uint32_t mant = (abs & 0x007FFFFFu) | 0x00800000u;
        const int shift = 126 - static_cast<int>(abs >> 23);  // 14..24
        const uint32_t half_mant = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        const uint32_t round = (rem > halfway || (rem == halfway && (half_mant & 1u))) ? 1u : 0u;
        return static_cast<uint16_t>(sign | (half_mant + round));
    }
    // normal: rebias exponent 127 -> 15, round mantissa 23 -> 10 bits (RNE)
    const uint32_t base = abs - 0x38000000u;
    const uint32_t low = base & 0x1FFFu;
    const uint32_t round = (low > 0x1000u || (low == 0x1000u && (base & 0x2000u))) ? 1u : 0u;
    return static_cast<uint16_t>(sign | ((base >> 13) + round));
}

inline float halfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {  // subnormal: normalise
            int e = -1;
            do {
                ++e;
                mant <<= 1;
            } while ((mant & 0x400u) == 0);
            f = sign | (static_cast<uint32_t>(112 - e) << 23) | ((mant & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

}  // namespace vsq::common
