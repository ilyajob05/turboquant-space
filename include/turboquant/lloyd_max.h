#pragma once
// lloyd_max.h — MSE-optimal scalar quantizer for N(0, 1) (Lloyd–Max).
//
// For b bits: L = 2^b levels, centroids c[0..L) ascending and symmetric
// (c[L-1-i] = -c[i]), boundaries t[0..L-1) with t[i] = (c[i] + c[i+1]) / 2.
// quantizeIndex(v) = #{i : v > t[i]} in [0, L).
//
// Tables for b = 1..8 are computed once per process (thread-safe static) by
// Lloyd iteration in double precision, initialised at the normal quantiles
// Phi^-1((i + 0.5) / L), which converges in a few hundred iterations even at
// 8 bits.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/config.h"

namespace vsq::turboquant {

struct LloydMaxQuantizer {
    int bits = 0;
    std::vector<float> centroids;   // 2^bits, ascending
    std::vector<float> boundaries;  // 2^bits - 1, ascending
    double mse = 0.0;               // E[(Z - Q(Z))^2], Z ~ N(0,1)
};

namespace detail {

inline double normalPdf(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * common::kPi); }
inline double normalCdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// Acklam's rational approximation of Phi^-1 (|rel err| < 1.2e-9); only used
// to seed the Lloyd iteration, so its accuracy does not affect the result.
inline double normalQuantile(double p) {
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                               -2.759285104469687e+02, 1.383577518672690e+02,
                               -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                               -1.556989798598866e+02, 6.680131188771972e+01,
                               -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                               -2.400758277161838e+00, -2.549732539343734e+00,
                               4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                               2.445134137142996e+00, 3.754408661907416e+00};
    const double lo = 0.02425, hi = 1.0 - lo;
    if (p < lo) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p > hi) {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const double q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

inline LloydMaxQuantizer buildLloydMax(int bits) {
    const int levels = 1 << bits;
    const int half = levels / 2;  // solve the positive half, mirror the rest
    std::vector<double> c(half);
    for (int i = 0; i < half; ++i)
        c[i] = normalQuantile((half + i + 0.5) / static_cast<double>(levels));

    // E[Z | a < Z < b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a)); b = inf allowed.
    auto condMean = [](double a, double b) {
        const double pb = std::isinf(b) ? 0.0 : normalPdf(b);
        const double mass = (std::isinf(b) ? 1.0 : normalCdf(b)) - normalCdf(a);
        return mass > 1e-300 ? (normalPdf(a) - pb) / mass : 0.5 * (a + b);
    };
    constexpr int kMaxIter = 20000;
    constexpr double kTol = 1e-13;
    for (int iter = 0; iter < kMaxIter; ++iter) {
        double delta = 0.0;
        double prev_edge = 0.0;
        for (int i = 0; i < half; ++i) {
            const double edge = (i + 1 < half) ? 0.5 * (c[i] + c[i + 1]) : INFINITY;
            const double nc = condMean(prev_edge, edge);
            delta = std::max(delta, std::fabs(nc - c[i]));
            c[i] = nc;
            prev_edge = edge;
        }
        if (delta < kTol) break;
    }

    LloydMaxQuantizer q;
    q.bits = bits;
    q.centroids.resize(levels);
    for (int i = 0; i < half; ++i) {
        q.centroids[half + i] = static_cast<float>(c[i]);
        q.centroids[half - 1 - i] = static_cast<float>(-c[i]);
    }
    q.boundaries.resize(levels - 1);
    for (int i = 0; i + 1 < levels; ++i)
        q.boundaries[i] = 0.5f * (q.centroids[i] + q.centroids[i + 1]);

    // MSE = 1 - sum_i c_i^2 P(cell_i) when each centroid is its cell's mean.
    double gain = 0.0, prev_edge = 0.0;
    for (int i = 0; i < half; ++i) {
        const double edge = (i + 1 < half) ? 0.5 * (c[i] + c[i + 1]) : INFINITY;
        const double mass = (std::isinf(edge) ? 1.0 : normalCdf(edge)) - normalCdf(prev_edge);
        gain += 2.0 * c[i] * c[i] * mass;
        prev_edge = edge;
    }
    q.mse = 1.0 - gain;
    return q;
}

}  // namespace detail

constexpr int kLloydMaxMinBits = 1;
constexpr int kLloydMaxMaxBits = 8;

// Shared table for `bits` in [1, 8]. Throws std::invalid_argument otherwise.
inline const LloydMaxQuantizer &lloydMax(int bits) {
    if (bits < kLloydMaxMinBits || bits > kLloydMaxMaxBits)
        throw std::invalid_argument("Lloyd-Max: bits must be in [1, 8], got " +
                                    std::to_string(bits));
    static const std::array<LloydMaxQuantizer, kLloydMaxMaxBits + 1> tables = [] {
        std::array<LloydMaxQuantizer, kLloydMaxMaxBits + 1> t{};
        for (int b = kLloydMaxMinBits; b <= kLloydMaxMaxBits; ++b) t[b] = detail::buildLloydMax(b);
        return t;
    }();
    return tables[bits];
}

// #{i : v > boundaries[i]} by a branch-free binary search: exactly `bits`
// compare/select steps (vs 2^bits - 1 compares of a linear scan).
// boundaries has 2^bits - 1 ascending entries.
VSQ_ALWAYS_INLINE uint32_t quantizeIndex(const float *boundaries, int bits, float v) {
    uint32_t idx = 0;
    for (int s = bits - 1; s >= 0; --s) {
        const uint32_t probe = idx + (1u << s);
        idx = (v > boundaries[probe - 1]) ? probe : idx;
    }
    return idx;
}

}  // namespace vsq::turboquant
