#pragma once
/// space_turbo_quant.h — TurboQuant standalone space
///
/// Self-contained. Provides:
///   - TurboQuantSpace: encode float vectors into TurboQuant codes
///   - distance(query, code):              asymmetric, single
///   - distanceSymmetric(code_a, code_b):  symmetric, single
///   - distanceBatch1ToN / distanceBatchMToN / distanceBatchMToNSymmetric
///
/// All buffers (vectors, codes, distance matrices) are owned by the caller;
/// the library never allocates user-visible storage.

#include "turbo_quant.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------
// SIMD feature detection
// ---------------------------------------------------------------------------
#if defined(__AVX__) || defined(__AVX2__)
#  define USE_AVX
#endif
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#  define USE_SSE
#endif
#if defined(__aarch64__) || defined(__ARM_NEON)
#  define USE_NEON
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#  include <immintrin.h>
#endif
#if defined(USE_NEON)
#  include <arm_neon.h>
#endif

// ---------------------------------------------------------------------------
// OpenMP — optional. Falls back to sequential execution when not available.
// ---------------------------------------------------------------------------
#if defined(TURBOQUANT_HAVE_OPENMP)
#  include <omp.h>
#  define TURBOQUANT_OMP_STRINGIFY(x) #x
#  define TURBOQUANT_OMP_PRAGMA(x) _Pragma(TURBOQUANT_OMP_STRINGIFY(x))
#  define TURBOQUANT_OMP_PARALLEL_FOR(nt, n)                                   \
      TURBOQUANT_OMP_PRAGMA(omp parallel for schedule(static)                  \
                            num_threads(nt) if ((n) > 64))
#else
#  define TURBOQUANT_OMP_PARALLEL_FOR(nt, n)
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#elif defined(_MSC_VER)
#  define PORTABLE_ALIGN32 __declspec(align(32))
#else
#  define PORTABLE_ALIGN32
#endif

namespace turboquant {

// ===========================================================================
// Lloyd-Max quantizer for Gaussian N(0,1)
// ===========================================================================

struct LloydMaxTable {
    std::vector<float> boundaries;
    std::vector<float> centroids;
};

inline LloydMaxTable computeLloydMax(int bits, int maxIter = 1000,
                                     double tol = 1e-12) {
    const int levels = 1 << bits;
    const int half = levels / 2;

    auto phi = [](double x) -> double {
        return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
    };
    auto Phi = [](double x) -> double {
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    };
    auto conditionalMean = [&](double a, double b) -> double {
        double denom = Phi(b) - Phi(a);
        if (denom < 1e-15)
            return 0.5 * (a + b);
        return (phi(a) - phi(b)) / denom;
    };

    std::vector<double> pos_c(half);
    for (int i = 0; i < half; ++i)
        pos_c[i] = (i + 0.5) * 3.5 / half;

    for (int iter = 0; iter < maxIter; ++iter) {
        std::vector<double> pos_b(half - 1);
        for (int i = 0; i < half - 1; ++i)
            pos_b[i] = 0.5 * (pos_c[i] + pos_c[i + 1]);

        std::vector<double> new_c(half);
        double maxDelta = 0.0;
        for (int i = 0; i < half; ++i) {
            double lo = (i == 0) ? 0.0 : pos_b[i - 1];
            double hi = (i == half - 1) ? 1e10 : pos_b[i];
            new_c[i] = conditionalMean(lo, hi);
            maxDelta = std::max(maxDelta, std::abs(new_c[i] - pos_c[i]));
        }
        pos_c = new_c;
        if (maxDelta < tol)
            break;
    }

    LloydMaxTable table;
    table.centroids.resize(levels);
    table.boundaries.resize(levels - 1);

    for (int i = 0; i < half; ++i) {
        table.centroids[half + i] = static_cast<float>(pos_c[i]);
        table.centroids[half - 1 - i] = -static_cast<float>(pos_c[i]);
    }
    table.boundaries[half - 1] = 0.0f;
    for (int i = 0; i < half - 1; ++i) {
        double b = 0.5 * (pos_c[i] + pos_c[i + 1]);
        table.boundaries[half + i] = static_cast<float>(b);
        table.boundaries[half - 2 - i] = -static_cast<float>(b);
    }
    return table;
}

// ===========================================================================
// TurboQuantPreparedQuery — pre-computed query state for asymmetric search.
// Lives on the caller's stack; no shared state.
// ===========================================================================

struct TurboQuantPreparedQuery {
    std::vector<float> q_rot;
    std::vector<float> s_q;
    float q_norm_sq;
    float q_norm;
    const float *centroids;
};

// ===========================================================================
// TurboQuantPreparedSymCode — pre-computed state for full-symmetric distance
// from one quantized code (the "query" side). Reuses the per-pair work that
// distBuildFullScalar would otherwise repeat for every base code.
// Holds:
//   recon_q  — centroids[sq_idx]*sigma_q AFTER QJL Hadamard, length padded_dim
//   sign_q   — ±1.0f sign bits, length padded_dim
//   norm_q, gamma_q, sigma_q — scalar metadata
//   scale_gamma_q  — space->scale() * gamma_q   (precomputed for term3)
//   pi_over_2d_gamma_q — (π/(2d)) * gamma_q     (precomputed for term4)
// ===========================================================================

struct TurboQuantPreparedSymCode {
    std::vector<float> recon_q;
    std::vector<float> sign_q;
    float norm_q;
    float gamma_q;
    float sigma_q;
    float scale_gamma_q;
    float pi_over_2d_gamma_q;
};

// ===========================================================================
// Forward declarations
// ===========================================================================

class TurboQuantSpace;

// Distance function pointer type: matches the (q, code, space_ptr) and
// (code_a, code_b, space_ptr) signatures used by the SIMD implementations.
using TQDistFunc = float (*)(const void *, const void *, const void *);

static float distSearchScalar(const void *q, const void *code_buf, const void *qty_ptr);
static float distBuildScalar(const void *pVect1, const void *pVect2, const void *param_ptr);
static float distBuildFullScalar(const void *pVect1, const void *pVect2, const void *param_ptr);
static float distBuildFullPreparedScalar(const void *pq_sym, const void *code_b, const void *param_ptr);
static float distBuildLightScalar(const void *pVect1, const void *pVect2, const void *param_ptr);
static float distSearchScalarB4(const void *q, const void *code_buf, const void *qty_ptr);
static float distBuildScalarB4(const void *pVect1, const void *pVect2, const void *param_ptr);

#if defined(USE_NEON)
static float distSearchNEON(const void *q, const void *code_buf, const void *qty_ptr);
static float distBuildNEON(const void *pVect1, const void *pVect2, const void *param_ptr);
static float distSearchNEONB4(const void *q, const void *code_buf, const void *qty_ptr);
static float distBuildNEONB4(const void *pVect1, const void *pVect2, const void *param_ptr);
#endif

#if defined(USE_SSE)
static float distSearchSSE(const void *q, const void *code_buf, const void *qty_ptr);
static float distBuildSSE(const void *pVect1, const void *pVect2, const void *param_ptr);
#endif

#if defined(USE_AVX)
static float distSearchAVX(const void *q, const void *code_buf, const void *qty_ptr);
static float distBuildAVX(const void *pVect1, const void *pVect2, const void *param_ptr);
#endif

// ===========================================================================
// TurboQuantSpace
// ===========================================================================

class TurboQuantSpace {
    LloydMaxTable lm_table_;
    float *boundaries_;
    uint16_t num_boundaries_;
    float *centroids_;

    std::vector<float> rotation_signs_;
    std::vector<float> qjl_signs_precomp_;

    const uint64_t rot_seed_;
    const uint64_t qjl_seed_;
    const size_t input_dim_;
    const size_t dim_;
    const bool padded_;
    const int bits_per_coord_;
    const bool packed_nibbles_;
    const size_t packed_bytes_;
    const size_t data_size_;
    const int num_levels_;
    const float scale_;
    const int num_threads_;

    TQDistFunc fstdistfunc_search_;
    TQDistFunc fstdistfunc_build_;

public:
    static size_t roundUpPow2(size_t d) {
        if (d <= 1) return 1;
        size_t p = 1;
        while (p < d) p <<= 1;
        return p;
    }

    static int resolveNumThreads(int requested) {
#if defined(TURBOQUANT_HAVE_OPENMP)
        if (requested > 0) return requested;
        return omp_get_max_threads();
#else
        (void)requested;
        return 1;
#endif
    }

    TurboQuantSpace(size_t dim,
                    int bits_per_coord = 4,
                    uint64_t rot_seed = 42,
                    uint64_t qjl_seed = 137,
                    int num_threads = 0)
        : rot_seed_(rot_seed)
        , qjl_seed_(qjl_seed)
        , input_dim_(dim)
        , dim_(roundUpPow2(dim < 4 ? 4 : dim))
        , padded_(dim_ != dim)
        , bits_per_coord_(bits_per_coord)
        , packed_nibbles_(bits_per_coord <= 4)
        , packed_bytes_(TurboQuantCode::packedBytes(dim_, bits_per_coord))
        , data_size_(TurboQuantCode::codeSizeBytes(dim_, bits_per_coord))
        , num_levels_(1 << (bits_per_coord - 1))
        , scale_(std::sqrt(static_cast<float>(M_PI) / 2.0f) /
                 std::sqrt(static_cast<float>(dim_)))
        , num_threads_(resolveNumThreads(num_threads)) {
        assert(dim >= 1 && "TurboQuantSpace: dim must be >= 1");
        if (padded_) {
            std::cerr << "[TurboQuantSpace] warning: input dim " << input_dim_
                      << " is not a power of 2 (or < 4); padding to " << dim_
                      << " with zeros for internal computation.\n";
        }
        assert(bits_per_coord >= 2 &&
               "TurboQuantSpace: need at least 2 bits (1 MSE + 1 QJL)");
        assert(bits_per_coord <= 9 &&
               "TurboQuantSpace: max 9 bits (8-bit MSE + 1 QJL)");

        // Runtime SIMD dispatch. Packed-nibble (b<=4) uses NEON LUT variant
        // where available; SSE/AVX b4 are not implemented — fall back to scalar.
        if (packed_nibbles_) {
#if defined(USE_NEON)
            fstdistfunc_search_ = distSearchNEONB4;
            fstdistfunc_build_ = distBuildNEONB4;
#else
            fstdistfunc_search_ = distSearchScalarB4;
            fstdistfunc_build_ = distBuildScalarB4;
#endif
        } else {
#if defined(USE_AVX)
            fstdistfunc_search_ = distSearchAVX;
            fstdistfunc_build_ = distBuildAVX;
#elif defined(USE_SSE)
            fstdistfunc_search_ = distSearchSSE;
            fstdistfunc_build_ = distBuildSSE;
#elif defined(USE_NEON)
            fstdistfunc_search_ = distSearchNEON;
            fstdistfunc_build_ = distBuildNEON;
#else
            fstdistfunc_search_ = distSearchScalar;
            fstdistfunc_build_ = distBuildScalar;
#endif
        }

        lm_table_ = computeLloydMax(bits_per_coord - 1);
        boundaries_ = lm_table_.boundaries.data();
        num_boundaries_ = static_cast<uint16_t>(lm_table_.boundaries.size());
        centroids_ = lm_table_.centroids.data();
        rotation_signs_ = generateSigns(dim_, rot_seed_);
        qjl_signs_precomp_ = generateSigns(dim_, qjl_seed_);
    }

    // -- Accessors ------------------------------------------------------------

    size_t dim() const { return input_dim_; }
    size_t paddedDim() const { return dim_; }
    bool padded() const { return padded_; }
    int numThreads() const { return num_threads_; }
    size_t codeSizeBytes() const { return data_size_; }
    size_t packedBytes() const { return packed_bytes_; }
    int bitsPerCoord() const { return bits_per_coord_; }
    bool packedNibbles() const { return packed_nibbles_; }
    int numLevels() const { return num_levels_; }
    float scale() const { return scale_; }
    const float *centroids() const { return centroids_; }
    uint64_t rotSeed() const { return rot_seed_; }
    uint64_t qjlSeed() const { return qjl_seed_; }
    const float *qjlSigns() const { return qjl_signs_precomp_.data(); }

    // -- Encoding -------------------------------------------------------------

    inline uint8_t quantize(const float val) const {
        uint8_t idx = 0;
        for (uint16_t i = 0; i < num_boundaries_; ++i)
            idx += (val > boundaries_[i]);
        return idx;
    }

    void encodeVector(const float *raw, void *out_buf) const {
        TurboQuantCode code(out_buf, dim_, bits_per_coord_);

        float norm_sq = 0.0f;
        for (size_t i = 0; i < input_dim_; ++i)
            norm_sq += raw[i] * raw[i];
        float norm = std::sqrt(norm_sq);

        std::vector<float> rotated(dim_, 0.0f);
        float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
        for (size_t i = 0; i < input_dim_; ++i)
            rotated[i] = raw[i] * inv_norm;
        randomizedHadamard(rotated.data(), rotation_signs_.data(), dim_);

        float var = 0.0f;
        for (size_t i = 0; i < dim_; ++i)
            var += rotated[i] * rotated[i];
        float sigma = std::sqrt(var / static_cast<float>(dim_));
        if (sigma < 1e-10f)
            sigma = 1e-10f;
        float inv_sigma = 1.0f / sigma;

        std::vector<float> residual(dim_);
        std::vector<uint8_t> sq_idx(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            float normalized = rotated[i] * inv_sigma;
            sq_idx[i] = quantize(normalized);
            residual[i] = rotated[i] - centroids_[sq_idx[i]] * sigma;
        }

        float gamma_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i)
            gamma_sq += residual[i] * residual[i];
        float gamma = std::sqrt(gamma_sq);

        randomizedHadamard(residual.data(), qjl_signs_precomp_.data(), dim_);
        uint8_t *out = code.sq_packed_;
        if (packed_nibbles_) {
            std::memset(out, 0, packed_bytes_);
            for (size_t i = 0; i < dim_; ++i) {
                uint8_t unit = static_cast<uint8_t>(
                    (sq_idx[i] << 1) | ((residual[i] >= 0.0f) ? 1u : 0u));
                if ((i & 1) == 0)
                    out[i >> 1] = unit;
                else
                    out[i >> 1] |= unit << 4;
            }
        } else {
            for (size_t i = 0; i < dim_; ++i) {
                out[i] = static_cast<uint8_t>(
                    (sq_idx[i] << 1) | ((residual[i] >= 0.0f) ? 1u : 0u));
            }
        }

        code.setNorm(norm);
        code.setGamma(gamma);
        code.setSigma(sigma);
    }

    /// Encode a contiguous batch of n vectors (row-major [n, dim])
    /// into out_buf of size n * codeSizeBytes().
    void encodeBatch(const float *raws, size_t n, void *out_buf) const {
        char *out = static_cast<char *>(out_buf);
        const size_t cs = data_size_;
        TURBOQUANT_OMP_PARALLEL_FOR(num_threads_, n)
        for (long long i = 0; i < static_cast<long long>(n); ++i)
            encodeVector(raws + static_cast<size_t>(i) * input_dim_,
                         out + static_cast<size_t>(i) * cs);
    }

    // -- Query preparation ----------------------------------------------------

    TurboQuantPreparedQuery prepareQuery(const float *raw_query) const {
        TurboQuantPreparedQuery pq;
        const size_t d = dim_;

        pq.q_norm_sq = 0.0f;
        for (size_t i = 0; i < input_dim_; ++i)
            pq.q_norm_sq += raw_query[i] * raw_query[i];
        pq.q_norm = std::sqrt(pq.q_norm_sq);
        float q_inv = (pq.q_norm > 1e-10f) ? (1.0f / pq.q_norm) : 0.0f;

        pq.q_rot.assign(d, 0.0f);
        for (size_t i = 0; i < input_dim_; ++i)
            pq.q_rot[i] = raw_query[i] * q_inv;
        randomizedHadamard(pq.q_rot.data(), rotation_signs_.data(), d);

        pq.centroids = centroids_;

        pq.s_q = pq.q_rot;
        randomizedHadamard(pq.s_q.data(), qjl_signs_precomp_.data(), d);

        return pq;
    }

    // -- Distance -------------------------------------------------------------

    /// Asymmetric distance: prepared query × code.
    float distance(const TurboQuantPreparedQuery &pq,
                   const void *code_buf) const {
        return fstdistfunc_search_(&pq, code_buf, this);
    }

    /// Convenience: prepare query inline.
    float distance(const float *raw_query, const void *code_buf) const {
        auto pq = prepareQuery(raw_query);
        return distance(pq, code_buf);
    }

    /// Symmetric distance: code × code (MSE-only, original).
    float distanceSymmetric(const void *code_a, const void *code_b) const {
        return fstdistfunc_build_(code_a, code_b, this);
    }

    /// Symmetric distance with full QJL correction (reconstructs centroids
    /// + Hadamard). Computes all 4 inner product terms.
    float distanceSymmetricFull(const void *code_a, const void *code_b) const {
        return distBuildFullScalar(code_a, code_b, this);
    }

    /// Symmetric distance with light QJL correction (sign-bit dot product only).
    /// Adds only the <e_a, e_b> term — cheap, no Hadamard needed.
    float distanceSymmetricLight(const void *code_a, const void *code_b) const {
        return distBuildLightScalar(code_a, code_b, this);
    }

    // -- Prepared symmetric query --------------------------------------------

    /// Pre-compute reconstruction + Hadamard for one quantized code, so that
    /// it can be reused across many base codes in 1-to-N / M-to-N symmetric
    /// full-distance loops. The base side is NOT cached — it is unpacked
    /// inline in the hot loop to keep memory flat.
    TurboQuantPreparedSymCode prepareSymmetricQuery(const void *code_q) const {
        TurboQuantPreparedSymCode pq;
        const size_t d = dim_;
        const size_t pb = packed_bytes_;

        const auto *buf = static_cast<const char *>(code_q);
        const auto *packed = reinterpret_cast<const uint8_t *>(buf);
        const float *meta = reinterpret_cast<const float *>(buf + pb);
        pq.norm_q = meta[0];
        pq.gamma_q = meta[1];
        pq.sigma_q = meta[2];

        pq.recon_q.assign(d, 0.0f);
        pq.sign_q.assign(d, 0.0f);
        const float sigma_q = pq.sigma_q;
        if (packed_nibbles_) {
            for (size_t i = 0; i < d; ++i) {
                uint8_t byte = packed[i >> 1];
                uint8_t u = (i & 1) ? (byte >> 4) : (byte & 0x0F);
                pq.recon_q[i] = centroids_[u >> 1] * sigma_q;
                pq.sign_q[i] = (u & 1) ? 1.0f : -1.0f;
            }
        } else {
            for (size_t i = 0; i < d; ++i) {
                uint8_t u = packed[i];
                pq.recon_q[i] = centroids_[u >> 1] * sigma_q;
                pq.sign_q[i] = (u & 1) ? 1.0f : -1.0f;
            }
        }
        randomizedHadamard(pq.recon_q.data(), qjl_signs_precomp_.data(), d);

        pq.scale_gamma_q = scale_ * pq.gamma_q;
        const float pi_over_2d =
            static_cast<float>(M_PI) / (2.0f * static_cast<float>(d));
        pq.pi_over_2d_gamma_q = pi_over_2d * pq.gamma_q;
        return pq;
    }

    /// Full-symmetric distance using a pre-computed query side.
    float distanceSymmetricFullPrepared(const TurboQuantPreparedSymCode &pq,
                                        const void *code_b) const {
        return distBuildFullPreparedScalar(&pq, code_b, this);
    }

    /// 1-to-N full-symmetric: one prepared code × n base codes.
    void distanceBatch1ToNSymmetricFull(const TurboQuantPreparedSymCode &pq,
                                        const void *codes, size_t n,
                                        float *out) const {
        const char *base = static_cast<const char *>(codes);
        const size_t cs = data_size_;
        TURBOQUANT_OMP_PARALLEL_FOR(num_threads_, n)
        for (long long i = 0; i < static_cast<long long>(n); ++i)
            out[i] = distBuildFullPreparedScalar(
                &pq, base + static_cast<size_t>(i) * cs, this);
    }

    /// M-to-N full-symmetric using prepared queries: prepares each of the m
    /// query-side codes once, then loops over n base codes. Output row-major.
    void distanceBatchMToNSymmetricFullPrepared(
        const void *codes_a, size_t m,
        const void *codes_b, size_t n,
        float *out) const {
        const char *ba = static_cast<const char *>(codes_a);
        const char *bb = static_cast<const char *>(codes_b);
        const size_t cs = data_size_;
        TURBOQUANT_OMP_PARALLEL_FOR(num_threads_, m)
        for (long long i = 0; i < static_cast<long long>(m); ++i) {
            auto pq = prepareSymmetricQuery(ba + static_cast<size_t>(i) * cs);
            float *row = out + static_cast<size_t>(i) * n;
            for (size_t j = 0; j < n; ++j)
                row[j] = distBuildFullPreparedScalar(
                    &pq, bb + j * cs, this);
        }
    }

    // -- Batch distance -------------------------------------------------------

    /// 1-to-N asymmetric: one raw query against n codes.
    void distanceBatch1ToN(const float *raw_query, const void *codes,
                           size_t n, float *out) const {
        auto pq = prepareQuery(raw_query);
        const char *base = static_cast<const char *>(codes);
        const size_t cs = data_size_;
        TURBOQUANT_OMP_PARALLEL_FOR(num_threads_, n)
        for (long long i = 0; i < static_cast<long long>(n); ++i)
            out[i] = fstdistfunc_search_(&pq, base + static_cast<size_t>(i) * cs, this);
    }

    /// M-to-N asymmetric: m raw queries × n codes. `out` is row-major [m, n].
    void distanceBatchMToN(const float *queries, size_t m,
                           const void *codes, size_t n, float *out) const {
        const char *base = static_cast<const char *>(codes);
        const size_t cs = data_size_;
        TURBOQUANT_OMP_PARALLEL_FOR(num_threads_, m)
        for (long long qi = 0; qi < static_cast<long long>(m); ++qi) {
            auto pq = prepareQuery(queries + static_cast<size_t>(qi) * input_dim_);
            float *row = out + static_cast<size_t>(qi) * n;
            for (size_t i = 0; i < n; ++i)
                row[i] = fstdistfunc_search_(&pq, base + i * cs, this);
        }
    }

    /// M-to-N symmetric: m codes × n codes. `out` is row-major [m, n].
    void distanceBatchMToNSymmetric(const void *codes_a, size_t m,
                                    const void *codes_b, size_t n,
                                    float *out) const {
        const char *ba = static_cast<const char *>(codes_a);
        const char *bb = static_cast<const char *>(codes_b);
        const size_t cs = data_size_;
        TURBOQUANT_OMP_PARALLEL_FOR(num_threads_, m)
        for (long long i = 0; i < static_cast<long long>(m); ++i) {
            float *row = out + static_cast<size_t>(i) * n;
            for (size_t j = 0; j < n; ++j)
                row[j] = fstdistfunc_build_(ba + static_cast<size_t>(i) * cs,
                                            bb + j * cs, this);
        }
    }

    /// M-to-N symmetric with full QJL correction.
    void distanceBatchMToNSymmetricFull(const void *codes_a, size_t m,
                                        const void *codes_b, size_t n,
                                        float *out) const {
        const char *ba = static_cast<const char *>(codes_a);
        const char *bb = static_cast<const char *>(codes_b);
        const size_t cs = data_size_;
        TURBOQUANT_OMP_PARALLEL_FOR(num_threads_, m)
        for (long long i = 0; i < static_cast<long long>(m); ++i) {
            float *row = out + static_cast<size_t>(i) * n;
            for (size_t j = 0; j < n; ++j)
                row[j] = distBuildFullScalar(ba + static_cast<size_t>(i) * cs,
                                             bb + j * cs, this);
        }
    }

    /// M-to-N symmetric with light QJL correction (sign-bit only).
    void distanceBatchMToNSymmetricLight(const void *codes_a, size_t m,
                                         const void *codes_b, size_t n,
                                         float *out) const {
        const char *ba = static_cast<const char *>(codes_a);
        const char *bb = static_cast<const char *>(codes_b);
        const size_t cs = data_size_;
        TURBOQUANT_OMP_PARALLEL_FOR(num_threads_, m)
        for (long long i = 0; i < static_cast<long long>(m); ++i) {
            float *row = out + static_cast<size_t>(i) * n;
            for (size_t j = 0; j < n; ++j)
                row[j] = distBuildLightScalar(ba + static_cast<size_t>(i) * cs,
                                              bb + j * cs, this);
        }
    }
};

// ===========================================================================
// Distance function implementations (scalar + SIMD)
// qty_ptr/param_ptr types (were SpaceInterface*) are now TurboQuantSpace*.
// ===========================================================================

// ---------------------------------------------------------------------------
// Scalar fallback (b=8 layout)
// ---------------------------------------------------------------------------

static float distSearchScalar(const void *q, const void *code_buf,
                              const void *qty_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(qty_ptr);
  const auto *pq = static_cast<const TurboQuantPreparedQuery *>(q);
  const size_t dim = space->paddedDim();
  const uint8_t *packed = reinterpret_cast<const uint8_t *>(code_buf);
  const float *meta = reinterpret_cast<const float *>(
      static_cast<const char *>(code_buf) + dim);
  const float x_norm = meta[0];
  const float gamma = meta[1];
  const float sigma = meta[2];

  float ip_mse = 0.0f;
  float dot_qjl = 0.0f;
  const float *centroids = pq->centroids;
  const float *q_rot = pq->q_rot.data();
  const float *s_q = pq->s_q.data();
  for (size_t i = 0; i < dim; ++i) {
    uint8_t byte = packed[i];
    ip_mse += q_rot[i] * centroids[byte >> 1];
    float sign = (byte & 1) ? 1.0f : -1.0f;
    dot_qjl += s_q[i] * sign;
  }
  ip_mse *= sigma;
  const float correction = space->scale() * gamma * dot_qjl;
  const float ip = (ip_mse + correction) * x_norm * pq->q_norm;
  return std::max(0.0f, pq->q_norm_sq + x_norm * x_norm - 2.0f * ip);
}

static float distBuildScalar(const void *pVect1, const void *pVect2,
                             const void *param_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
  const size_t dim = space->paddedDim();
  const float *centroids = space->centroids();

  const char *buf_a = static_cast<const char *>(pVect1);
  const char *buf_b = static_cast<const char *>(pVect2);

  const uint8_t *packed_a = reinterpret_cast<const uint8_t *>(buf_a);
  const float *meta_a = reinterpret_cast<const float *>(buf_a + dim);
  const float norm_a = meta_a[0];
  const float sigma_a = meta_a[2];

  const uint8_t *packed_b = reinterpret_cast<const uint8_t *>(buf_b);
  const float *meta_b = reinterpret_cast<const float *>(buf_b + dim);
  const float norm_b = meta_b[0];
  const float sigma_b = meta_b[2];

  float ip_rot = 0.0f;
  for (size_t i = 0; i < dim; ++i)
    ip_rot += (centroids[packed_a[i] >> 1] * sigma_a) *
              (centroids[packed_b[i] >> 1] * sigma_b);

  const float ip = ip_rot * norm_a * norm_b;
  return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
}

// ---------------------------------------------------------------------------
// Symmetric distance with QJL correction — "full" variant
// Reconstructs r̃ from centroids, applies QJL Hadamard, computes all 4 IP terms:
//   <r_a, r_b> = <r̃_a, r̃_b> + <r̃_a, e_b> + <e_a, r̃_b> + <e_a, e_b>
// ---------------------------------------------------------------------------

// Helper: extract packed unit (sq_idx << 1 | qjl_bit) for coordinate i
static inline uint8_t extractUnit(const uint8_t *packed, size_t i, bool nibble) {
  if (nibble) {
    uint8_t byte = packed[i >> 1];
    return (i & 1) ? (byte >> 4) : (byte & 0x0F);
  }
  return packed[i];
}

static float distBuildFullScalar(const void *pVect1, const void *pVect2,
                                 const void *param_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
  const size_t dim = space->paddedDim();
  const size_t pb = space->packedBytes();
  const bool nibble = space->packedNibbles();
  const float *centroids = space->centroids();

  const char *buf_a = static_cast<const char *>(pVect1);
  const char *buf_b = static_cast<const char *>(pVect2);

  const uint8_t *packed_a = reinterpret_cast<const uint8_t *>(buf_a);
  const float *meta_a = reinterpret_cast<const float *>(buf_a + pb);
  const float norm_a = meta_a[0];
  const float gamma_a = meta_a[1];
  const float sigma_a = meta_a[2];

  const uint8_t *packed_b = reinterpret_cast<const uint8_t *>(buf_b);
  const float *meta_b = reinterpret_cast<const float *>(buf_b + pb);
  const float norm_b = meta_b[0];
  const float gamma_b = meta_b[1];
  const float sigma_b = meta_b[2];

  // Term 1: <r̃_a, r̃_b> and reconstruct for Hadamard
  float ip_mse = 0.0f;
  std::vector<float> recon_a(dim), recon_b(dim);
  for (size_t i = 0; i < dim; ++i) {
    uint8_t ua = extractUnit(packed_a, i, nibble);
    uint8_t ub = extractUnit(packed_b, i, nibble);
    float ca = centroids[ua >> 1] * sigma_a;
    float cb = centroids[ub >> 1] * sigma_b;
    ip_mse += ca * cb;
    recon_a[i] = ca;
    recon_b[i] = cb;
  }

  // Apply QJL Hadamard to reconstructed vectors
  randomizedHadamard(recon_a.data(), space->qjlSigns(), dim);
  randomizedHadamard(recon_b.data(), space->qjlSigns(), dim);

  // Term 2: <r̃_a, e_b> ≈ scale * gamma_b * Σ(s̃_a[i] * sign_b[i])
  // Term 3: <e_a, r̃_b> ≈ scale * gamma_a * Σ(sign_a[i] * s̃_b[i])
  // Term 4: <e_a, e_b> ≈ (π/(2d)) * gamma_a * gamma_b * Σ(sign_a * sign_b)
  float dot_recon_a_sign_b = 0.0f;
  float dot_sign_a_recon_b = 0.0f;
  float dot_signs = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    uint8_t ua = extractUnit(packed_a, i, nibble);
    uint8_t ub = extractUnit(packed_b, i, nibble);
    float sa = (ua & 1) ? 1.0f : -1.0f;
    float sb = (ub & 1) ? 1.0f : -1.0f;
    dot_recon_a_sign_b += recon_a[i] * sb;
    dot_sign_a_recon_b += sa * recon_b[i];
    dot_signs += sa * sb;
  }
  float term2 = space->scale() * gamma_b * dot_recon_a_sign_b;
  float term3 = space->scale() * gamma_a * dot_sign_a_recon_b;
  const float pi_over_2d = static_cast<float>(M_PI) /
                           (2.0f * static_cast<float>(dim));
  float term4 = pi_over_2d * gamma_a * gamma_b * dot_signs;

  float total_correction = term2 + term3 + term4;
  // Clamp correction to avoid overflow when residuals are large (low bit budgets)
  float max_corr = std::abs(ip_mse);
  total_correction = std::max(-max_corr, std::min(max_corr, total_correction));

  const float ip = (ip_mse + total_correction) * norm_a * norm_b;
  return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
}

// ---------------------------------------------------------------------------
// Symmetric distance — full QJL correction with PREPARED query side.
// Reuses recon_q + sign_q + scalars precomputed by prepareSymmetricQuery,
// so the per-pair cost drops from 2 reconstructions + 2 Hadamards to
// 1 reconstruction + 1 Hadamard for the base side only.
// ---------------------------------------------------------------------------

static float distBuildFullPreparedScalar(const void *pq_ptr,
                                         const void *code_b,
                                         const void *param_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
  const auto *pq = static_cast<const TurboQuantPreparedSymCode *>(pq_ptr);
  const size_t dim = space->paddedDim();
  const size_t pb = space->packedBytes();
  const bool nibble = space->packedNibbles();
  const float *centroids = space->centroids();

  const char *buf_b = static_cast<const char *>(code_b);
  const uint8_t *packed_b = reinterpret_cast<const uint8_t *>(buf_b);
  const float *meta_b = reinterpret_cast<const float *>(buf_b + pb);
  const float norm_b = meta_b[0];
  const float gamma_b = meta_b[1];
  const float sigma_b = meta_b[2];

  const float *recon_q = pq->recon_q.data();
  const float *sign_q = pq->sign_q.data();

  // Reconstruct r̃_b from centroids (pre-Hadamard), then apply QJL Hadamard.
  // recon_q is already POST-Hadamard from prepareSymmetricQuery.
  std::vector<float> recon_b(dim);
  if (nibble) {
    for (size_t i = 0; i < dim; ++i) {
      uint8_t byte = packed_b[i >> 1];
      uint8_t ub = (i & 1) ? (byte >> 4) : (byte & 0x0F);
      recon_b[i] = centroids[ub >> 1] * sigma_b;
    }
  } else {
    for (size_t i = 0; i < dim; ++i) {
      uint8_t ub = packed_b[i];
      recon_b[i] = centroids[ub >> 1] * sigma_b;
    }
  }
  randomizedHadamard(recon_b.data(), space->qjlSigns(), dim);

  // All four dot products in one pass.
  // Term 1: <recon_q, recon_b>   (Hadamard orthonormal → equals pre-Hadamard ip)
  // Term 2: <recon_q, sign_b>    × scale * gamma_b
  // Term 3: <sign_q, recon_b>    × scale * gamma_q   (precomputed in pq)
  // Term 4: <sign_q, sign_b>     × (π/(2d)) * gamma_q * gamma_b   (gamma_q precomp.)
  float dot_recon = 0.0f;
  float dot_recon_q_sign_b = 0.0f;
  float dot_sign_q_recon_b = 0.0f;
  float dot_signs = 0.0f;
  if (nibble) {
    for (size_t i = 0; i < dim; ++i) {
      uint8_t byte = packed_b[i >> 1];
      uint8_t ub = (i & 1) ? (byte >> 4) : (byte & 0x0F);
      float sb = (ub & 1) ? 1.0f : -1.0f;
      float rq = recon_q[i];
      float rb = recon_b[i];
      float sq = sign_q[i];
      dot_recon += rq * rb;
      dot_recon_q_sign_b += rq * sb;
      dot_sign_q_recon_b += sq * rb;
      dot_signs += sq * sb;
    }
  } else {
    for (size_t i = 0; i < dim; ++i) {
      uint8_t ub = packed_b[i];
      float sb = (ub & 1) ? 1.0f : -1.0f;
      float rq = recon_q[i];
      float rb = recon_b[i];
      float sq = sign_q[i];
      dot_recon += rq * rb;
      dot_recon_q_sign_b += rq * sb;
      dot_sign_q_recon_b += sq * rb;
      dot_signs += sq * sb;
    }
  }

  const float ip_mse = dot_recon;
  const float scale = space->scale();
  float term2 = scale * gamma_b * dot_recon_q_sign_b;
  float term3 = pq->scale_gamma_q * dot_sign_q_recon_b;
  float term4 = pq->pi_over_2d_gamma_q * gamma_b * dot_signs;

  float total_correction = term2 + term3 + term4;
  float max_corr = std::abs(ip_mse);
  total_correction = std::max(-max_corr, std::min(max_corr, total_correction));

  const float norm_q = pq->norm_q;
  const float ip = (ip_mse + total_correction) * norm_q * norm_b;
  return std::max(0.0f, norm_q * norm_q + norm_b * norm_b - 2.0f * ip);
}

// ---------------------------------------------------------------------------
// Symmetric distance with QJL correction — "light" variant
// Only adds the cheap <e_a, e_b> term via sign bit dot product (popcount).
// No Hadamard needed — just XOR + popcount on packed sign bits.
// ---------------------------------------------------------------------------

static float distBuildLightScalar(const void *pVect1, const void *pVect2,
                                  const void *param_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
  const size_t dim = space->paddedDim();
  const size_t pb = space->packedBytes();
  const bool nibble = space->packedNibbles();
  const float *centroids = space->centroids();

  const char *buf_a = static_cast<const char *>(pVect1);
  const char *buf_b = static_cast<const char *>(pVect2);

  const uint8_t *packed_a = reinterpret_cast<const uint8_t *>(buf_a);
  const float *meta_a = reinterpret_cast<const float *>(buf_a + pb);
  const float norm_a = meta_a[0];
  const float gamma_a = meta_a[1];
  const float sigma_a = meta_a[2];

  const uint8_t *packed_b = reinterpret_cast<const uint8_t *>(buf_b);
  const float *meta_b = reinterpret_cast<const float *>(buf_b + pb);
  const float norm_b = meta_b[0];
  const float gamma_b = meta_b[1];
  const float sigma_b = meta_b[2];

  float ip_mse = 0.0f;
  int agree = 0;
  for (size_t i = 0; i < dim; ++i) {
    uint8_t ua = extractUnit(packed_a, i, nibble);
    uint8_t ub = extractUnit(packed_b, i, nibble);
    ip_mse += (centroids[ua >> 1] * sigma_a) *
              (centroids[ub >> 1] * sigma_b);
    agree += ((ua ^ ub) & 1) == 0 ? 1 : 0;
  }
  float dot_signs = static_cast<float>(2 * agree - static_cast<int>(dim));

  const float pi_over_2d = static_cast<float>(M_PI) /
                           (2.0f * static_cast<float>(dim));
  float correction = pi_over_2d * gamma_a * gamma_b * dot_signs;
  // Clamp correction to avoid overflow when residuals are large (low bit budgets)
  float max_corr = std::abs(ip_mse);
  correction = std::max(-max_corr, std::min(max_corr, correction));

  const float ip = (ip_mse + correction) * norm_a * norm_b;
  return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
}

// ---------------------------------------------------------------------------
// Packed-nibble (b<=4) scalar variants
// ---------------------------------------------------------------------------

static float distSearchScalarB4(const void *q, const void *code_buf,
                                const void *qty_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(qty_ptr);
  const auto *pq = static_cast<const TurboQuantPreparedQuery *>(q);
  const size_t dim = space->paddedDim();
  const size_t packed_bytes = space->packedBytes();
  const uint8_t *packed = reinterpret_cast<const uint8_t *>(code_buf);
  const float *meta = reinterpret_cast<const float *>(
      static_cast<const char *>(code_buf) + packed_bytes);
  const float x_norm = meta[0];
  const float gamma = meta[1];
  const float sigma = meta[2];

  const float *centroids = pq->centroids;
  const float *q_rot = pq->q_rot.data();
  const float *s_q = pq->s_q.data();

  float ip_mse = 0.0f;
  float dot_qjl = 0.0f;
  for (size_t i = 0, b = 0; i < dim; i += 2, ++b) {
    uint8_t byte = packed[b];
    uint8_t lo = byte & 0x0F;
    uint8_t hi = byte >> 4;
    ip_mse += q_rot[i]     * centroids[lo >> 1];
    ip_mse += q_rot[i + 1] * centroids[hi >> 1];
    dot_qjl += s_q[i]     * ((lo & 1) ? 1.0f : -1.0f);
    dot_qjl += s_q[i + 1] * ((hi & 1) ? 1.0f : -1.0f);
  }
  ip_mse *= sigma;
  const float correction = space->scale() * gamma * dot_qjl;
  const float ip = (ip_mse + correction) * x_norm * pq->q_norm;
  return std::max(0.0f, pq->q_norm_sq + x_norm * x_norm - 2.0f * ip);
}

static float distBuildScalarB4(const void *pVect1, const void *pVect2,
                               const void *param_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
  const size_t dim = space->paddedDim();
  const size_t packed_bytes = space->packedBytes();
  const float *centroids = space->centroids();

  const char *buf_a = static_cast<const char *>(pVect1);
  const char *buf_b = static_cast<const char *>(pVect2);

  const uint8_t *packed_a = reinterpret_cast<const uint8_t *>(buf_a);
  const float *meta_a = reinterpret_cast<const float *>(buf_a + packed_bytes);
  const float norm_a = meta_a[0];
  const float sigma_a = meta_a[2];

  const uint8_t *packed_b = reinterpret_cast<const uint8_t *>(buf_b);
  const float *meta_b = reinterpret_cast<const float *>(buf_b + packed_bytes);
  const float norm_b = meta_b[0];
  const float sigma_b = meta_b[2];

  float ip_rot = 0.0f;
  for (size_t i = 0, bi = 0; i < dim; i += 2, ++bi) {
    uint8_t ba = packed_a[bi];
    uint8_t bb = packed_b[bi];
    uint8_t lo_a = (ba & 0x0F) >> 1;
    uint8_t hi_a = (ba >> 4)  >> 1;
    uint8_t lo_b = (bb & 0x0F) >> 1;
    uint8_t hi_b = (bb >> 4)  >> 1;
    ip_rot += (centroids[lo_a] * sigma_a) * (centroids[lo_b] * sigma_b);
    ip_rot += (centroids[hi_a] * sigma_a) * (centroids[hi_b] * sigma_b);
  }

  const float ip = ip_rot * norm_a * norm_b;
  return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
}

// ---------------------------------------------------------------------------
// NEON implementation (ARM, 4 floats per iteration)
// ---------------------------------------------------------------------------
#if defined(USE_NEON)

static float distSearchNEON(const void *q, const void *code_buf,
                            const void *qty_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(qty_ptr);
  const auto *pq = static_cast<const TurboQuantPreparedQuery *>(q);
  const size_t dim = space->paddedDim();
  const uint8_t *packed = reinterpret_cast<const uint8_t *>(code_buf);
  const float *meta = reinterpret_cast<const float *>(
      static_cast<const char *>(code_buf) + dim);
  const float x_norm = meta[0];
  const float gamma = meta[1];
  const float sigma = meta[2];

  const float *centroids = pq->centroids;
  const float *q_rot = pq->q_rot.data();
  const float *s_q = pq->s_q.data();

  const uint32x4_t sign_bit = vdupq_n_u32(0x80000000u);
  const uint8x8_t one_u8 = vdup_n_u8(1);

  float32x4_t sum_ip0 = vdupq_n_f32(0.0f);
  float32x4_t sum_ip1 = vdupq_n_f32(0.0f);
  float32x4_t sum_qjl0 = vdupq_n_f32(0.0f);
  float32x4_t sum_qjl1 = vdupq_n_f32(0.0f);

  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    float c0[4], c1[4];
    c0[0] = centroids[packed[i] >> 1];
    c0[1] = centroids[packed[i + 1] >> 1];
    c0[2] = centroids[packed[i + 2] >> 1];
    c0[3] = centroids[packed[i + 3] >> 1];
    c1[0] = centroids[packed[i + 4] >> 1];
    c1[1] = centroids[packed[i + 5] >> 1];
    c1[2] = centroids[packed[i + 6] >> 1];
    c1[3] = centroids[packed[i + 7] >> 1];
    sum_ip0 = vmlaq_f32(sum_ip0, vld1q_f32(q_rot + i), vld1q_f32(c0));
    sum_ip1 = vmlaq_f32(sum_ip1, vld1q_f32(q_rot + i + 4), vld1q_f32(c1));

    uint8x8_t bytes = vld1_u8(packed + i);
    uint8x8_t bits = vand_u8(bytes, one_u8);
    int8x8_t signs = vsub_s8(vreinterpret_s8_u8(vadd_u8(bits, bits)),
                             vdup_n_s8(1));
    int16x8_t s16 = vmovl_s8(signs);
    uint32x4_t neg0 = vcltq_s32(vmovl_s16(vget_low_s16(s16)), vdupq_n_s32(0));
    uint32x4_t neg1 = vcltq_s32(vmovl_s16(vget_high_s16(s16)), vdupq_n_s32(0));

    sum_qjl0 =
        vaddq_f32(sum_qjl0, vreinterpretq_f32_u32(veorq_u32(
                                vreinterpretq_u32_f32(vld1q_f32(s_q + i)),
                                vandq_u32(neg0, sign_bit))));
    sum_qjl1 =
        vaddq_f32(sum_qjl1, vreinterpretq_f32_u32(veorq_u32(
                                vreinterpretq_u32_f32(vld1q_f32(s_q + i + 4)),
                                vandq_u32(neg1, sign_bit))));
  }

  for (; i + 4 <= dim; i += 4) {
    float cv[4];
    cv[0] = centroids[packed[i] >> 1];
    cv[1] = centroids[packed[i + 1] >> 1];
    cv[2] = centroids[packed[i + 2] >> 1];
    cv[3] = centroids[packed[i + 3] >> 1];
    sum_ip0 = vmlaq_f32(sum_ip0, vld1q_f32(q_rot + i), vld1q_f32(cv));

    float sq_arr[4];
    for (int j = 0; j < 4; ++j)
      sq_arr[j] = s_q[i + j] * ((packed[i + j] & 1) ? 1.0f : -1.0f);
    sum_qjl0 = vaddq_f32(sum_qjl0, vld1q_f32(sq_arr));
  }

  float ip_mse = vaddvq_f32(vaddq_f32(sum_ip0, sum_ip1)) * sigma;
  float dot_qjl = vaddvq_f32(vaddq_f32(sum_qjl0, sum_qjl1));

  const float correction = space->scale() * gamma * dot_qjl;
  const float ip = (ip_mse + correction) * x_norm * pq->q_norm;
  return std::max(0.0f, pq->q_norm_sq + x_norm * x_norm - 2.0f * ip);
}

static float distBuildNEON(const void *pVect1, const void *pVect2,
                           const void *param_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
  const size_t dim = space->paddedDim();
  const float *centroids = space->centroids();

  const char *buf_a = static_cast<const char *>(pVect1);
  const char *buf_b = static_cast<const char *>(pVect2);

  const uint8_t *packed_a = reinterpret_cast<const uint8_t *>(buf_a);
  const float *meta_a = reinterpret_cast<const float *>(buf_a + dim);
  const float norm_a = meta_a[0];
  const float sigma_a = meta_a[2];

  const uint8_t *packed_b = reinterpret_cast<const uint8_t *>(buf_b);
  const float *meta_b = reinterpret_cast<const float *>(buf_b + dim);
  const float norm_b = meta_b[0];
  const float sigma_b = meta_b[2];

  const float32x4_t v_sigma_a = vdupq_n_f32(sigma_a);
  const float32x4_t v_sigma_b = vdupq_n_f32(sigma_b);
  float32x4_t sum = vdupq_n_f32(0.0f);

  size_t i = 0;
  for (; i + 4 <= dim; i += 4) {
    float ca[4], cb[4];
    ca[0] = centroids[packed_a[i] >> 1];
    cb[0] = centroids[packed_b[i] >> 1];
    ca[1] = centroids[packed_a[i + 1] >> 1];
    cb[1] = centroids[packed_b[i + 1] >> 1];
    ca[2] = centroids[packed_a[i + 2] >> 1];
    cb[2] = centroids[packed_b[i + 2] >> 1];
    ca[3] = centroids[packed_a[i + 3] >> 1];
    cb[3] = centroids[packed_b[i + 3] >> 1];

    float32x4_t va = vmulq_f32(vld1q_f32(ca), v_sigma_a);
    float32x4_t vb = vmulq_f32(vld1q_f32(cb), v_sigma_b);
    sum = vmlaq_f32(sum, va, vb);
  }

  float ip_rot = vaddvq_f32(sum);

  for (; i < dim; ++i)
    ip_rot += (centroids[packed_a[i] >> 1] * sigma_a) *
              (centroids[packed_b[i] >> 1] * sigma_b);

  const float ip = ip_rot * norm_a * norm_b;
  return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
}

// ---------------------------------------------------------------------------
// NEON b=4 (packed nibble) — tbl-based centroid gather
// ---------------------------------------------------------------------------

static inline uint8x16x2_t loadCentroidLUTb4(const float *centroids) {
  uint8x16x2_t lut;
  lut.val[0] = vld1q_u8(reinterpret_cast<const uint8_t *>(centroids));
  lut.val[1] = vld1q_u8(reinterpret_cast<const uint8_t *>(centroids) + 16);
  return lut;
}

static inline uint8x16_t buildByteIdx4(uint8x8_t sq4) {
  static const uint8_t dup_tbl[16] = {
      0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  static const uint8_t off_tbl[16] = {
      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

  uint8x16_t dup_idx = vld1q_u8(dup_tbl);
  uint8x16_t off = vld1q_u8(off_tbl);

  uint8x16_t sq16 = vcombine_u8(sq4, vdup_n_u8(0));
  uint8x16_t sq_dup = vqtbl1q_u8(sq16, dup_idx);
  return vaddq_u8(vshlq_n_u8(sq_dup, 2), off);
}

static float distSearchNEONB4(const void *q, const void *code_buf,
                              const void *qty_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(qty_ptr);
  const auto *pq = static_cast<const TurboQuantPreparedQuery *>(q);
  const size_t dim = space->paddedDim();
  const size_t packed_bytes = space->packedBytes();
  const uint8_t *packed = reinterpret_cast<const uint8_t *>(code_buf);
  const float *meta = reinterpret_cast<const float *>(
      static_cast<const char *>(code_buf) + packed_bytes);
  const float x_norm = meta[0];
  const float gamma = meta[1];
  const float sigma = meta[2];

  const float *centroids = pq->centroids;
  const float *q_rot = pq->q_rot.data();
  const float *s_q = pq->s_q.data();

  const uint8x16x2_t lut = loadCentroidLUTb4(centroids);
  const uint32x4_t sign_bit = vdupq_n_u32(0x80000000u);
  const uint8x8_t mask_lo = vdup_n_u8(0x0F);
  const uint8x8_t mask_1 = vdup_n_u8(0x01);

  float32x4_t sum_ip0 = vdupq_n_f32(0.0f);
  float32x4_t sum_ip1 = vdupq_n_f32(0.0f);
  float32x4_t sum_qjl0 = vdupq_n_f32(0.0f);
  float32x4_t sum_qjl1 = vdupq_n_f32(0.0f);

  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    uint8x8_t bytes = vld1_u8(packed + (i >> 1));
    uint8x8_t lo = vand_u8(bytes, mask_lo);
    uint8x8_t hi = vshr_n_u8(bytes, 4);
    uint8x8_t units = vzip1_u8(lo, hi);
    uint8x8_t sq = vshr_n_u8(units, 1);
    uint8x8_t qjl = vand_u8(units, mask_1);

    uint8x8_t sq_lo4 = sq;
    uint8x8_t sq_hi4 = vext_u8(sq, sq, 4);

    uint8x16_t idx0 = buildByteIdx4(sq_lo4);
    uint8x16_t idx1 = buildByteIdx4(sq_hi4);
    float32x4_t c0 = vreinterpretq_f32_u8(vqtbl2q_u8(lut, idx0));
    float32x4_t c1 = vreinterpretq_f32_u8(vqtbl2q_u8(lut, idx1));

    sum_ip0 = vmlaq_f32(sum_ip0, vld1q_f32(q_rot + i), c0);
    sum_ip1 = vmlaq_f32(sum_ip1, vld1q_f32(q_rot + i + 4), c1);

    int8x8_t signs = vsub_s8(vreinterpret_s8_u8(vadd_u8(qjl, qjl)),
                             vdup_n_s8(1));
    int16x8_t s16 = vmovl_s8(signs);
    uint32x4_t neg0 = vcltq_s32(vmovl_s16(vget_low_s16(s16)), vdupq_n_s32(0));
    uint32x4_t neg1 = vcltq_s32(vmovl_s16(vget_high_s16(s16)), vdupq_n_s32(0));
    sum_qjl0 =
        vaddq_f32(sum_qjl0, vreinterpretq_f32_u32(veorq_u32(
                                vreinterpretq_u32_f32(vld1q_f32(s_q + i)),
                                vandq_u32(neg0, sign_bit))));
    sum_qjl1 =
        vaddq_f32(sum_qjl1, vreinterpretq_f32_u32(veorq_u32(
                                vreinterpretq_u32_f32(vld1q_f32(s_q + i + 4)),
                                vandq_u32(neg1, sign_bit))));
  }

  float ip_mse = vaddvq_f32(vaddq_f32(sum_ip0, sum_ip1));
  float dot_qjl = vaddvq_f32(vaddq_f32(sum_qjl0, sum_qjl1));

  for (; i < dim; ++i) {
    uint8_t byte = packed[i >> 1];
    uint8_t unit = (i & 1) ? (byte >> 4) : (byte & 0x0F);
    ip_mse += q_rot[i] * centroids[unit >> 1];
    dot_qjl += s_q[i] * ((unit & 1) ? 1.0f : -1.0f);
  }

  ip_mse *= sigma;
  const float correction = space->scale() * gamma * dot_qjl;
  const float ip = (ip_mse + correction) * x_norm * pq->q_norm;
  return std::max(0.0f, pq->q_norm_sq + x_norm * x_norm - 2.0f * ip);
}

static float distBuildNEONB4(const void *pVect1, const void *pVect2,
                             const void *param_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
  const size_t dim = space->paddedDim();
  const size_t packed_bytes = space->packedBytes();
  const float *centroids = space->centroids();

  const char *buf_a = static_cast<const char *>(pVect1);
  const char *buf_b = static_cast<const char *>(pVect2);

  const uint8_t *packed_a = reinterpret_cast<const uint8_t *>(buf_a);
  const float *meta_a = reinterpret_cast<const float *>(buf_a + packed_bytes);
  const float norm_a = meta_a[0];
  const float sigma_a = meta_a[2];

  const uint8_t *packed_b = reinterpret_cast<const uint8_t *>(buf_b);
  const float *meta_b = reinterpret_cast<const float *>(buf_b + packed_bytes);
  const float norm_b = meta_b[0];
  const float sigma_b = meta_b[2];

  const uint8x16x2_t lut = loadCentroidLUTb4(centroids);
  const float32x4_t v_sigma_a = vdupq_n_f32(sigma_a);
  const float32x4_t v_sigma_b = vdupq_n_f32(sigma_b);
  const uint8x8_t mask_lo = vdup_n_u8(0x0F);

  float32x4_t sum = vdupq_n_f32(0.0f);

  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    uint8x8_t ba = vld1_u8(packed_a + (i >> 1));
    uint8x8_t bb = vld1_u8(packed_b + (i >> 1));

    uint8x8_t lo_a = vand_u8(ba, mask_lo);
    uint8x8_t hi_a = vshr_n_u8(ba, 4);
    uint8x8_t lo_b = vand_u8(bb, mask_lo);
    uint8x8_t hi_b = vshr_n_u8(bb, 4);

    uint8x8_t units_a = vzip1_u8(lo_a, hi_a);
    uint8x8_t units_b = vzip1_u8(lo_b, hi_b);
    uint8x8_t sq_a = vshr_n_u8(units_a, 1);
    uint8x8_t sq_b = vshr_n_u8(units_b, 1);

    uint8x8_t sq_a_hi = vext_u8(sq_a, sq_a, 4);
    uint8x8_t sq_b_hi = vext_u8(sq_b, sq_b, 4);

    float32x4_t ca0 = vreinterpretq_f32_u8(vqtbl2q_u8(lut, buildByteIdx4(sq_a)));
    float32x4_t ca1 = vreinterpretq_f32_u8(vqtbl2q_u8(lut, buildByteIdx4(sq_a_hi)));
    float32x4_t cb0 = vreinterpretq_f32_u8(vqtbl2q_u8(lut, buildByteIdx4(sq_b)));
    float32x4_t cb1 = vreinterpretq_f32_u8(vqtbl2q_u8(lut, buildByteIdx4(sq_b_hi)));

    float32x4_t va0 = vmulq_f32(ca0, v_sigma_a);
    float32x4_t va1 = vmulq_f32(ca1, v_sigma_a);
    float32x4_t vb0 = vmulq_f32(cb0, v_sigma_b);
    float32x4_t vb1 = vmulq_f32(cb1, v_sigma_b);

    sum = vmlaq_f32(sum, va0, vb0);
    sum = vmlaq_f32(sum, va1, vb1);
  }

  float ip_rot = vaddvq_f32(sum);
  for (; i < dim; ++i) {
    uint8_t ba = packed_a[i >> 1];
    uint8_t bb = packed_b[i >> 1];
    uint8_t ua = (i & 1) ? (ba >> 4) : (ba & 0x0F);
    uint8_t ub = (i & 1) ? (bb >> 4) : (bb & 0x0F);
    ip_rot += (centroids[ua >> 1] * sigma_a) * (centroids[ub >> 1] * sigma_b);
  }

  const float ip = ip_rot * norm_a * norm_b;
  return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
}

#endif // USE_NEON

// ---------------------------------------------------------------------------
// SSE implementation (4 floats per iteration)
// ---------------------------------------------------------------------------
#if defined(USE_SSE)

static float distSearchSSE(const void *q, const void *code_buf,
                           const void *qty_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(qty_ptr);
  const auto *pq = static_cast<const TurboQuantPreparedQuery *>(q);
  const size_t dim = space->paddedDim();
  const uint8_t *packed = reinterpret_cast<const uint8_t *>(code_buf);
  const float *meta = reinterpret_cast<const float *>(
      static_cast<const char *>(code_buf) + dim);
  const float x_norm = meta[0];
  const float gamma = meta[1];
  const float sigma = meta[2];

  const float *centroids = pq->centroids;
  const float *q_rot = pq->q_rot.data();
  const float *s_q = pq->s_q.data();

  const __m128i sign_bit = _mm_set1_epi32(static_cast<int>(0x80000000u));

  __m128 sum_ip = _mm_setzero_ps();
  __m128 sum_qjl = _mm_setzero_ps();

  size_t i = 0;
  for (; i + 4 <= dim; i += 4) {
    float cv[4];
    cv[0] = centroids[packed[i] >> 1];
    cv[1] = centroids[packed[i + 1] >> 1];
    cv[2] = centroids[packed[i + 2] >> 1];
    cv[3] = centroids[packed[i + 3] >> 1];
    sum_ip = _mm_add_ps(sum_ip,
                        _mm_mul_ps(_mm_loadu_ps(q_rot + i), _mm_loadu_ps(cv)));

    __m128i bytes =
        _mm_cvtsi32_si128(*reinterpret_cast<const int32_t *>(packed + i));
    bytes = _mm_unpacklo_epi8(bytes, bytes);
    bytes = _mm_unpacklo_epi16(bytes, bytes);
    __m128i bit0 = _mm_and_si128(bytes, _mm_set1_epi32(1));
    __m128i neg_mask = _mm_cmpeq_epi32(bit0, _mm_setzero_si128());

    __m128 vsq = _mm_loadu_ps(s_q + i);
    __m128i vsq_i =
        _mm_xor_si128(_mm_castps_si128(vsq), _mm_and_si128(neg_mask, sign_bit));
    sum_qjl = _mm_add_ps(sum_qjl, _mm_castsi128_ps(vsq_i));
  }

  float PORTABLE_ALIGN32 tmp[4];

  _mm_store_ps(tmp, sum_ip);
  float ip_mse = (tmp[0] + tmp[1] + tmp[2] + tmp[3]) * sigma;

  _mm_store_ps(tmp, sum_qjl);
  float dot_qjl = tmp[0] + tmp[1] + tmp[2] + tmp[3];

  for (; i < dim; ++i) {
    ip_mse += q_rot[i] * centroids[packed[i] >> 1] * sigma;
    dot_qjl += s_q[i] * ((packed[i] & 1) ? 1.0f : -1.0f);
  }

  const float correction = space->scale() * gamma * dot_qjl;
  const float ip = (ip_mse + correction) * x_norm * pq->q_norm;
  return std::max(0.0f, pq->q_norm_sq + x_norm * x_norm - 2.0f * ip);
}

static float distBuildSSE(const void *pVect1, const void *pVect2,
                          const void *param_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
  const size_t dim = space->paddedDim();
  const float *centroids = space->centroids();

  const char *buf_a = static_cast<const char *>(pVect1);
  const char *buf_b = static_cast<const char *>(pVect2);

  const uint8_t *packed_a = reinterpret_cast<const uint8_t *>(buf_a);
  const float *meta_a = reinterpret_cast<const float *>(buf_a + dim);
  const float norm_a = meta_a[0];
  const float sigma_a = meta_a[2];

  const uint8_t *packed_b = reinterpret_cast<const uint8_t *>(buf_b);
  const float *meta_b = reinterpret_cast<const float *>(buf_b + dim);
  const float norm_b = meta_b[0];
  const float sigma_b = meta_b[2];

  const __m128 v_sigma_a = _mm_set1_ps(sigma_a);
  const __m128 v_sigma_b = _mm_set1_ps(sigma_b);
  __m128 sum = _mm_setzero_ps();

  size_t i = 0;
  for (; i + 4 <= dim; i += 4) {
    float ca[4], cb[4];
    ca[0] = centroids[packed_a[i] >> 1];
    cb[0] = centroids[packed_b[i] >> 1];
    ca[1] = centroids[packed_a[i + 1] >> 1];
    cb[1] = centroids[packed_b[i + 1] >> 1];
    ca[2] = centroids[packed_a[i + 2] >> 1];
    cb[2] = centroids[packed_b[i + 2] >> 1];
    ca[3] = centroids[packed_a[i + 3] >> 1];
    cb[3] = centroids[packed_b[i + 3] >> 1];

    __m128 va = _mm_mul_ps(_mm_loadu_ps(ca), v_sigma_a);
    __m128 vb = _mm_mul_ps(_mm_loadu_ps(cb), v_sigma_b);
    sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
  }

  float PORTABLE_ALIGN32 tmp[4];
  _mm_store_ps(tmp, sum);
  float ip_rot = tmp[0] + tmp[1] + tmp[2] + tmp[3];

  for (; i < dim; ++i)
    ip_rot += (centroids[packed_a[i] >> 1] * sigma_a) *
              (centroids[packed_b[i] >> 1] * sigma_b);

  const float ip = ip_rot * norm_a * norm_b;
  return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
}

#endif // USE_SSE

// ---------------------------------------------------------------------------
// AVX2 implementation (8 floats per iteration, uses gather)
// ---------------------------------------------------------------------------
#if defined(USE_AVX)

static float distSearchAVX(const void *q, const void *code_buf,
                           const void *qty_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(qty_ptr);
  const auto *pq = static_cast<const TurboQuantPreparedQuery *>(q);
  const size_t dim = space->paddedDim();
  const uint8_t *packed = reinterpret_cast<const uint8_t *>(code_buf);
  const float *meta = reinterpret_cast<const float *>(
      static_cast<const char *>(code_buf) + dim);
  const float x_norm = meta[0];
  const float gamma = meta[1];
  const float sigma = meta[2];

  const float *centroids = pq->centroids;
  const float *q_rot = pq->q_rot.data();
  const float *s_q = pq->s_q.data();

  const __m256i sign_bit = _mm256_set1_epi32(static_cast<int>(0x80000000u));

  __m256 sum_ip = _mm256_setzero_ps();
  __m256 sum_qjl = _mm256_setzero_ps();

  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
#ifdef __AVX2__
    __m256i idx = _mm256_set_epi32(packed[i + 7] >> 1, packed[i + 6] >> 1,
                                   packed[i + 5] >> 1, packed[i + 4] >> 1,
                                   packed[i + 3] >> 1, packed[i + 2] >> 1,
                                   packed[i + 1] >> 1, packed[i] >> 1);
    __m256 vc = _mm256_i32gather_ps(centroids, idx, 4);
#else
    __m256 vc = _mm256_set_ps(
        centroids[packed[i + 7] >> 1], centroids[packed[i + 6] >> 1],
        centroids[packed[i + 5] >> 1], centroids[packed[i + 4] >> 1],
        centroids[packed[i + 3] >> 1], centroids[packed[i + 2] >> 1],
        centroids[packed[i + 1] >> 1], centroids[packed[i] >> 1]);
#endif
    sum_ip =
        _mm256_add_ps(sum_ip, _mm256_mul_ps(_mm256_loadu_ps(q_rot + i), vc));

    __m256 vsq = _mm256_loadu_ps(s_q + i);

#ifdef __AVX2__
    __m128i bytes8 =
        _mm_loadl_epi64(reinterpret_cast<const __m128i *>(packed + i));
    __m256i bytes32 = _mm256_cvtepu8_epi32(bytes8);
    __m256i bit0 = _mm256_and_si256(bytes32, _mm256_set1_epi32(1));
    __m256i neg_mask = _mm256_cmpeq_epi32(bit0, _mm256_setzero_si256());
#else
    __m256i neg_mask = _mm256_set_epi32(
        (packed[i + 7] & 1) ? 0 : -1, (packed[i + 6] & 1) ? 0 : -1,
        (packed[i + 5] & 1) ? 0 : -1, (packed[i + 4] & 1) ? 0 : -1,
        (packed[i + 3] & 1) ? 0 : -1, (packed[i + 2] & 1) ? 0 : -1,
        (packed[i + 1] & 1) ? 0 : -1, (packed[i] & 1) ? 0 : -1);
#endif

    __m256i vsq_i = _mm256_xor_si256(_mm256_castps_si256(vsq),
                                     _mm256_and_si256(neg_mask, sign_bit));
    sum_qjl = _mm256_add_ps(sum_qjl, _mm256_castsi256_ps(vsq_i));
  }

  __m128 hi_ip = _mm256_extractf128_ps(sum_ip, 1);
  __m128 lo_ip = _mm256_castps256_ps128(sum_ip);
  __m128 s_ip = _mm_add_ps(lo_ip, hi_ip);
  s_ip = _mm_add_ps(s_ip, _mm_movehl_ps(s_ip, s_ip));
  s_ip = _mm_add_ss(s_ip, _mm_shuffle_ps(s_ip, s_ip, 1));
  float ip_mse = _mm_cvtss_f32(s_ip) * sigma;

  __m128 hi_qjl = _mm256_extractf128_ps(sum_qjl, 1);
  __m128 lo_qjl = _mm256_castps256_ps128(sum_qjl);
  __m128 s_qjl = _mm_add_ps(lo_qjl, hi_qjl);
  s_qjl = _mm_add_ps(s_qjl, _mm_movehl_ps(s_qjl, s_qjl));
  s_qjl = _mm_add_ss(s_qjl, _mm_shuffle_ps(s_qjl, s_qjl, 1));
  float dot_qjl = _mm_cvtss_f32(s_qjl);

  for (; i < dim; ++i) {
    ip_mse += q_rot[i] * centroids[packed[i] >> 1] * sigma;
    dot_qjl += s_q[i] * ((packed[i] & 1) ? 1.0f : -1.0f);
  }

  const float correction = space->scale() * gamma * dot_qjl;
  const float ip = (ip_mse + correction) * x_norm * pq->q_norm;
  return std::max(0.0f, pq->q_norm_sq + x_norm * x_norm - 2.0f * ip);
}

static float distBuildAVX(const void *pVect1, const void *pVect2,
                          const void *param_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
  const size_t dim = space->paddedDim();
  const float *centroids = space->centroids();

  const char *buf_a = static_cast<const char *>(pVect1);
  const char *buf_b = static_cast<const char *>(pVect2);

  const uint8_t *packed_a = reinterpret_cast<const uint8_t *>(buf_a);
  const float *meta_a = reinterpret_cast<const float *>(buf_a + dim);
  const float norm_a = meta_a[0];
  const float sigma_a = meta_a[2];

  const uint8_t *packed_b = reinterpret_cast<const uint8_t *>(buf_b);
  const float *meta_b = reinterpret_cast<const float *>(buf_b + dim);
  const float norm_b = meta_b[0];
  const float sigma_b = meta_b[2];

  const __m256 v_sigma_a = _mm256_set1_ps(sigma_a);
  const __m256 v_sigma_b = _mm256_set1_ps(sigma_b);
  __m256 sum = _mm256_setzero_ps();

  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
#ifdef __AVX2__
    __m256i idx_a = _mm256_set_epi32(packed_a[i + 7] >> 1, packed_a[i + 6] >> 1,
                                     packed_a[i + 5] >> 1, packed_a[i + 4] >> 1,
                                     packed_a[i + 3] >> 1, packed_a[i + 2] >> 1,
                                     packed_a[i + 1] >> 1, packed_a[i] >> 1);
    __m256 va =
        _mm256_mul_ps(_mm256_i32gather_ps(centroids, idx_a, 4), v_sigma_a);

    __m256i idx_b = _mm256_set_epi32(packed_b[i + 7] >> 1, packed_b[i + 6] >> 1,
                                     packed_b[i + 5] >> 1, packed_b[i + 4] >> 1,
                                     packed_b[i + 3] >> 1, packed_b[i + 2] >> 1,
                                     packed_b[i + 1] >> 1, packed_b[i] >> 1);
    __m256 vb =
        _mm256_mul_ps(_mm256_i32gather_ps(centroids, idx_b, 4), v_sigma_b);
#else
    __m256 va = _mm256_mul_ps(
        _mm256_set_ps(
            centroids[packed_a[i + 7] >> 1], centroids[packed_a[i + 6] >> 1],
            centroids[packed_a[i + 5] >> 1], centroids[packed_a[i + 4] >> 1],
            centroids[packed_a[i + 3] >> 1], centroids[packed_a[i + 2] >> 1],
            centroids[packed_a[i + 1] >> 1], centroids[packed_a[i] >> 1]),
        v_sigma_a);
    __m256 vb = _mm256_mul_ps(
        _mm256_set_ps(
            centroids[packed_b[i + 7] >> 1], centroids[packed_b[i + 6] >> 1],
            centroids[packed_b[i + 5] >> 1], centroids[packed_b[i + 4] >> 1],
            centroids[packed_b[i + 3] >> 1], centroids[packed_b[i + 2] >> 1],
            centroids[packed_b[i + 1] >> 1], centroids[packed_b[i] >> 1]),
        v_sigma_b);
#endif
    sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
  }

  __m128 hi = _mm256_extractf128_ps(sum, 1);
  __m128 lo = _mm256_castps256_ps128(sum);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  float ip_rot = _mm_cvtss_f32(s);

  for (; i < dim; ++i)
    ip_rot += (centroids[packed_a[i] >> 1] * sigma_a) *
              (centroids[packed_b[i] >> 1] * sigma_b);

  const float ip = ip_rot * norm_a * norm_b;
  return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
}

#endif // USE_AVX

} // namespace turboquant
