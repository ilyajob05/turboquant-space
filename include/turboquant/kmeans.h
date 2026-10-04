#pragma once
// kmeans.h — deterministic k-means for IVF centering (no BLAS dependency).
//
// Input X is row-major float32 [n, dim]. Training runs on a sample of at most
// k * max_points_per_centroid rows (FAISS uses the same cap idea), seeded by
// k-means++ and refined by Lloyd iterations. Empty clusters are re-seeded with
// the sample point farthest from its centroid. Results depend only on
// (X, params): the RNG is splitmix64 and every reduction has a fixed order.
//
// Cost per Lloyd iteration: O(n_sample * k * dim); with the default cap of 64
// points per centroid that is 64 * k^2 * dim flops (k = 256, dim = 1536:
// ~6.4 GFLOP), parallelised over points with OpenMP.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/config.h"
#include "../common/vecops.h"
#include "../common/srht.h"

namespace vsq::turboquant {

struct KMeansParams {
    size_t k = 256;                       // requested clusters (clamped to n_sample)
    int iters = 10;                       // Lloyd iterations after k-means++
    size_t max_points_per_centroid = 64;  // training sample cap: k * this rows
    uint64_t seed = 1234;
    int num_threads = 0;                  // 0 = OpenMP default
};

namespace detail {

// Nearest row of C [k, dim] to x: argmin_j ||c_j||^2 - 2 <x, c_j>.
// c_sqnorm[j] = ||c_j||^2. Ties resolve to the smallest j.
inline uint32_t nearestCentroid(const float *x, const float *C, const float *c_sqnorm,
                                size_t k, size_t dim, common::Isa isa, float *best_out = nullptr) {
    uint32_t best = 0;
    float best_v = std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < k; ++j) {
        const float v = c_sqnorm[j] - 2.0f * common::dotF32(x, C + j * dim, dim, isa);
        if (v < best_v) {
            best_v = v;
            best = static_cast<uint32_t>(j);
        }
    }
    if (best_out) *best_out = best_v;
    return best;
}

}  // namespace detail

// Returns centroids [k_eff, dim]; k_eff = min(k, n_sample) is written to *k_out.
inline std::vector<float> kmeansTrain(const float *X, size_t n, size_t dim,
                                      const KMeansParams &p, size_t *k_out,
                                      common::Isa isa = common::detectIsa()) {
    if (X == nullptr || n == 0) throw std::invalid_argument("kmeans: empty training set");
    if (dim == 0) throw std::invalid_argument("kmeans: dim must be >= 1");
    if (p.k == 0) throw std::invalid_argument("kmeans: k must be >= 1");
    if (p.iters < 0) throw std::invalid_argument("kmeans: iters must be >= 0");
    const int nt = common::resolveNumThreads(p.num_threads);
    (void)nt;
    common::RndGen64 rng(p.seed);

    // 1. Sample (partial Fisher–Yates over row indices).
    const size_t cap = p.k * std::max<size_t>(p.max_points_per_centroid, 1);
    const size_t ns = std::min(n, cap);
    std::vector<size_t> rows(n);
    for (size_t i = 0; i < n; ++i) rows[i] = i;
    if (ns < n) {
        for (size_t i = 0; i < ns; ++i) {
            const size_t j = i + static_cast<size_t>(rng.next() % (n - i));
            std::swap(rows[i], rows[j]);
        }
    }
    rows.resize(ns);
    std::sort(rows.begin(), rows.end());  // sequential access into X
    std::vector<float> S(ns * dim);
    for (size_t i = 0; i < ns; ++i)
        std::copy(X + rows[i] * dim, X + rows[i] * dim + dim, S.begin() + i * dim);

    const size_t k = std::min(p.k, ns);
    std::vector<float> C(k * dim);
    std::vector<float> csq(k);

    // 2. k-means++ seeding: next centre drawn with probability ∝ D(x)^2.
    std::vector<float> d2(ns, std::numeric_limits<float>::infinity());
    const size_t first = static_cast<size_t>(rng.next() % ns);
    std::copy(S.begin() + first * dim, S.begin() + first * dim + dim, C.begin());
    for (size_t c = 1; c < k; ++c) {
        const float *last = C.data() + (c - 1) * dim;
        VSQ_OMP_PARALLEL_FOR(nt, ns)
        for (long long ii = 0; ii < static_cast<long long>(ns); ++ii) {
            const size_t i = static_cast<size_t>(ii);
            d2[i] = std::min(d2[i], common::l2sqF32(S.data() + i * dim, last, dim, isa));
        }
        double total = 0.0;
        for (size_t i = 0; i < ns; ++i) total += d2[i];
        size_t pick = ns - 1;
        if (total > 0.0) {
            const double u = (static_cast<double>(rng.next() >> 11) * 0x1.0p-53) * total;
            double acc = 0.0;
            for (size_t i = 0; i < ns; ++i) {
                acc += d2[i];
                if (acc >= u) {
                    pick = i;
                    break;
                }
            }
        } else {
            pick = static_cast<size_t>(rng.next() % ns);  // every point is a centre
        }
        std::copy(S.begin() + pick * dim, S.begin() + pick * dim + dim, C.begin() + c * dim);
    }

    // 3. Lloyd iterations.
    std::vector<uint32_t> assign(ns);
    std::vector<float> best_v(ns);
    std::vector<double> sums(k * dim);
    std::vector<size_t> counts(k);
    for (int it = 0; it < p.iters; ++it) {
        for (size_t j = 0; j < k; ++j) csq[j] = common::normSqF32(C.data() + j * dim, dim, isa);
        VSQ_OMP_PARALLEL_FOR(nt, ns)
        for (long long ii = 0; ii < static_cast<long long>(ns); ++ii) {
            const size_t i = static_cast<size_t>(ii);
            assign[i] = detail::nearestCentroid(S.data() + i * dim, C.data(), csq.data(), k,
                                                dim, isa, &best_v[i]);
        }
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < ns; ++i) {  // fixed order: deterministic
            const uint32_t a = assign[i];
            ++counts[a];
            const float *x = S.data() + i * dim;
            double *s = sums.data() + static_cast<size_t>(a) * dim;
            for (size_t t = 0; t < dim; ++t) s[t] += x[t];
        }
        for (size_t j = 0; j < k; ++j) {
            if (counts[j] == 0) {
                // Re-seed with the point farthest from its centroid
                // (best_v + ||x||^2 = squared distance).
                size_t far = 0;
                float far_d = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < ns; ++i) {
                    const float dd = best_v[i] + common::normSqF32(S.data() + i * dim, dim, isa);
                    if (dd > far_d) {
                        far_d = dd;
                        far = i;
                    }
                }
                std::copy(S.begin() + far * dim, S.begin() + far * dim + dim,
                          C.begin() + j * dim);
                best_v[far] = -std::numeric_limits<float>::infinity();  // not reused
                continue;
            }
            const double inv = 1.0 / static_cast<double>(counts[j]);
            for (size_t t = 0; t < dim; ++t)
                C[j * dim + t] = static_cast<float>(sums[j * dim + t] * inv);
        }
    }
    if (k_out) *k_out = k;
    return C;
}

}  // namespace vsq::turboquant
