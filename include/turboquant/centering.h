#pragma once
// centering.h — what a vector is quantized relative to (review item 2.6).
//
//   None  c = 0. No training.
//   Mean  c = dataset mean (one centroid). train() computes it.
//   Ivf   c = nearest of k k-means centroids; the centroid id is stored in
//         the code. train() runs kmeans.h.
//
// Quantization error scales with ||x - c||, while neighbour distances are
// much smaller than ||x|| for real embeddings (they share a large common
// component), so centring buys accuracy for free at query time: the query
// side only needs ||q - c_j||^2 for the k centroids (a k-float table).
//
// Centroids are stored in the original (unrotated) space, row-major [k, dim].

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/config.h"
#include "../common/vecops.h"
#include "kmeans.h"

namespace vsq::turboquant {

enum class Centering : uint8_t { None = 0, Mean = 1, Ivf = 2 };

inline const char *centeringName(Centering c) {
    switch (c) {
    case Centering::Mean: return "mean";
    case Centering::Ivf: return "ivf";
    default: return "none";
    }
}

class Centerer {
public:
    // Cluster ids are stored as uint16 in codes.
    static constexpr size_t kMaxClusters = 65535;
    // Fewer training points per centroid than this degrades k-means; train()
    // lowers k to n / kMinPointsPerCentroid (at least 1).
    static constexpr size_t kMinPointsPerCentroid = 39;

    Centerer(Centering mode, size_t dim, size_t n_clusters, common::Isa isa = common::detectIsa())
        : mode_(mode), dim_(dim), requested_k_(n_clusters), isa_(isa) {
        if (dim == 0) throw std::invalid_argument("Centerer: dim must be >= 1");
        if (mode == Centering::Ivf && (n_clusters == 0 || n_clusters > kMaxClusters))
            throw std::invalid_argument("Centerer: n_clusters must be in [1, 65535], got " +
                                        std::to_string(n_clusters));
    }

    Centering mode() const { return mode_; }
    size_t dim() const { return dim_; }
    size_t requestedClusters() const { return requested_k_; }
    // 0 before training for Mean/Ivf; None never needs training.
    size_t numCentroids() const { return k_; }
    bool needsTraining() const { return mode_ != Centering::None; }
    bool trained() const { return mode_ == Centering::None || k_ > 0; }
    const std::vector<float> &centroids() const { return centroids_; }

    // X: row-major float32 [n, dim], n >= 1.
    void train(const float *X, size_t n, uint64_t seed = 1234, int num_threads = 0,
               int iters = 10) {
        if (X == nullptr || n == 0) throw std::invalid_argument("Centerer::train: empty data");
        if (mode_ == Centering::None) return;
        if (mode_ == Centering::Mean) {
            std::vector<double> acc(dim_, 0.0);
            for (size_t i = 0; i < n; ++i)
                for (size_t t = 0; t < dim_; ++t) acc[t] += X[i * dim_ + t];
            std::vector<float> mu(dim_);
            for (size_t t = 0; t < dim_; ++t)
                mu[t] = static_cast<float>(acc[t] / static_cast<double>(n));
            setCentroids(mu.data(), 1);
            return;
        }
        KMeansParams p;
        p.k = std::max<size_t>(1, std::min(requested_k_, n / kMinPointsPerCentroid));
        p.iters = iters;
        p.seed = seed;
        p.num_threads = num_threads;
        size_t k = 0;
        std::vector<float> C = kmeansTrain(X, n, dim_, p, &k, isa_);
        setCentroids(C.data(), k);
    }

    // C: row-major float32 [k, dim]. Mean requires k == 1.
    void setCentroids(const float *C, size_t k) {
        if (mode_ == Centering::None) throw std::invalid_argument("Centerer: centering is none");
        if (C == nullptr || k == 0) throw std::invalid_argument("Centerer: empty centroids");
        if (mode_ == Centering::Mean && k != 1)
            throw std::invalid_argument("Centerer: mean centering takes exactly 1 centroid");
        if (k > kMaxClusters) throw std::invalid_argument("Centerer: more than 65535 centroids");
        centroids_.assign(C, C + k * dim_);
        sqnorm_.resize(k);
        for (size_t j = 0; j < k; ++j) sqnorm_[j] = common::normSqF32(C + j * dim_, dim_, isa_);
        k_ = k;
    }

    // Cluster of x (dim floats); 0 for None/Mean.
    uint32_t assign(const float *x) const {
        if (mode_ != Centering::Ivf) return 0;
        return detail::nearestCentroid(x, centroids_.data(), sqnorm_.data(), k_, dim_, isa_);
    }

    // Centroid j (dim floats), or nullptr for None (the zero vector).
    const float *centroid(uint32_t j) const {
        return mode_ == Centering::None ? nullptr
                                        : centroids_.data() + static_cast<size_t>(j) * dim_;
    }

    // out[j] = ||q - c_j||^2 for j < max(k, 1); None writes ||q||^2.
    void queryTable(const float *q, float *out) const {
        const float qq = common::normSqF32(q, dim_, isa_);
        if (mode_ == Centering::None) {
            out[0] = qq;
            return;
        }
        for (size_t j = 0; j < k_; ++j)
            out[j] = qq + sqnorm_[j] - 2.0f * common::dotF32(q, centroids_.data() + j * dim_, dim_, isa_);
    }

private:
    Centering mode_;
    size_t dim_;
    size_t requested_k_;
    common::Isa isa_;
    size_t k_ = 0;
    std::vector<float> centroids_;  // [k, dim]
    std::vector<float> sqnorm_;     // [k]
};

}  // namespace vsq::turboquant
