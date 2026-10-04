#pragma once
// FastScan index for RaBitQ codes with the paper's two-stage search
// (RaBitQ SIGMOD 2024 §3.3.1 + Extended RaBitQ §4): an estimate with a
// confidence interval from the 1-bit signs, then exact re-scoring with the
// full code only where the interval cannot rule a candidate out.
//
// Stage 1 (FastScan): the sign bits of o' (for 4/8-bit codes: the MSB of each
// code, the same bit) in groups of 4 coordinates, LUT_g[mask] =
// sum_{j<4} (+-1) q'_{4g+j} / sqrt(D) ~> <o_bar_1, q'>; estimate
// est = score / <o_bar_1, o'> and the bound
//   eps = eps0 * sqrt(1 - df1^2) / df1 / sqrt(D - 1) + 3 sigma_LUT / df1,
//   lower = d_est - 2 ||x-c|| ||q-c|| eps.
// Stage 2: candidates in ascending `lower` are re-scored with the space's
// own distance on the full slot until lower >= the current k-th distance.
//
// Built from raw vectors (stage 1 needs <o_bar_1, o'>, which a 4/8-bit slot
// does not store). Keeps the full slots; the space must outlive the index.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "space.h"
#include "../common/fastscan.h"

namespace vsq::rabitq {

class RaBitQFastScan {
public:
    // X: row-major float32 [n, space.dim()].
    RaBitQFastScan(const RaBitQSpace &space, const float *X, size_t n,
                   float eps0 = RaBitQSpace::kDefaultEps0)
        : space_(&space), eps0_(eps0), isa_(space.kernelIsa()) {
        if (n > 0 && X == nullptr) throw std::invalid_argument("RaBitQFastScan: null data");
        const size_t D = space.paddedDim();
        const size_t dim = space.dim();
        cs_ = space.codeSizeBytes();
        slots_.resize(n * cs_);
        space.encodeBatch(X, n, slots_.data());
        xnorm_.resize(n);
        df1_.resize(n);
        signs_.assign(n * (D / 8), 0);
        std::vector<float> rot(D);
        for (size_t i = 0; i < n; ++i) {
            float norm = 0.0f;
            space.rotateResidual(X + i * dim, rot.data(), &norm);
            double acc = 0.0;
            uint8_t *sg = signs_.data() + i * (D / 8);
            for (size_t t = 0; t < D; ++t) {
                acc += std::fabs(rot[t]);
                if (rot[t] >= 0.0f) sg[t >> 3] = static_cast<uint8_t>(sg[t >> 3] | (1u << (t & 7)));
            }
            xnorm_[i] = norm;
            df1_[i] = norm > 0.0f ? static_cast<float>(acc / std::sqrt(static_cast<double>(D))) : 1.0f;
        }
        packed_ = common::fastscan::PackedCodes::pack(n, D / 4, [&](size_t i, size_t g) {
            const uint8_t b = signs_[i * (D / 8) + (g >> 1)];
            return static_cast<uint8_t>((g & 1) ? (b >> 4) : (b & 0x0F));
        });
    }

    size_t size() const { return packed_.n; }
    size_t dim() const { return space_->dim(); }
    const std::vector<uint8_t> &codes() const { return slots_; }

    // Stage-1 estimates and lower bounds for all n codes (est/lower: n floats).
    void scan(const float *q, float *est, float *lower) const {
        std::vector<float> rot(space_->paddedDim());
        float qnorm = 0.0f;
        space_->rotateResidual(q, rot.data(), &qnorm);
        stageOne(rot.data(), qnorm, est, lower);
    }

    // Top-k ids/dists (ascending) by the stage-2 (full-code) estimate.
    // Returns the number of codes re-scored in stage 2.
    size_t search(const float *q, size_t k, uint32_t *ids, float *dists) const {
        const size_t n = packed_.n;
        if (n == 0 || k == 0) return 0;
        std::vector<float> rot(space_->paddedDim()), est(n), lower(n);
        float qnorm = 0.0f;
        space_->rotateResidual(q, rot.data(), &qnorm);
        stageOne(rot.data(), qnorm, est.data(), lower.data());

        std::vector<uint8_t> pq(space_->querySizeBytes());
        space_->prepareQuery(q, pq.data());
        std::vector<uint32_t> order(n);
        for (size_t i = 0; i < n; ++i) order[i] = static_cast<uint32_t>(i);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return lower[a] < lower[b] || (lower[a] == lower[b] && a < b);
        });
        using Item = std::pair<float, uint32_t>;  // max-heap of the k best
        std::priority_queue<Item> heap;
        size_t refined = 0;
        for (uint32_t i : order) {
            if (heap.size() == k && lower[i] >= heap.top().first) break;
            const float d = space_->distancePrepared(pq.data(), slots_.data() + i * cs_);
            ++refined;
            if (heap.size() < k) {
                heap.emplace(d, i);
            } else if (d < heap.top().first) {
                heap.pop();
                heap.emplace(d, i);
            }
        }
        const size_t out = heap.size();
        for (size_t t = out; t-- > 0;) {
            ids[t] = heap.top().second;
            dists[t] = heap.top().first;
            heap.pop();
        }
        return refined;
    }

private:
    void stageOne(const float *rot, float qnorm, float *est, float *lower) const {
        const size_t D = space_->paddedDim();
        const size_t G = D / 4;
        const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
        std::vector<float> flut(G * 16);
        for (size_t g = 0; g < G; ++g)
            for (size_t mask = 0; mask < 16; ++mask) {
                float v = 0.0f;
                for (size_t j = 0; j < 4; ++j) v += ((mask >> j) & 1u ? 1.0f : -1.0f) * rot[4 * g + j];
                flut[g * 16 + mask] = v * inv_sqrt_d;
            }
        common::fastscan::QuantLut lut;
        lut.build(flut.data(), G, packed_.gpad);
        std::vector<uint32_t> sums(packed_.blocks * common::fastscan::kBlock);
        common::fastscan::scan(packed_, lut, sums.data(), isa_);
        const float lut_sigma3 = 3.0f * lut.delta * std::sqrt(static_cast<float>(G) / 12.0f);
        const float root = std::sqrt(static_cast<float>(D - 1));
        for (size_t i = 0; i < packed_.n; ++i) {
            const float df = df1_[i], xn = xnorm_[i];
            const float ip = lut.score(sums[i]) / df;
            const float d = std::max(0.0f, xn * xn + qnorm * qnorm - 2.0f * xn * qnorm * ip);
            const float eps = eps0_ * std::sqrt(std::max(0.0f, 1.0f - df * df)) / df / root +
                              lut_sigma3 / df;
            est[i] = d;
            lower[i] = std::max(0.0f, d - 2.0f * xn * qnorm * eps);
        }
    }

    const RaBitQSpace *space_;
    float eps0_;
    common::Isa isa_;
    size_t cs_ = 0;
    std::vector<uint8_t> slots_;  // [n, codeSizeBytes]
    std::vector<uint8_t> signs_;  // [n, D/8] sign bits of o'
    std::vector<float> xnorm_;    // [n] ||x - c||
    std::vector<float> df1_;      // [n] <o_bar_1, o'> = sum |o'_i| / sqrt(D)
    common::fastscan::PackedCodes packed_;
};

}  // namespace vsq::rabitq
