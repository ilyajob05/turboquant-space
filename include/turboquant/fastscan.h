#pragma once
// FastScan index over TurboQuant v2 4-bit codes (non-QJL), for flat 1-to-N
// scans. HNSW keeps using the per-pair kernels; this is the batch layout.
//
// Group g = coordinate g, LUT_g[k] = (R q')_g * centroid[k], so the FastScan
// score approximates <Rq', dec> and the distance is assembled exactly as in
// TurboQuantSpace::distance (d = T[cid] + f_sq - 2 (f_mul * ip - f_ct)).
// search() takes the k * rerank best FastScan candidates and re-scores them
// with the float LUT (the per-pair estimate, without LUT quantization).
//
// The index copies what it needs (packed sub-codes + per-code factors);
// the space must outlive it.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "space.h"
#include "../common/fastscan.h"

namespace vsq::turboquant {

class TurboQuantFastScan {
public:
    // codes: n * space.codeSizeBytes() bytes encoded by `space`.
    TurboQuantFastScan(const TurboQuantSpace &space, const void *codes, size_t n)
        : space_(&space), isa_(space.kernelIsa()) {
        if (space.bits() != 4 || space.qjl())
            throw std::invalid_argument("TurboQuantFastScan: needs a 4-bit space without qjl");
        if (n > 0 && codes == nullptr) throw std::invalid_argument("TurboQuantFastScan: null codes");
        const auto *base = static_cast<const uint8_t *>(codes);
        const size_t cs = space.codeSizeBytes();
        const size_t D = space.paddedDim();
        const bool ivf = space.centering() == Centering::Ivf;
        packed_ = common::fastscan::PackedCodes::pack(n, D, [&](size_t i, size_t g) {
            const uint8_t b = base[i * cs + (g >> 1)];
            return static_cast<uint8_t>((g & 1) ? (b >> 4) : (b & 0x0F));
        });
        f_sq_.resize(n);
        f_mul_.resize(n);
        f_ct_.assign(n, 0.0f);
        cid_.assign(n, 0);
        for (size_t i = 0; i < n; ++i) {
            const uint8_t *c = base + i * cs;
            f_sq_[i] = common::loadUnaligned<float>(c + space.offsetSq());
            f_mul_[i] = common::loadUnaligned<float>(c + space.offsetMul());
            if (ivf) {
                f_ct_[i] = common::loadUnaligned<float>(c + space.offsetCt());
                cid_[i] = common::loadUnaligned<uint16_t>(c + space.offsetCid());
            }
        }
    }

    size_t size() const { return packed_.n; }
    size_t dim() const { return space_->dim(); }

    // Approximate distances of q (dim floats) to all n codes -> out[n].
    void scan(const float *q, float *out) const {
        TurboQuantQuery pq;
        space_->prepareQuery(q, pq);
        scanPrepared(pq, out);
    }

    // Top-k ids/dists (ascending). rerank >= 1: candidates = k * rerank.
    // ids/dists hold min(k, n) entries.
    void search(const float *q, size_t k, size_t rerank, uint32_t *ids, float *dists) const {
        const size_t n = packed_.n;
        if (n == 0 || k == 0) return;
        TurboQuantQuery pq;
        space_->prepareQuery(q, pq);
        std::vector<float> approx(n);
        scanPrepared(pq, approx.data());
        const size_t cand = std::min(n, k * std::max<size_t>(rerank, 1));
        std::vector<uint32_t> top = common::fastscan::smallestK(approx.data(), n, cand);
        std::vector<float> exact(top.size());
        const float *table = space_->decodeTable().data();
        for (size_t t = 0; t < top.size(); ++t) {
            const uint32_t i = top[t];
            float ip = 0.0f;
            for (size_t g = 0; g < packed_.groups; ++g) ip += pq.qrot[g] * table[packed_.subcode(i, g)];
            exact[t] = assemble(pq, i, ip);
        }
        std::vector<uint32_t> order = common::fastscan::smallestK(exact.data(), exact.size(), k);
        for (size_t t = 0; t < order.size(); ++t) {
            ids[t] = top[order[t]];
            dists[t] = exact[order[t]];
        }
    }

private:
    float assemble(const TurboQuantQuery &pq, size_t i, float ip) const {
        const bool ivf = space_->centering() == Centering::Ivf;
        const float base = ivf ? pq.table[cid_[i]] : pq.table[0];
        return std::max(0.0f, base + f_sq_[i] - 2.0f * (f_mul_[i] * ip - f_ct_[i]));
    }

    void scanPrepared(const TurboQuantQuery &pq, float *out) const {
        const size_t G = packed_.groups;
        const float *table = space_->decodeTable().data();
        std::vector<float> flut(G * 16);
        for (size_t g = 0; g < G; ++g)
            for (size_t k = 0; k < 16; ++k) flut[g * 16 + k] = pq.qrot[g] * table[k];
        common::fastscan::QuantLut lut;
        lut.build(flut.data(), G, packed_.gpad);
        std::vector<uint32_t> sums(packed_.blocks * common::fastscan::kBlock);
        common::fastscan::scan(packed_, lut, sums.data(), isa_);
        for (size_t i = 0; i < packed_.n; ++i) out[i] = assemble(pq, i, lut.score(sums[i]));
    }

    const TurboQuantSpace *space_;
    common::Isa isa_;
    common::fastscan::PackedCodes packed_;
    std::vector<float> f_sq_, f_mul_, f_ct_;
    std::vector<uint16_t> cid_;
};

}  // namespace vsq::turboquant
