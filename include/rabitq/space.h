#pragma once

// RaBitQ (Gao & Long, SIGMOD 2024) and Extended RaBitQ
// (Gao, Gou, Xu, Yang, Liu, Long; arXiv:2409.09913). Header-only.
// Asymmetric squared L2 (prepared query x code). Not a code-to-code score,
// so it must not be used as the HNSW link-construction metric.
//
// bits is 1, 4 (default), or 8. Coordinates [dim, D) are 0 before the rotation.
// Rotation (rotation.h):
//   BlockKac (default)  D = dim rounded up to a multiple of 64, 3 rounds of
//                       signs + overlapping Walsh–Hadamard blocks + Kac step
//   LegacySrht          D = roundUpPow2(max(dim, 4)), one SRHT round —
//                       reproduces codes written by turboquant-space 0.1.x
//
// Slot, little-endian. The two float32 tails are copied with memcpy because
// the payload length is not always a multiple of 4.
//   bits 1: uint8 signs[(D + 7) / 8]
//           bit i set => coordinate +1/sqrt(D), else -1/sqrt(D)
//   bits 4: uint8 nibbles[D / 2]
//           low nibble = even index, high nibble = odd index, value 0..15
//   bits 8: uint8 codes[D], value 0..255
//   float32 norm_to_centroid     ||x - c|| in the original coordinates
//   float32 dot_factor           <y, o'>
//           1-bit: y_i = ±1/sqrt(D)
//           4/8-bit: y_i = code_i - (2^bits - 1) / 2
//                    (the centered grid of eq. 7; ||y|| cancels in the ratio
//                    and is not stored)
//   x == c is encoded as norm 0, dot_factor 1: its distance is exactly ||q - c||^2.
//
// Prepared query (querySizeBytes()):
//   float query (4/8-bit, or 1-bit with query_bits = 0), (D + 2) float32:
//     rotated unit residual q' [D], ||q - c||, sum of q'.
//     4/8-bit inner product is <code, q'> - center * sum(q'), which is eq. 12.
//   quantized query (1-bit, query_bits = B_q > 0, the paper's default B_q = 4):
//     bitwise.h header (4 float32) + B_q bit-planes of D/64 uint64;
//     the distance is (B_q + 1) * D/64 popcounts per code.
//
// Kernels are chosen once in the constructor: AVX2 by CPUID (runtime
// dispatch; the TU may be compiled for baseline x86-64), NEON on aarch64,
// scalar otherwise. Distance kernels never throw and return values >= 0.
//
// Error bound (RaBitQ Theorem 3.2): with probability >= 1 - delta,
//   |<o,q'> - est| <= eps0 * sqrt((1 - <o_bar,o'>^2) / <o_bar,o'>^2) / sqrt(D - 1)
// distanceBound() turns it into [lower, upper] for the squared distance
// (eps0 = 1.9 by default; quantized-query error is not included).
//
// 4/8-bit encode_mode: how the grid scale t (codes = round(t * o') clamped
// to the grid) is chosen. All modes write the same slot and use the same
// distance kernel; they differ only in encode cost and code quality. The
// constructor default is -1, resolved by defaultEncodeMode(bits), the most
// accurate mode per unit of encode cost measured on N(0,1) at dim
// 128/768/1024 and dbpedia-1536 (docs/benchmarks.md, 2026-10-06):
//   1-bit  algorithm1      the sign code has no scale
//   4-bit  fixed_scale     recall@10 equal to windowed_scale within noise
//                          (|diff| <= 0.005), ~10x faster encode
//   8-bit  windowed_scale  fixed_scale loses 0.001-0.015 recall@10 here
//   0 algorithm1      per vector, sort every threshold. Bit-exact Extended
//                     RaBitQ Algorithm 1; O(D log D). Reference accuracy.
//   1 fixed_scale     static: one t frozen for the whole space, O(1) per
//                     coordinate. t is the mean Algorithm 1 plateau edge
//                     over 100 N(0,1) residuals at seed 42 (not rot_seed).
//                     Fastest encode; at 8 bits recall is lower.
//                     The 4-bit default.
//   2 windowed_scale  per vector, inside the RaBitQ-Library tight interval,
//                     chosen by a min-heap of the next magnitude event.
//                     Same accuracy as algorithm1, several times faster.
//                     The 8-bit default.
//   3 trained_scale   like fixed_scale, but t is calibrated by train(X) on
//                     the residuals x - c of the data itself. encode before
//                     train() throws. After the random rotation residual
//                     coordinates are near N(0, 1/D) for any data, so t (and
//                     accuracy) is within ~1% of fixed_scale.
// fixed_scale, windowed_scale and trained_scale require bits 4 or 8.
//
// Centroid. Codes quantize the residual x - c. c is the constructor
// argument (zero by default) or, after train(X), the mean of X, in every
// mode. On real embeddings, which are far from zero-mean, centering is the
// largest accuracy factor (recall@10 0.94 -> 0.97 at 4 bits on dbpedia).

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../common/config.h"
#include "bitwise.h"
#include "kernels.h"
#include "../common/rotation.h"
#include "../common/srht.h"

// Namespaces follow include/ directories: RaBitQ lives in rabitq::, shared
// code is always called with an explicit common:: qualifier.
namespace vsq::rabitq {

// One space: fixed centroid, rotation, bit width and D.
class RaBitQSpace {
public:
    using DistFunc = float (*)(const void *, const void *, const void *);
    static constexpr float kDefaultEps0 = 1.9f;  // RaBitQ confidence constant

    // Encode modes (see the file header). -1 resolves to defaultEncodeMode(bits).
    enum EncodeMode : int { Algorithm1 = 0, FixedScale = 1, WindowedScale = 2, TrainedScale = 3 };
    static constexpr int kDefaultBits = 4;

    // Default encode mode for a bit width in {1, 4, 8} (see the file header).
    static constexpr int defaultEncodeMode(int bits) {
        return bits == 1 ? Algorithm1 : (bits == 4 ? FixedScale : WindowedScale);
    }
    // Rows of X used by train() to calibrate trained_scale (even stride).
    static constexpr size_t kTrainCalibRows = 1024;

    // centroid == nullptr means the zero vector of length `dim` (copied);
    // train(X) replaces it with the mean of X.
    // encode_mode -1/0/1/2/3: default, algorithm1, fixed_scale,
    // windowed_scale, trained_scale (needs train() before encode).
    // query_bits -1: default (1-bit: 4, the paper's B_q; 4/8-bit: 0 = float
    // query). 1-bit accepts 0..8; 4/8-bit accept 0 only.
    RaBitQSpace(size_t dim, uint64_t rot_seed, const float *centroid, int bits = kDefaultBits,
                int encode_mode = -1, common::RotationKind rotation = common::RotationKind::BlockKac,
                int rotation_rounds = 3, int query_bits = -1, int num_threads = 0,
                common::Isa isa = common::detectIsa())
        : dim_(checkedDim(dim)),
          bits_(bits),
          rot_seed_(rot_seed),
          rot_(dim, rot_seed, rotation_rounds, rotation, isa),
          padded_(rot_.paddedDim()),
          centroid_(dim, 0.0f),
          inv_sqrt_d_(1.0f / std::sqrt(static_cast<float>(padded_))),
          center_(bits >= 8 ? 127.5f : (bits >= 4 ? 7.5f : 0.5f)),
          encode_mode_(encode_mode),
          t_fixed_(0.0),
          isa_(common::resolveIsa(isa)),
          num_threads_(common::resolveNumThreads(num_threads)) {
        if (bits_ != 1 && bits_ != 4 && bits_ != 8)
            throw std::invalid_argument("RaBitQ: bits must be 1, 4, or 8");
        if (encode_mode_ < -1 || encode_mode_ > TrainedScale)
            throw std::invalid_argument(
                "RaBitQ: encode_mode must be -1, 0, 1, 2, or 3");
        if (encode_mode_ < 0)
            encode_mode_ = defaultEncodeMode(bits_);
        if (encode_mode_ != Algorithm1 && bits_ == 1)
            throw std::invalid_argument(
                "RaBitQ: fixed_scale, windowed_scale and trained_scale require bits 4 or 8");
        query_bits_ = resolveQueryBits(query_bits);
        if (centroid != nullptr)
            std::copy(centroid, centroid + dim_, centroid_.begin());
        dist_func_ = selectDist();
        bitwise_ = bitwise::selectBitwise(isa_);
        if (encode_mode_ == FixedScale)
            t_fixed_ = calibrateFixedScale();
    }

    // Fits the space to data. X: row-major float32 [n, dim], n >= 1, finite.
    // Sets the centroid to the mean of X (every mode), and for trained_scale
    // calibrates the frozen scale on the residuals x - c of up to
    // kTrainCalibRows rows. Not thread-safe against concurrent encode/search;
    // codes written before train() use the old centroid and must be re-encoded.
    void train(const float *X, size_t n) {
        if (X == nullptr || n == 0)
            throw std::invalid_argument("RaBitQ train: empty data");
        std::vector<double> acc(dim_, 0.0);
        for (size_t r = 0; r < n; ++r) {
            const float *x = X + r * dim_;
            for (size_t i = 0; i < dim_; ++i) acc[i] += x[i];
        }
        std::vector<float> mean(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            mean[i] = static_cast<float>(acc[i] / static_cast<double>(n));
            if (!std::isfinite(mean[i]))
                throw std::invalid_argument("RaBitQ train: input is not finite");
        }
        centroid_ = std::move(mean);
        if (encode_mode_ == TrainedScale)
            t_fixed_ = calibrateTrainedScale(X, n);
        trained_ = true;
    }
    // True once train() has run.
    bool trained() const { return trained_; }

    size_t dim() const { return dim_; }
    size_t paddedDim() const { return padded_; }
    int bits() const { return bits_; }
    uint64_t rotSeed() const { return rot_seed_; }
    int queryBits() const { return query_bits_; }
    common::Isa kernelIsa() const { return isa_; }
    int rotationRounds() const { return rot_.rounds(); }
    common::RotationKind rotationKind() const { return rot_.kind(); }
    const std::vector<float> &centroid() const { return centroid_; }
    // "algorithm1", "fixed_scale", "windowed_scale", or "trained_scale".
    const char *encodeModeName() const {
        switch (encode_mode_) {
            case FixedScale: return "fixed_scale";
            case WindowedScale: return "windowed_scale";
            case TrainedScale: return "trained_scale";
            default: return "algorithm1";
        }
    }
    // Frozen scale of fixed_scale / trained_scale (0 before train() for the
    // latter). 0 for algorithm1 and windowed_scale.
    double fixedScale() const { return t_fixed_; }

    // Bytes of one data slot.
    size_t codeSizeBytes() const { return payloadBytes() + 2 * sizeof(float); }

    // Bytes of one prepared query.
    size_t querySizeBytes() const {
        return bitwiseQuery() ? bitwise::preparedBytes(padded_, query_bits_)
                              : (padded_ + 2) * sizeof(float);
    }

    // HNSWLIB SpaceInterface shape (non-virtual: the header does not include
    // hnswlib; a copy next to space_l2.h can inherit and forward).
    size_t get_data_size() { return codeSizeBytes(); }
    DistFunc get_dist_func() { return dist_func_; }
    void *get_dist_func_param() { return this; }

    // Writes one slot. x has dim() floats. Throws only for null pointers or a
    // non-finite input; x == c is a valid input (see the slot layout).
    void encode(const float *x, void *slot) const {
        if (x == nullptr || slot == nullptr)
            throw std::invalid_argument("RaBitQ encode: null pointer");
        if (encode_mode_ == TrainedScale && !trained_)
            throw std::runtime_error("RaBitQ encode: trained_scale needs train(X) first");
        float *rotated = scratch();
        float norm = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            const float v = x[i] - centroid_[i];
            rotated[i] = v;
            norm += v * v;
        }
        std::memset(rotated + dim_, 0, (padded_ - dim_) * sizeof(float));
        norm = std::sqrt(norm);
        if (!std::isfinite(norm))
            throw std::invalid_argument("RaBitQ encode: input is not finite");
        auto *bytes = static_cast<uint8_t *>(slot);
        if (!(norm > 0.0f)) {
            writeCenterSlot(bytes);
            return;
        }
        const float inv = 1.0f / norm;
        for (size_t i = 0; i < dim_; ++i)
            rotated[i] *= inv;
        rotateUnit(rotated);

        if (bits_ == 1) {
            const float dot = dotWithCube(rotated);
            std::memset(bytes, 0, payloadBytes());
            for (size_t i = 0; i < padded_; ++i) {
                if (rotated[i] >= 0.0f)
                    bytes[i >> 3] = static_cast<uint8_t>(
                        bytes[i >> 3] | (1u << (i & 7u)));
            }
            std::memcpy(bytes + payloadBytes(), &norm, sizeof(float));
            std::memcpy(bytes + payloadBytes() + sizeof(float), &dot,
                        sizeof(float));
            return;
        }

        static thread_local std::vector<uint8_t> codes;
        if (codes.size() < padded_) codes.resize(padded_);
        float dot = 0.0f;
        if (encode_mode_ == FixedScale || encode_mode_ == TrainedScale)
            dot = quantizeFixedScale(rotated, codes.data());
        else if (encode_mode_ == WindowedScale)
            dot = quantizeWindowedScale(rotated, codes.data());
        else
            dot = quantizeExtended(rotated, codes.data());
        packCodes(codes.data(), norm, dot, bytes);
    }

    // Row-major [n, dim] into n packed slots, OpenMP-parallel above 64 rows.
    // The first exception thrown by any row is rethrown after the loop.
    void encodeBatch(const float *raws, size_t n, void *out) const {
        if (out == nullptr || (n > 0 && raws == nullptr))
            throw std::invalid_argument("RaBitQ encodeBatch: null pointer");
        auto *bytes = static_cast<uint8_t *>(out);
        const size_t stride = codeSizeBytes();
        std::exception_ptr error;
        std::mutex error_mutex;
        VSQ_OMP_PARALLEL_FOR(num_threads_, n)
        for (long long ii = 0; ii < static_cast<long long>(n); ++ii) {
            const size_t i = static_cast<size_t>(ii);
            try {
                encode(raws + i * dim_, bytes + i * stride);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
        }
        if (error) std::rethrow_exception(error);
    }

    // `out` must hold querySizeBytes(). q has length dim().
    void prepareQuery(const float *q, void *out) const {
        if (q == nullptr || out == nullptr)
            throw std::invalid_argument("RaBitQ prepareQuery: null pointer");
        float *rotated = scratch();
        float norm = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            const float v = q[i] - centroid_[i];
            rotated[i] = v;
            norm += v * v;
        }
        std::memset(rotated + dim_, 0, (padded_ - dim_) * sizeof(float));
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            const float inv = 1.0f / norm;
            for (size_t i = 0; i < dim_; ++i)
                rotated[i] *= inv;
            rotateUnit(rotated);
        } else {
            std::memset(rotated, 0, padded_ * sizeof(float));
        }
        if (bitwiseQuery()) {
            bitwise::quantizeQueryBitplanes(rotated, padded_, query_bits_,
                                                               norm, kQuerySeed, out);
            return;
        }
        float sum_q = 0.0f;
        for (size_t i = 0; i < padded_; ++i)
            sum_q += rotated[i];
        auto *dst = static_cast<float *>(out);
        std::memcpy(dst, rotated, padded_ * sizeof(float));
        std::memcpy(dst + padded_, &norm, sizeof(float));
        std::memcpy(dst + padded_ + 1, &sum_q, sizeof(float));
    }

    // Rotated unit residual R (x - c) / ||x - c|| into out (paddedDim()
    // floats) and ||x - c|| into *norm; zeros when x == c. x has dim() floats.
    void rotateResidual(const float *x, float *out, float *norm) const {
        float nsq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            out[i] = x[i] - centroid_[i];
            nsq += out[i] * out[i];
        }
        std::memset(out + dim_, 0, (padded_ - dim_) * sizeof(float));
        *norm = std::sqrt(nsq);
        if (!(*norm > 0.0f)) {
            std::memset(out, 0, padded_ * sizeof(float));
            return;
        }
        const float inv = 1.0f / *norm;
        for (size_t i = 0; i < dim_; ++i) out[i] *= inv;
        rotateUnit(out);
    }

    // First pointer: prepared query. Second: code slot.
    float distancePrepared(const void *prepared, const void *slot) const {
        return dist_func_(prepared, slot, this);
    }

    // Raw query of length dim(). Uses get_dist_func(), so the HNSW pointer
    // and this path cannot drift.
    float distanceRaw(const float *q, const void *slot) const {
        uint8_t *prepared = queryScratch();
        prepareQuery(q, prepared);
        return dist_func_(prepared, slot, this);
    }

    // Same estimate with the scalar kernel (tests compare every ISA to it).
    float distancePreparedScalar(const void *prepared, const void *slot) const {
        if (bitwiseQuery())
            return distBitwiseWith(prepared, slot, &bitwise::bitwiseScalar);
        if (bits_ == 1)
            return distKernel<1, detail::DotIsa::Scalar>(prepared, slot, this);
        if (bits_ == 4)
            return distKernel<4, detail::DotIsa::Scalar>(prepared, slot, this);
        return distKernel<8, detail::DotIsa::Scalar>(prepared, slot, this);
    }

    float distanceRawScalar(const float *q, const void *slot) const {
        uint8_t *prepared = queryScratch();
        prepareQuery(q, prepared);
        return distancePreparedScalar(prepared, slot);
    }

    // Estimate plus the RaBitQ confidence interval for the squared distance:
    // *lower <= d_true <= *upper with probability >= 1 - delta (eps0).
    float distanceBound(const void *prepared, const void *slot, float *lower, float *upper,
                        float eps0 = kDefaultEps0) const {
        const float d = dist_func_(prepared, slot, this);
        const auto *bytes = static_cast<const uint8_t *>(slot);
        const float xnorm = common::loadUnaligned<float>(bytes + payloadBytes());
        const float df = normalizedDotFactor(bytes);
        const float qnorm = bitwiseQuery()
                                ? common::loadUnaligned<float>(prepared)
                                : common::loadUnaligned<float>(
                                      static_cast<const float *>(prepared) + padded_);
        const float ratio = df > 0.0f ? std::sqrt(std::max(0.0f, 1.0f - df * df)) / df
                                      : std::numeric_limits<float>::infinity();
        const float eps = eps0 * ratio / std::sqrt(static_cast<float>(padded_ - 1));
        const float half = 2.0f * xnorm * qnorm * eps;
        if (lower) *lower = std::max(0.0f, d - half);
        if (upper) *upper = d + half;
        return d;
    }

    // 1-to-N asymmetric search. `codes` is n slots back to back.
    void distanceBatch1ToN(const float *query, const void *codes, size_t n,
                           float *out) const {
        if (query == nullptr || out == nullptr)
            throw std::invalid_argument("RaBitQ distanceBatch1ToN: null pointer");
        if (n > 0 && codes == nullptr)
            throw std::invalid_argument("RaBitQ distanceBatch1ToN: null codes");
        std::vector<uint8_t> prepared(querySizeBytes());
        prepareQuery(query, prepared.data());
        const auto *base = static_cast<const uint8_t *>(codes);
        const size_t stride = codeSizeBytes();
        const DistFunc fn = dist_func_;
        VSQ_OMP_PARALLEL_FOR(num_threads_, n)
        for (long long i = 0; i < static_cast<long long>(n); ++i)
            out[i] = fn(prepared.data(), base + static_cast<size_t>(i) * stride, this);
    }

    // M-to-N asymmetric search. `queries` is row-major [m, dim()], `out` is
    // row-major [m, n]. Queries are prepared in blocks and each code tile is
    // scored against the whole block (codes stay in cache).
    void distanceBatchMToN(const float *queries, size_t m, const void *codes,
                           size_t n, float *out) const {
        if (out == nullptr)
            throw std::invalid_argument("RaBitQ distanceBatchMToN: null out");
        if (m > 0 && queries == nullptr)
            throw std::invalid_argument("RaBitQ distanceBatchMToN: null queries");
        if (n > 0 && codes == nullptr)
            throw std::invalid_argument("RaBitQ distanceBatchMToN: null codes");
        const size_t qsz = querySizeBytes();
        const size_t stride = codeSizeBytes();
        const auto *base = static_cast<const uint8_t *>(codes);
        const DistFunc fn = dist_func_;
        std::vector<uint8_t> block(std::min(m, kQueryBlock) * qsz);
        for (size_t q0 = 0; q0 < m; q0 += kQueryBlock) {
            const size_t qb = std::min(kQueryBlock, m - q0);
            VSQ_OMP_PARALLEL_FOR_DYNAMIC(num_threads_, qb)
            for (long long j = 0; j < static_cast<long long>(qb); ++j)
                prepareQuery(queries + (q0 + static_cast<size_t>(j)) * dim_,
                             block.data() + static_cast<size_t>(j) * qsz);
            const size_t tiles = (n + kCodeTile - 1) / kCodeTile;
            VSQ_OMP_PARALLEL_FOR_DYNAMIC(num_threads_, tiles)
            for (long long t = 0; t < static_cast<long long>(tiles); ++t) {
                const size_t c0 = static_cast<size_t>(t) * kCodeTile;
                const size_t c1 = std::min(n, c0 + kCodeTile);
                for (size_t c = c0; c < c1; ++c)
                    for (size_t j = 0; j < qb; ++j)
                        out[(q0 + j) * n + c] = fn(block.data() + j * qsz, base + c * stride, this);
            }
        }
    }

    // "1-neon-q4", "4-avx2", "8-scalar", ...: bits, ISA, quantized query.
    std::string distanceKernel() const {
        std::string name = std::to_string(bits_) + "-" + common::isaName(isa_);
        if (bitwiseQuery()) name += "-q" + std::to_string(query_bits_);
        return name;
    }

private:
    static constexpr uint64_t kQuerySeed = 0x5241424954515ULL;  // dither seed
    static constexpr size_t kQueryBlock = 64;
    static constexpr size_t kCodeTile = 256;

    static size_t checkedDim(size_t dim) {
        if (dim == 0)
            throw std::invalid_argument("RaBitQ: dim must be positive");
        return dim;
    }

    int resolveQueryBits(int requested) const {
        const bool bitplanes_ok = padded_ % 64 == 0;
        if (requested < 0)
            return (bits_ == 1 && bitplanes_ok) ? 4 : 0;
        if (bits_ != 1 && requested != 0)
            throw std::invalid_argument("RaBitQ: query_bits applies to 1-bit codes only");
        if (requested > bitwise::kMaxQueryBits)
            throw std::invalid_argument("RaBitQ: query_bits must be in [0, 8]");
        if (requested > 0 && !bitplanes_ok)
            throw std::invalid_argument(
                "RaBitQ: a quantized query needs padded_dim % 64 == 0, padded_dim=" +
                std::to_string(padded_));
        return requested;
    }

    bool bitwiseQuery() const { return bits_ == 1 && query_bits_ > 0; }

    // Per-thread D-float scratch; grows once, never per call.
    float *scratch() const {
        static thread_local std::vector<float> buf;
        if (buf.size() < padded_) buf.resize(padded_);
        return buf.data();
    }
    uint8_t *queryScratch() const {
        static thread_local std::vector<uint8_t> buf;
        if (buf.size() < querySizeBytes()) buf.resize(querySizeBytes());
        return buf.data();
    }

    size_t payloadBytes() const {
        if (bits_ == 1)
            return (padded_ + 7) / 8;
        if (bits_ == 4)
            return padded_ / 2;
        return padded_;
    }

    void rotateUnit(float *unit) const { rot_.applyPadded(unit); }

    // x == c: norm 0 and a non-zero dot_factor, so the estimate is exact.
    void writeCenterSlot(uint8_t *slot) const {
        std::memset(slot, 0, payloadBytes());
        const float zero = 0.0f, one = 1.0f;
        std::memcpy(slot + payloadBytes(), &zero, sizeof(float));
        std::memcpy(slot + payloadBytes() + sizeof(float), &one, sizeof(float));
    }

    // <o_bar, o'> with o_bar the unit-norm reconstruction (error bound input).
    float normalizedDotFactor(const uint8_t *slot) const {
        const float df = common::loadUnaligned<float>(slot + payloadBytes() + sizeof(float));
        if (bits_ == 1)
            return df;
        double ysq = 0.0;
        for (size_t i = 0; i < padded_; ++i) {
            const unsigned code = bits_ == 4 ? ((slot[i >> 1] >> ((i & 1u) * 4u)) & 0x0Fu) : slot[i];
            const double y = static_cast<double>(code) - static_cast<double>(center_);
            ysq += y * y;
        }
        return ysq > 0.0 ? static_cast<float>(df / std::sqrt(ysq)) : 0.0f;
    }

    // <ō, o> where o is the rotated unit vector and ō_i = ±1/sqrt(D).
    float dotWithCube(const float *rotated_unit) const {
        float acc = 0.0f;
        for (size_t i = 0; i < padded_; ++i) {
            const float s = (rotated_unit[i] >= 0.0f) ? inv_sqrt_d_
                                                       : -inv_sqrt_d_;
            acc += s * rotated_unit[i];
        }
        return acc;
    }

    // Extended RaBitQ Algorithm 1. `rotated_unit` is o'. Writes one unsigned
    // grid index per coordinate and returns <y, o'> in float32.
    //
    // Grid coordinates are the half-integers
    //   -(2^B-1)/2, -(2^B-1)/2+1, ..., (2^B-1)/2.
    // As t → 0+ the nearest in-orthant point is +1/2 where o' >= 0 and
    // -1/2 where o' < 0. Later critical values move one step further into
    // that orthant. The initial rounding is scored: it is the nearest
    // neighbour on (0, t_first).
    float quantizeExtended(const float *rotated_unit, uint8_t *codes) const {
        const int levels = 1 << bits_;
        const int start_pos = levels >> 1;       // round(center) = 2^{B-1}
        const int start_neg = start_pos - 1;     // floor(center)
        const double center = static_cast<double>(center_);

        std::vector<int> code(padded_);
        double dot = 0.0;
        double normsq = 0.0;
        for (size_t i = 0; i < padded_; ++i) {
            const int c = (rotated_unit[i] >= 0.0f) ? start_pos : start_neg;
            code[i] = c;
            const double v = static_cast<double>(c) - center;
            const double oi = static_cast<double>(rotated_unit[i]);
            dot += v * oi;
            normsq += v * v;
        }
        const std::vector<int> initial = code;

        // One key per critical value: high 32 bits are the sortable
        // threshold, low 32 bits are the coordinate. Radix order matches
        // sorting by (t, index). Below a few thousand keys a comparison
        // sort is faster, so that path stays.
        const size_t steps_per_coord =
            static_cast<size_t>(std::max(start_pos - 1, 0));
        std::vector<uint64_t> keys;
        keys.reserve(padded_ * steps_per_coord);
        for (size_t i = 0; i < padded_; ++i) {
            const float oi = rotated_unit[i];
            if (oi > 0.0f) {
                for (int k = start_pos + 1; k <= levels - 1; ++k) {
                    const float t =
                        (static_cast<float>(k) - 0.5f - center_) / oi;
                    const uint64_t key =
                        (static_cast<uint64_t>(detail::sortableFloat(t))
                         << 32) |
                        static_cast<uint32_t>(i);
                    keys.push_back(key);
                }
            } else if (oi < 0.0f) {
                for (int m = start_neg - 1; m >= 0; --m) {
                    const float t =
                        (static_cast<float>(m) + 0.5f - center_) / oi;
                    const uint64_t key =
                        (static_cast<uint64_t>(detail::sortableFloat(t))
                         << 32) |
                        static_cast<uint32_t>(i);
                    keys.push_back(key);
                }
            }
        }
        if (keys.size() >= 4096)
            detail::radixSortU64(keys);
        else
            std::sort(keys.begin(), keys.end());

        // Squared cosine is monotone with the cosine for a positive dot,
        // and it avoids a sqrt in the comparison.
        auto consider = [](double d, double nsq) {
            return (d * d) / nsq;
        };
        double best_score = -1.0;
        long long best_step = -1;
        bool have = false;
        if (dot > 0.0 && normsq > 0.0) {
            best_score = consider(dot, normsq);
            have = true;
        }
        for (size_t s = 0; s < keys.size(); ++s) {
            const size_t i = static_cast<size_t>(keys[s] & 0xFFFFFFFFu);
            const int delta = (rotated_unit[i] > 0.0f) ? 1 : -1;
            const int neu = code[i] + delta;
            if (neu < 0 || neu >= levels) {
                throw std::logic_error(
                    "RaBitQ quantize: code left the grid, padded_dim=" +
                    std::to_string(padded_));
            }
            const double v = static_cast<double>(code[i]) - center;
            const double dv = static_cast<double>(delta);
            dot += dv * static_cast<double>(rotated_unit[i]);
            normsq += 2.0 * v * dv + dv * dv;
            code[i] = neu;
            if (dot > 0.0 && normsq > 0.0) {
                const double score = consider(dot, normsq);
                if (!have || score > best_score) {
                    best_score = score;
                    best_step = static_cast<long long>(s);
                    have = true;
                }
            }
        }
        if (!have) {
            throw std::invalid_argument(
                "RaBitQ encode: dot_factor is 0, padded_dim=" +
                std::to_string(padded_));
        }

        if (best_step >= 0) {
            code = initial;
            for (long long s = 0; s <= best_step; ++s) {
                const size_t i =
                    static_cast<size_t>(keys[static_cast<size_t>(s)] & 0xFFFFFFFFu);
                const int delta = (rotated_unit[i] > 0.0f) ? 1 : -1;
                code[i] += delta;
            }
        } else {
            code = initial;
        }

        float acc = 0.0f;
        for (size_t i = 0; i < padded_; ++i) {
            codes[i] = static_cast<uint8_t>(code[i]);
            acc += (static_cast<float>(code[i]) - center_) * rotated_unit[i];
        }
        if (!(acc > 0.0f) && !(acc < 0.0f)) {
            throw std::invalid_argument(
                "RaBitQ encode: dot_factor is 0, padded_dim=" +
                std::to_string(padded_));
        }
        return acc;
    }

    // Grid index from a positive scale. o >= 0 maps to [2^{B-1}, 2^B).
    // o < 0 maps to [0, 2^{B-1}). |code - center| = magnitude + 0.5.
    float emitGrid(const float *rotated_unit, uint8_t *codes, double t) const {
        const int levels = 1 << bits_;
        const int max_mag = (levels >> 1) - 1;
        const int start_pos = levels >> 1;
        const int start_neg = start_pos - 1;
        if (!(t > 0.0) || !std::isfinite(t))
            throw std::invalid_argument(
                "RaBitQ encode: scale is not positive, padded_dim=" +
                std::to_string(padded_));
        float acc = 0.0f;
        for (size_t i = 0; i < padded_; ++i) {
            const float oi = rotated_unit[i];
            const int mag = detail::magnitudeAtScale(
                std::fabs(static_cast<double>(oi)), t, max_mag);
            const int c = (oi >= 0.0f) ? start_pos + mag : start_neg - mag;
            if (c < 0 || c >= levels)
                throw std::logic_error(
                    "RaBitQ quantize: code left the grid, padded_dim=" +
                    std::to_string(padded_));
            codes[i] = static_cast<uint8_t>(c);
            acc += (static_cast<float>(c) - center_) * oi;
        }
        if (!(acc > 0.0f) && !(acc < 0.0f))
            throw std::invalid_argument(
                "RaBitQ encode: dot_factor is 0, padded_dim=" +
                std::to_string(padded_));
        return acc;
    }

    float quantizeFixedScale(const float *rotated_unit, uint8_t *codes) const {
        return emitGrid(rotated_unit, codes, t_fixed_);
    }

    // Left edge of the Algorithm 1 plateau that produced `codes`.
    // The last applied threshold is the max t among coordinates that left
    // the initial rounding. If none did, the edge is half the next threshold.
    double scaleOfCodes(const float *rotated_unit,
                        const uint8_t *codes) const {
        const int levels = 1 << bits_;
        const int start_pos = levels >> 1;
        const int start_neg = start_pos - 1;
        float max_t = 0.0f;
        bool any = false;
        float next_t = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < padded_; ++i) {
            const float oi = rotated_unit[i];
            const int c = static_cast<int>(codes[i]);
            if (oi > 0.0f) {
                if (c < start_pos || c >= levels)
                    throw std::logic_error(
                        "RaBitQ fixed scale: positive code left the ray");
                if (c > start_pos) {
                    const float t =
                        (static_cast<float>(c) - 0.5f - center_) / oi;
                    if (!any || t > max_t)
                        max_t = t;
                    any = true;
                }
                if (c + 1 < levels) {
                    const float t =
                        (static_cast<float>(c + 1) - 0.5f - center_) / oi;
                    if (t < next_t)
                        next_t = t;
                }
            } else if (oi < 0.0f) {
                if (c < 0 || c > start_neg)
                    throw std::logic_error(
                        "RaBitQ fixed scale: negative code left the ray");
                if (c < start_neg) {
                    const float t =
                        (static_cast<float>(c) + 0.5f - center_) / oi;
                    if (!any || t > max_t)
                        max_t = t;
                    any = true;
                }
                if (c > 0) {
                    const float t =
                        (static_cast<float>(c - 1) + 0.5f - center_) / oi;
                    if (t < next_t)
                        next_t = t;
                }
            }
        }
        if (any)
            return static_cast<double>(max_t);
        if (std::isfinite(next_t) && next_t > 0.0f)
            return 0.5 * static_cast<double>(next_t);
        throw std::invalid_argument(
            "RaBitQ fixed scale: no threshold, padded_dim=" +
            std::to_string(padded_));
    }

    // Mean Algorithm 1 plateau edge over `count` residuals. fill(n, r) writes
    // residual n into r[0, dim) (r[dim, D) is already zero); each one is
    // normalized and rotated the same way encode is. Zero or degenerate
    // residuals are skipped. Throws if none is usable or t is not positive.
    template <class Fill>
    double meanPlateauScale(size_t count, Fill fill, const char *what) const {
        std::vector<float> rotated(padded_, 0.0f);
        std::vector<uint8_t> codes(padded_);
        double sum = 0.0;
        size_t used = 0;
        for (size_t n = 0; n < count; ++n) {
            std::fill(rotated.begin(), rotated.end(), 0.0f);
            fill(n, rotated.data());
            float norm = 0.0f;
            for (size_t i = 0; i < dim_; ++i)
                norm += rotated[i] * rotated[i];
            norm = std::sqrt(norm);
            if (!(norm > 0.0f) || !std::isfinite(norm))
                continue;
            const float inv = 1.0f / norm;
            for (size_t i = 0; i < dim_; ++i)
                rotated[i] *= inv;
            rotateUnit(rotated.data());
            try {
                quantizeExtended(rotated.data(), codes.data());
            } catch (const std::invalid_argument &) {
                continue;
            }
            sum += scaleOfCodes(rotated.data(), codes.data());
            ++used;
        }
        if (used == 0)
            throw std::invalid_argument(std::string("RaBitQ ") + what +
                                        ": calibration produced no vector, padded_dim=" +
                                        std::to_string(padded_));
        const double t = sum / static_cast<double>(used);
        if (!(t > 0.0) || !std::isfinite(t))
            throw std::invalid_argument(std::string("RaBitQ ") + what +
                                        ": scale is not positive, padded_dim=" +
                                        std::to_string(padded_));
        return t;
    }

    // fixed_scale: mean plateau edge over 100 standard-normal residuals. Seed
    // 42 is the library's calibration seed and does not depend on rot_seed.
    double calibrateFixedScale() const {
        constexpr size_t kCount = 100;
        std::mt19937_64 rng(42);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        return meanPlateauScale(
            kCount,
            [&](size_t, float *r) {
                for (size_t i = 0; i < dim_; ++i) r[i] = normal(rng);
            },
            "fixed scale");
    }

    // trained_scale: mean plateau edge over the residuals x - c of up to
    // kTrainCalibRows rows of X [n, dim], taken at an even stride.
    double calibrateTrainedScale(const float *X, size_t n) const {
        const size_t count = std::min(n, kTrainCalibRows);
        const size_t stride = n / count;  // >= 1 because count <= n
        return meanPlateauScale(
            count,
            [&](size_t k, float *r) {
                const float *x = X + k * stride * dim_;
                for (size_t i = 0; i < dim_; ++i) r[i] = x[i] - centroid_[i];
            },
            "trained scale");
    }

    // windowed_scale: min-heap of the next magnitude step inside [t_start, t_end).
    // Score is N/sqrt(S) with q_i = magnitude_i + 0.5, the same argmax as
    // Algorithm 1's squared cosine. One live event per coordinate.
    double bestWindowedScale(const float *rotated_unit) const {
        const int ex_bits = bits_ - 1;
        const int max_code = (1 << ex_bits) - 1;
        double max_o = 0.0;
        for (size_t i = 0; i < padded_; ++i)
            max_o = std::max(max_o, std::fabs(static_cast<double>(rotated_unit[i])));
        if (!(max_o > 0.0))
            throw std::invalid_argument(
                "RaBitQ encode: rotated residual is 0, padded_dim=" +
                std::to_string(padded_));
        const double t_end =
            static_cast<double>(max_code + 10) / max_o;
        const double t_start =
            t_end * static_cast<double>(detail::tightStart(ex_bits));

        using Event = std::pair<double, size_t>;
        std::vector<Event> next_t;
        next_t.reserve(padded_);
        std::vector<int> cur(padded_);
        double sqr = static_cast<double>(padded_) * 0.25;
        double num = 0.0;
        for (size_t i = 0; i < padded_; ++i) {
            const double magnitude =
                std::fabs(static_cast<double>(rotated_unit[i]));
            const int c = detail::magnitudeAtScale(
                magnitude, t_start, max_code);
            cur[i] = c;
            sqr += static_cast<double>(c) * static_cast<double>(c) + c;
            num += (static_cast<double>(c) + 0.5) * magnitude;
            if (magnitude > 0.0 && c < max_code) {
                const double nxt = static_cast<double>(c + 1) / magnitude;
                if (nxt < t_end)
                    next_t.emplace_back(nxt, i);
            }
        }
        std::make_heap(next_t.begin(), next_t.end(), std::greater<Event>{});
        double max_ip = num / std::sqrt(sqr);
        double best_t = t_start;
        while (!next_t.empty()) {
            const double cur_t = next_t.front().first;
            do {
                const size_t i = next_t.front().second;
                ++cur[i];
                sqr += 2.0 * static_cast<double>(cur[i]);
                const double magnitude =
                    std::fabs(static_cast<double>(rotated_unit[i]));
                num += magnitude;
                Event next{t_end, i};
                if (cur[i] < max_code)
                    next.first = static_cast<double>(cur[i] + 1) / magnitude;
                if (next.first >= t_end) {
                    next = next_t.back();
                    next_t.pop_back();
                }
                if (!next_t.empty()) {
                    size_t parent = 0;
                    size_t child = 1;
                    while (child < next_t.size()) {
                        if (child + 1 < next_t.size() &&
                            next_t[child + 1] < next_t[child])
                            ++child;
                        if (!(next_t[child] < next))
                            break;
                        next_t[parent] = next_t[child];
                        parent = child;
                        child = 2 * parent + 1;
                    }
                    next_t[parent] = next;
                }
            } while (!next_t.empty() && next_t.front().first == cur_t);
            const double cur_ip = num / std::sqrt(sqr);
            if (cur_ip > max_ip) {
                max_ip = cur_ip;
                best_t = cur_t;
            }
        }
        return best_t;
    }

    float quantizeWindowedScale(const float *rotated_unit, uint8_t *codes) const {
        return emitGrid(rotated_unit, codes, bestWindowedScale(rotated_unit));
    }

    void packCodes(const uint8_t *codes, float norm, float dot,
                   uint8_t *slot) const {
        const size_t n = payloadBytes();
        std::memset(slot, 0, n);
        if (bits_ == 8) {
            std::memcpy(slot, codes, padded_);
        } else {
            for (size_t i = 0; i < padded_; ++i) {
                const unsigned shift = (i & 1u) ? 4u : 0u;
                slot[i >> 1] = static_cast<uint8_t>(slot[i >> 1] |
                                                     (codes[i] << shift));
            }
        }
        std::memcpy(slot + n, &norm, sizeof(float));
        std::memcpy(slot + n + sizeof(float), &dot, sizeof(float));
    }

    // Float-query kernel. No validation: the slot and query are trusted
    // (hnswlib calls this millions of times). Clamped to >= 0.
    template <int Bits, detail::DotIsa Isa>
    static float distKernel(const void *prepared, const void *slot,
                            const void *param) {
        const auto *self = static_cast<const RaBitQSpace *>(param);
        const auto *qrot = static_cast<const float *>(prepared);
        const float qnorm = common::loadUnaligned<float>(qrot + self->padded_);
        const auto *bytes = static_cast<const uint8_t *>(slot);
        const size_t payload = self->payloadBytes();
        const float xnorm = common::loadUnaligned<float>(bytes + payload);
        const float dot_factor = common::loadUnaligned<float>(bytes + payload + sizeof(float));
        float ip = 0.0f;
        if (qnorm > 0.0f && xnorm > 0.0f) {
            if constexpr (Bits == 1) {
                ip = detail::dot1<Isa>(qrot, bytes, self->padded_, self->inv_sqrt_d_) / dot_factor;
            } else {
                const float sum_q = common::loadUnaligned<float>(qrot + self->padded_ + 1);
                const float acc = (Bits == 4) ? detail::dot4<Isa>(qrot, bytes, self->padded_)
                                              : detail::dot8<Isa>(qrot, bytes, self->padded_);
                ip = (acc - self->center_ * sum_q) / dot_factor;
            }
        }
        return std::max(0.0f, xnorm * xnorm + qnorm * qnorm - 2.0f * xnorm * qnorm * ip);
    }

    // Quantized-query 1-bit kernel (popcount).
    float distBitwiseWith(const void *prepared, const void *slot,
                          bitwise::BitwiseFn fn) const {
        using bitwise::QueryHeader;
        QueryHeader h;
        std::memcpy(&h, prepared, sizeof(h));
        const auto *bytes = static_cast<const uint8_t *>(slot);
        const size_t payload = payloadBytes();
        const float xnorm = common::loadUnaligned<float>(bytes + payload);
        const float dot_factor = common::loadUnaligned<float>(bytes + payload + sizeof(float));
        float ip = 0.0f;
        if (h.qnorm > 0.0f && xnorm > 0.0f) {
            uint64_t acc = 0, pop = 0;
            fn(bytes, static_cast<const uint8_t *>(prepared) + sizeof(QueryHeader),
               bitwise::planeWords(padded_), query_bits_, &acc, &pop);
            ip = bitwise::estimateDot(h, acc, pop, inv_sqrt_d_) / dot_factor;
        }
        return std::max(0.0f, xnorm * xnorm + h.qnorm * h.qnorm - 2.0f * xnorm * h.qnorm * ip);
    }

    static float distBitwise(const void *prepared, const void *slot, const void *param) {
        const auto *self = static_cast<const RaBitQSpace *>(param);
        return self->distBitwiseWith(prepared, slot, self->bitwise_);
    }

    template <detail::DotIsa Isa>
    DistFunc selectFloatDist() const {
        if (bits_ == 1)
            return &RaBitQSpace::distKernel<1, Isa>;
        if (bits_ == 4)
            return &RaBitQSpace::distKernel<4, Isa>;
        return &RaBitQSpace::distKernel<8, Isa>;
    }

    DistFunc selectDist() const {
        if (bitwiseQuery())
            return &RaBitQSpace::distBitwise;
#if defined(VSQ_HAVE_AVX2_KERNELS)
        if (isa_ == common::Isa::Avx2)
            return selectFloatDist<detail::DotIsa::Avx2>();
#endif
#if defined(VSQ_NEON)
        if (isa_ == common::Isa::Neon)
            return selectFloatDist<detail::DotIsa::Neon>();
#endif
        return selectFloatDist<detail::DotIsa::Scalar>();
    }

    size_t dim_;
    int bits_;
    uint64_t rot_seed_;
    common::Rotation rot_;
    size_t padded_;
    std::vector<float> centroid_;
    float inv_sqrt_d_;
    float center_;
    int encode_mode_;
    double t_fixed_;  // frozen scale of fixed_scale / trained_scale
    bool trained_ = false;
    common::Isa isa_;
    int num_threads_;
    int query_bits_ = 0;
    DistFunc dist_func_ = nullptr;
    bitwise::BitwiseFn bitwise_ = nullptr;
};

}  // namespace vsq::rabitq
