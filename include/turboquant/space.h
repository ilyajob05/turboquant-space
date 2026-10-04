#pragma once
// TurboQuantSpace — TurboQuant code format v2.
//
// Pipeline for a vector x (dim floats):
//   c   = centroid of x (centering.h: none | mean | ivf)
//   n   = ||x - c||,  u = (x - c) / n
//   ru  = R u                                   (rotation.h, D floats, unit norm)
//   bits 4/8: idx_i = LloydMax_b(ru_i * sqrt(D)),  rec_i = cent[idx_i] / sqrt(D)
//             (b = bits, or bits - 1 when the QJL sign takes one bit)
//   bits 16:  rec_i = fp16(ru_i)
//   s   = 1 / <ru, rec>   (Corrected estimator, RaBitQ-style unbiasing) or 1
//
// Code layout (little-endian, unaligned; total = codeSizeBytes()):
//   payload  bits 4: D/2 bytes, nibble = idx (QJL: idx3 << 1 | sign), low nibble = even coord
//            bits 8: D bytes,   byte   = idx (QJL: idx7 << 1 | sign)
//            bits 16: D x fp16
//   f_sq   float32  ||x - c||^2
//   f_mul  float32  n * s / sqrt(D)   (bits 16: n * s)
//   f_ct   float32  estimate of <c, x - c>             [ivf only]
//   f_qjl  float32  n * sqrt(pi/2) / sqrt(D) * gamma   [qjl only; gamma = ||ru - rec||]
//   cid    uint16   cluster id                         [ivf only]
//
// Asymmetric squared L2 for a prepared query q:
//   d = T[cid] + f_sq - 2 * (f_mul * <Rq', dec> [+ f_qjl * <S Rq', sign>] - f_ct)
//   T[j] = ||q - c_j||^2; q' = q - mu for mean centering, q' = q otherwise.
// The query costs one rotation and a k-float table, so IVF adds no per-code work.
//
// Symmetric (code x code, the HNSW build metric): same-cluster pairs use
//   d = f_sq_a + f_sq_b - 2 f_mul_a f_mul_b <dec_a, dec_b>
// and IVF cross-cluster pairs add ||c_a - c_b||^2 and the <c, r> cross terms
// through the rotated centroids (one fused pass, kernel symCross).
//
// Distance functions never throw and never allocate; results are clamped >= 0.
// Encoding, query preparation and training validate and may throw.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "centering.h"
#include "../common/config.h"
#include "../common/fp16.h"
#include "../common/vecops.h"
#include "lloyd_max.h"
#include "../common/rotation.h"
#include "kernels.h"

namespace vsq::turboquant {

enum class Estimator : uint8_t { Plain = 0, Corrected = 1 };
enum class QjlCrossTerm : uint8_t { Linear = 0, Arcsine = 1 };

constexpr uint32_t kTurboQuantFormatVersion = 2;

struct TurboQuantConfig {
    size_t dim = 0;                                 // input dimension, >= 1
    int bits = 4;                                   // stored bits per coordinate: 4 | 8 | 16
    bool qjl = false;                               // 4/8 only: 1 of `bits` is a QJL sign
    Estimator estimator = Estimator::Corrected;     // forced to Plain when qjl
    QjlCrossTerm qjl_cross = QjlCrossTerm::Linear;  // full-symmetric <e_a, e_b> term
    Centering centering = Centering::Ivf;
    size_t n_clusters = 256;                        // ivf only
    int rotation_rounds = 3;
    uint64_t rot_seed = 42;
    uint64_t qjl_seed = 137;
    int num_threads = 0;                            // batch helpers; 0 = OpenMP default
    common::Isa isa = common::detectIsa();                          // requested kernels (clamped to CPU)
};

// Prepared asymmetric query. Reusable across calls (buffers keep capacity).
struct TurboQuantQuery {
    std::vector<float> qrot;   // [D]  R q'   (q' = q - mu for mean centering)
    std::vector<float> sq;     // [D]  S R q' (qjl only)
    std::vector<float> table;  // [max(k,1)] ||q - c_j||^2
};

// Prepared code for the full-QJL symmetric estimate (qjl spaces only).
struct TurboQuantSymQuery {
    std::vector<float> w1;    // [D] weights on dec_b      (same cluster)
    std::vector<float> w1x;   // [D] w1 + R c_q            (ivf, other cluster)
    std::vector<float> w2;    // [D] weights on sign_b
    std::vector<float> sign;  // [D] +-1 signs of the query code (arcsine term)
    std::vector<float> xq;    // [k] <R c_j, dec_q>        (ivf)
    float f_sq = 0, f_mul = 0, f_ct = 0, f_qjl = 0;
    uint32_t cid = 0;
};

class TurboQuantSpace {
public:
    explicit TurboQuantSpace(const TurboQuantConfig &cfg)
        : cfg_(validated(cfg)),
          rot_(cfg_.dim, cfg_.rot_seed, cfg_.rotation_rounds, common::RotationKind::BlockKac, cfg_.isa),
          centerer_(cfg_.centering, cfg_.dim, cfg_.n_clusters, common::resolveIsa(cfg_.isa)) {
        D_ = rot_.paddedDim();
        sqrt_d_ = std::sqrt(static_cast<float>(D_));
        isa_ = common::resolveIsa(cfg_.isa);
        num_threads_ = common::resolveNumThreads(cfg_.num_threads);
        if (cfg_.qjl) {
            qjl_rot_.emplace(D_, cfg_.qjl_seed, cfg_.rotation_rounds, common::RotationKind::BlockKac, isa_);
            cfg_.estimator = Estimator::Plain;  // QJL is the bias correction
        }
        buildTablesAndLayout();
    }

    // ---- configuration / layout -----------------------------------------------------

    const TurboQuantConfig &config() const { return cfg_; }
    size_t dim() const { return cfg_.dim; }
    size_t paddedDim() const { return D_; }
    int bits() const { return cfg_.bits; }
    bool qjl() const { return cfg_.qjl; }
    Centering centering() const { return cfg_.centering; }
    size_t numClusters() const { return centerer_.numCentroids(); }
    bool trained() const { return centerer_.trained(); }
    size_t codeSizeBytes() const { return code_size_; }
    size_t payloadBytes() const { return payload_bytes_; }
    common::Isa kernelIsa() const { return kernels_.isa; }
    int numThreads() const { return num_threads_; }
    const std::vector<float> &centroids() const { return centerer_.centroids(); }
    const common::Rotation &rotation() const { return rot_; }
    // Value of each payload unit (16 or 256 entries; empty for bits 16).
    const std::vector<float> &decodeTable() const { return table_a_; }
    // Byte offsets of the metadata fields inside a code (see the layout above).
    size_t offsetSq() const { return off_sq_; }
    size_t offsetMul() const { return off_mul_; }
    size_t offsetCt() const { return off_ct_; }
    size_t offsetCid() const { return off_cid_; }

    // ---- training -------------------------------------------------------------------

    // X: row-major float32 [n, dim]. No-op for centering none.
    void train(const float *X, size_t n, uint64_t seed = 1234, int iters = 10) {
        centerer_.train(X, n, seed, num_threads_, iters);
        rebuildCentroidCaches();
    }

    // C: row-major float32 [k, dim] (k == 1 for mean).
    void setCentroids(const float *C, size_t k) {
        centerer_.setCentroids(C, k);
        rebuildCentroidCaches();
    }

    // ---- encoding -------------------------------------------------------------------

    // x: dim floats. out: codeSizeBytes() bytes.
    void encode(const float *x, void *out) const {
        requireTrained("encode");
        encodeWith(x, static_cast<uint8_t *>(out), scratch(0), scratch(1));
    }

    // X: row-major [n, dim]; out: n * codeSizeBytes() bytes.
    void encodeBatch(const float *X, size_t n, void *out) const {
        requireTrained("encodeBatch");
        auto *dst = static_cast<uint8_t *>(out);
        VSQ_OMP_PARALLEL_FOR(num_threads_, n)
        for (long long ii = 0; ii < static_cast<long long>(n); ++ii) {
            const size_t i = static_cast<size_t>(ii);
            encodeWith(X + i * cfg_.dim, dst + i * code_size_, scratch(0), scratch(1));
        }
    }

    // Approximate reconstruction x_hat = c + R^T(f_mul * dec) (dim floats).
    void decode(const void *code, float *out) const {
        const auto *p = static_cast<const uint8_t *>(code);
        float *buf = scratch(0);
        const float f_mul = common::loadUnaligned<float>(p + off_mul_);
        decodePayload(p, buf);
        for (size_t i = 0; i < D_; ++i) buf[i] *= f_mul;
        rot_.applyTransposePadded(buf);
        const float *c = centroidOf(p);
        for (size_t i = 0; i < cfg_.dim; ++i) out[i] = buf[i] + (c ? c[i] : 0.0f);
    }

    // ---- asymmetric (raw query x code) ----------------------------------------------

    void prepareQuery(const float *q, TurboQuantQuery &pq) const {
        requireTrained("prepareQuery");
        pq.table.resize(std::max<size_t>(centerer_.numCentroids(), 1));
        centerer_.queryTable(q, pq.table.data());
        pq.qrot.resize(D_);
        if (cfg_.centering == Centering::Mean) {
            const float *mu = centerer_.centroid(0);
            float *buf = scratch(0);
            for (size_t i = 0; i < cfg_.dim; ++i) buf[i] = q[i] - mu[i];
            rot_.apply(buf, pq.qrot.data());
        } else {
            rot_.apply(q, pq.qrot.data());
        }
        if (cfg_.qjl) {
            pq.sq = pq.qrot;
            qjl_rot_->applyPadded(pq.sq.data());
        }
    }

    float distance(const TurboQuantQuery &pq, const void *code) const {
        const auto *p = static_cast<const uint8_t *>(code);
        const float f_sq = common::loadUnaligned<float>(p + off_sq_);
        const float f_mul = common::loadUnaligned<float>(p + off_mul_);
        float ip;
        if (cfg_.qjl) {
            float o[2];
            kernels_.dot2(pq.qrot.data(), pq.sq.data(), p, table_a_.data(), table_b_.data(), D_, o);
            ip = f_mul * o[0] + common::loadUnaligned<float>(p + off_qjl_) * o[1];
        } else {
            ip = f_mul * kernels_.dot1(pq.qrot.data(), p, table_a_.data(), D_);
        }
        float base = pq.table[0];
        if (cfg_.centering == Centering::Ivf) {
            base = pq.table[common::loadUnaligned<uint16_t>(p + off_cid_)];
            ip -= common::loadUnaligned<float>(p + off_ct_);
        }
        return std::max(0.0f, base + f_sq - 2.0f * ip);
    }

    float distance(const float *q, const void *code) const {
        TurboQuantQuery pq;
        prepareQuery(q, pq);
        return distance(pq, code);
    }

    void distanceBatch1ToN(const float *q, const void *codes, size_t n, float *out) const {
        TurboQuantQuery pq;
        prepareQuery(q, pq);
        const auto *base = static_cast<const uint8_t *>(codes);
        VSQ_OMP_PARALLEL_FOR(num_threads_, n)
        for (long long i = 0; i < static_cast<long long>(n); ++i)
            out[i] = distance(pq, base + static_cast<size_t>(i) * code_size_);
    }

    // Q: row-major [m, dim]; out: row-major [m, n]. Queries are prepared in
    // blocks; each code tile is then scored against the whole query block so
    // the codes stay in cache (the plain loop is memory-bound).
    void distanceBatchMToN(const float *Q, size_t m, const void *codes, size_t n,
                           float *out) const {
        const auto *base = static_cast<const uint8_t *>(codes);
        std::vector<TurboQuantQuery> block(std::min(m, kQueryBlock));
        for (size_t q0 = 0; q0 < m; q0 += kQueryBlock) {
            const size_t qb = std::min(kQueryBlock, m - q0);
            VSQ_OMP_PARALLEL_FOR_DYNAMIC(num_threads_, qb)
            for (long long j = 0; j < static_cast<long long>(qb); ++j)
                prepareQuery(Q + (q0 + static_cast<size_t>(j)) * cfg_.dim,
                             block[static_cast<size_t>(j)]);
            const size_t tiles = (n + kCodeTile - 1) / kCodeTile;
            VSQ_OMP_PARALLEL_FOR_DYNAMIC(num_threads_, tiles)
            for (long long t = 0; t < static_cast<long long>(tiles); ++t) {
                const size_t c0 = static_cast<size_t>(t) * kCodeTile;
                const size_t c1 = std::min(n, c0 + kCodeTile);
                for (size_t c = c0; c < c1; ++c) {
                    const uint8_t *code = base + c * code_size_;
                    for (size_t j = 0; j < qb; ++j) out[(q0 + j) * n + c] = distance(block[j], code);
                }
            }
        }
    }

    // ---- symmetric (code x code) ----------------------------------------------------

    float distanceSymmetric(const void *code_a, const void *code_b) const {
        const auto *a = static_cast<const uint8_t *>(code_a);
        const auto *b = static_cast<const uint8_t *>(code_b);
        const float sq_a = common::loadUnaligned<float>(a + off_sq_);
        const float sq_b = common::loadUnaligned<float>(b + off_sq_);
        const float mul_a = common::loadUnaligned<float>(a + off_mul_);
        const float mul_b = common::loadUnaligned<float>(b + off_mul_);
        if (cfg_.centering == Centering::Ivf) {
            const uint16_t ca = common::loadUnaligned<uint16_t>(a + off_cid_);
            const uint16_t cb = common::loadUnaligned<uint16_t>(b + off_cid_);
            if (ca != cb) {
                float o[3];
                kernels_.symCross(a, b, rc_.data() + ca * D_, rc_.data() + cb * D_,
                                  table_a_.data(), D_, o);
                const float ct_a = common::loadUnaligned<float>(a + off_ct_);
                const float ct_b = common::loadUnaligned<float>(b + off_ct_);
                const float d = cc_[static_cast<size_t>(ca) * k_ + cb] + sq_a + sq_b -
                                2.0f * mul_a * mul_b * o[0] +
                                2.0f * ((ct_a - mul_a * o[1]) - (mul_b * o[2] - ct_b));
                return std::max(0.0f, d);
            }
        }
        const float s = kernels_.sym(a, b, table_a_.data(), D_);
        return std::max(0.0f, sq_a + sq_b - 2.0f * mul_a * mul_b * s);
    }

    // A: m codes, B: n codes; out row-major [m, n]. Tiled over B.
    void distanceBatchMToNSymmetric(const void *codes_a, size_t m, const void *codes_b,
                                    size_t n, float *out) const {
        const auto *A = static_cast<const uint8_t *>(codes_a);
        const auto *B = static_cast<const uint8_t *>(codes_b);
        const size_t tiles = (n + kCodeTile - 1) / kCodeTile;
        VSQ_OMP_PARALLEL_FOR_DYNAMIC(num_threads_, tiles)
        for (long long t = 0; t < static_cast<long long>(tiles); ++t) {
            const size_t c0 = static_cast<size_t>(t) * kCodeTile;
            const size_t c1 = std::min(n, c0 + kCodeTile);
            for (size_t i = 0; i < m; ++i)
                for (size_t c = c0; c < c1; ++c)
                    out[i * n + c] = distanceSymmetric(A + i * code_size_, B + c * code_size_);
        }
    }

    // ---- full-QJL symmetric (qjl spaces) --------------------------------------------
    //
    // <u_a, u_b> = <rec_a, rec_b> + <rec_a, e_b> + <e_a, rec_b> + <e_a, e_b>, with
    // the residual terms estimated from QJL signs. S rec_q and S^T sign_q are
    // prepared once, so a pair costs one dot2 pass over code_b: no Hadamard and
    // no allocation per pair. <e_a, e_b> is linear ((pi/2) rho) or the exact
    // arcsine inversion sin((pi/2) rho) (one extra pass).

    void prepareSymmetric(const void *code, TurboQuantSymQuery &sq) const {
        if (!cfg_.qjl)
            throw std::invalid_argument("prepareSymmetric: space was built without qjl");
        const auto *p = static_cast<const uint8_t *>(code);
        sq.f_sq = common::loadUnaligned<float>(p + off_sq_);
        sq.f_mul = common::loadUnaligned<float>(p + off_mul_);
        sq.f_qjl = common::loadUnaligned<float>(p + off_qjl_);
        const bool ivf = cfg_.centering == Centering::Ivf;
        sq.f_ct = ivf ? common::loadUnaligned<float>(p + off_ct_) : 0.0f;
        sq.cid = ivf ? common::loadUnaligned<uint16_t>(p + off_cid_) : 0;
        const float n = std::sqrt(std::max(sq.f_sq, 0.0f));

        sq.w1.resize(D_);
        sq.w2.resize(D_);
        sq.sign.resize(D_);
        std::vector<float> dec(D_), st(D_), srec(D_);
        for (size_t i = 0; i < D_; ++i) {
            const uint8_t unit = unitAt(p, i);
            dec[i] = table_a_[unit];
            sq.sign[i] = table_b_[unit];
            srec[i] = dec[i] / sqrt_d_;
            st[i] = sq.sign[i];
        }
        qjl_rot_->applyPadded(srec.data());         // S rec_q
        qjl_rot_->applyTransposePadded(st.data());  // S^T sign_q
        const bool linear = cfg_.qjl_cross == QjlCrossTerm::Linear;
        for (size_t i = 0; i < D_; ++i) {
            sq.w1[i] = sq.f_mul * dec[i] + sq.f_qjl * st[i];
            sq.w2[i] = n * srec[i] + (linear ? sq.f_qjl * sq.sign[i] : 0.0f);
        }
        if (ivf) {
            const float *rcq = rc_.data() + static_cast<size_t>(sq.cid) * D_;
            sq.w1x.resize(D_);
            for (size_t i = 0; i < D_; ++i) sq.w1x[i] = sq.w1[i] + rcq[i];
            sq.xq.resize(k_);
            for (size_t j = 0; j < k_; ++j) sq.xq[j] = common::dotF32(rc_.data() + j * D_, dec.data(), D_, isa_);
        }
    }

    float distanceSymmetricFull(const TurboQuantSymQuery &sq, const void *code_b) const {
        const auto *b = static_cast<const uint8_t *>(code_b);
        const float sq_b = common::loadUnaligned<float>(b + off_sq_);
        const float mul_b = common::loadUnaligned<float>(b + off_mul_);
        const float qjl_b = common::loadUnaligned<float>(b + off_qjl_);
        uint16_t cb = 0;
        bool cross = false;
        if (cfg_.centering == Centering::Ivf) {
            cb = common::loadUnaligned<uint16_t>(b + off_cid_);
            cross = cb != sq.cid;
        }
        float o[2];
        kernels_.dot2(cross ? sq.w1x.data() : sq.w1.data(), sq.w2.data(), b, table_a_.data(),
                      table_b_.data(), D_, o);
        float ip = mul_b * o[0] + qjl_b * o[1];
        if (cfg_.qjl_cross == QjlCrossTerm::Arcsine) {
            const float rho = kernels_.dot1(sq.sign.data(), b, table_b_.data(), D_) /
                              static_cast<float>(D_);
            ip += sq.f_qjl * qjl_b * (2.0f * static_cast<float>(D_) / static_cast<float>(common::kPi)) *
                  std::sin(static_cast<float>(common::kPi) * 0.5f * rho);
        }
        float d = sq.f_sq + sq_b - 2.0f * ip;
        if (cross) {
            const float ct_b = common::loadUnaligned<float>(b + off_ct_);
            d += cc_[static_cast<size_t>(sq.cid) * k_ + cb] +
                 2.0f * (sq.f_ct - sq.f_mul * sq.xq[cb]) + 2.0f * ct_b;
        }
        return std::max(0.0f, d);
    }

    float distanceSymmetricFull(const void *code_a, const void *code_b) const {
        TurboQuantSymQuery sq;
        prepareSymmetric(code_a, sq);
        return distanceSymmetricFull(sq, code_b);
    }

    void distanceBatchMToNSymmetricFull(const void *codes_a, size_t m, const void *codes_b,
                                        size_t n, float *out) const {
        const auto *A = static_cast<const uint8_t *>(codes_a);
        const auto *B = static_cast<const uint8_t *>(codes_b);
        if (!cfg_.qjl)
            throw std::invalid_argument("distanceBatchMToNSymmetricFull: space was built without qjl");
        VSQ_OMP_PARALLEL_FOR_DYNAMIC(num_threads_, m)
        for (long long ii = 0; ii < static_cast<long long>(m); ++ii) {
            const size_t i = static_cast<size_t>(ii);
            TurboQuantSymQuery sq;
            prepareSymmetric(A + i * code_size_, sq);
            for (size_t c = 0; c < n; ++c) out[i * n + c] = distanceSymmetricFull(sq, B + c * code_size_);
        }
    }

    // ---- hnswlib adapters (DISTFUNC<float> signature) -------------------------------
    // search: (const TurboQuantQuery*, code, space*)   build: (code, code, space*)

    static float searchDistFunc(const void *query, const void *code, const void *space) {
        return static_cast<const TurboQuantSpace *>(space)->distance(
            *static_cast<const TurboQuantQuery *>(query), code);
    }
    static float buildDistFunc(const void *a, const void *b, const void *space) {
        return static_cast<const TurboQuantSpace *>(space)->distanceSymmetric(a, b);
    }

    // ---- serialization (magic, format version, config, centroids) -------------------

    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> s;
        auto put = [&s](const void *p, size_t n) {
            const auto *b = static_cast<const uint8_t *>(p);
            s.insert(s.end(), b, b + n);
        };
        const uint32_t magic = kMagic, version = kTurboQuantFormatVersion;
        const uint64_t dim = cfg_.dim, ncl = cfg_.n_clusters, k = centerer_.numCentroids();
        const int32_t bits = cfg_.bits, rounds = cfg_.rotation_rounds;
        const uint8_t flags[4] = {static_cast<uint8_t>(cfg_.qjl), static_cast<uint8_t>(cfg_.estimator),
                                  static_cast<uint8_t>(cfg_.centering),
                                  static_cast<uint8_t>(cfg_.qjl_cross)};
        put(&magic, 4);
        put(&version, 4);
        put(&dim, 8);
        put(&bits, 4);
        put(flags, 4);
        put(&rounds, 4);
        put(&ncl, 8);
        put(&cfg_.rot_seed, 8);
        put(&cfg_.qjl_seed, 8);
        put(&k, 8);
        if (k) put(centerer_.centroids().data(), k * cfg_.dim * sizeof(float));
        return s;
    }

    static TurboQuantSpace deserialize(const uint8_t *data, size_t size, int num_threads = 0,
                                       common::Isa isa = common::detectIsa()) {
        size_t off = 0;
        auto get = [&](void *p, size_t n) {
            if (off + n > size) throw std::invalid_argument("TurboQuantSpace: truncated state");
            std::memcpy(p, data + off, n);
            off += n;
        };
        uint32_t magic = 0, version = 0;
        get(&magic, 4);
        get(&version, 4);
        if (magic != kMagic) throw std::invalid_argument("TurboQuantSpace: bad state magic");
        if (version != kTurboQuantFormatVersion)
            throw std::invalid_argument("TurboQuantSpace: unsupported format version " +
                                        std::to_string(version));
        uint64_t dim = 0, ncl = 0, k = 0;
        int32_t bits = 0, rounds = 0;
        uint8_t flags[4];
        TurboQuantConfig c;
        get(&dim, 8);
        get(&bits, 4);
        get(flags, 4);
        get(&rounds, 4);
        get(&ncl, 8);
        get(&c.rot_seed, 8);
        get(&c.qjl_seed, 8);
        get(&k, 8);
        c.dim = dim;
        c.bits = bits;
        c.qjl = flags[0] != 0;
        c.estimator = static_cast<Estimator>(flags[1]);
        c.centering = static_cast<Centering>(flags[2]);
        c.qjl_cross = static_cast<QjlCrossTerm>(flags[3]);
        c.rotation_rounds = rounds;
        c.n_clusters = ncl;
        c.num_threads = num_threads;
        c.isa = isa;
        TurboQuantSpace s(c);
        if (k) {
            std::vector<float> C(k * dim);
            get(C.data(), C.size() * sizeof(float));
            s.setCentroids(C.data(), k);
        }
        return s;
    }

private:
    static constexpr uint32_t kMagic = 0x32535154u;  // "TQS2"
    static constexpr size_t kQueryBlock = 64;
    static constexpr size_t kCodeTile = 256;

    static TurboQuantConfig validated(TurboQuantConfig c) {
        if (c.dim == 0) throw std::invalid_argument("TurboQuantSpace: dim must be >= 1");
        if (c.bits != 4 && c.bits != 8 && c.bits != 16)
            throw std::invalid_argument("TurboQuantSpace: bits must be 4, 8 or 16, got " +
                                        std::to_string(c.bits));
        if (c.qjl && c.bits == 16)
            throw std::invalid_argument("TurboQuantSpace: qjl requires bits 4 or 8");
        if (static_cast<uint8_t>(c.estimator) > 1 || static_cast<uint8_t>(c.qjl_cross) > 1 ||
            static_cast<uint8_t>(c.centering) > 2)
            throw std::invalid_argument("TurboQuantSpace: invalid enum value in config");
        return c;
    }

    void buildTablesAndLayout() {
        if (cfg_.bits == 16) {
            payload_ = kernels::Payload::Half;
            payload_bytes_ = 2 * D_;
        } else {
            payload_ = cfg_.bits == 4 ? kernels::Payload::Nibble : kernels::Payload::Byte;
            payload_bytes_ = cfg_.bits == 4 ? D_ / 2 : D_;
            mse_bits_ = cfg_.qjl ? cfg_.bits - 1 : cfg_.bits;
            const LloydMaxQuantizer &lm = lloydMax(mse_bits_);
            const size_t units = size_t{1} << cfg_.bits;
            table_a_.resize(units);
            table_b_.resize(units);
            for (size_t u = 0; u < units; ++u) {
                table_a_[u] = cfg_.qjl ? lm.centroids[u >> 1] : lm.centroids[u];
                table_b_[u] = (u & 1) ? 1.0f : -1.0f;
            }
        }
        kernels_ = kernels::selectKernels(payload_, isa_);
        size_t off = payload_bytes_;
        off_sq_ = off;
        off += 4;
        off_mul_ = off;
        off += 4;
        if (cfg_.centering == Centering::Ivf) {
            off_ct_ = off;
            off += 4;
        }
        if (cfg_.qjl) {
            off_qjl_ = off;
            off += 4;
        }
        if (cfg_.centering == Centering::Ivf) {
            off_cid_ = off;
            off += 2;
        }
        code_size_ = off;
    }

    // R c_j (and S R c_j for qjl) and ||c_a - c_b||^2, for ivf.
    void rebuildCentroidCaches() {
        k_ = centerer_.numCentroids();
        if (cfg_.centering != Centering::Ivf || k_ == 0) return;
        rc_.assign(k_ * D_, 0.0f);
        for (size_t j = 0; j < k_; ++j)
            rot_.apply(centerer_.centroid(static_cast<uint32_t>(j)), rc_.data() + j * D_);
        if (cfg_.qjl) {
            src_ = rc_;
            for (size_t j = 0; j < k_; ++j) qjl_rot_->applyPadded(src_.data() + j * D_);
        }
        cc_.assign(k_ * k_, 0.0f);
        for (size_t a = 0; a < k_; ++a)
            for (size_t b = a + 1; b < k_; ++b) {
                const float d = common::l2sqF32(centerer_.centroid(static_cast<uint32_t>(a)),
                                        centerer_.centroid(static_cast<uint32_t>(b)), cfg_.dim, isa_);
                cc_[a * k_ + b] = cc_[b * k_ + a] = d;
            }
    }

    void requireTrained(const char *what) const {
        if (!centerer_.trained())
            throw std::runtime_error(std::string("TurboQuantSpace::") + what + ": centering '" +
                                     centeringName(cfg_.centering) +
                                     "' needs train(X) or setCentroids() first");
    }

    // Per-thread scratch of D floats (slots 0/1); grows once, never per call.
    float *scratch(int slot) const {
        static thread_local std::vector<float> bufs[2];
        std::vector<float> &b = bufs[slot];
        if (b.size() < D_) b.resize(D_);
        return b.data();
    }

    uint8_t unitAt(const uint8_t *p, size_t i) const {
        return cfg_.bits == 4 ? static_cast<uint8_t>((p[i >> 1] >> ((i & 1) * 4)) & 0x0F) : p[i];
    }

    const float *centroidOf(const uint8_t *code) const {
        if (cfg_.centering == Centering::None) return nullptr;
        const uint32_t cid =
            cfg_.centering == Centering::Ivf ? common::loadUnaligned<uint16_t>(code + off_cid_) : 0;
        return centerer_.centroid(cid);
    }

    // buf[0, D) <- decoded payload (raw table values, or fp16 values).
    void decodePayload(const uint8_t *p, float *buf) const {
        for (size_t i = 0; i < D_; ++i)
            buf[i] = cfg_.bits == 16 ? common::halfToFloat(common::loadUnaligned<uint16_t>(p + 2 * i))
                                     : table_a_[unitAt(p, i)];
    }

    void encodeWith(const float *x, uint8_t *out, float *buf, float *aux) const {
        const uint32_t cid = centerer_.assign(x);
        const float *c = centerer_.centroid(cid);
        float nsq = 0.0f;
        for (size_t i = 0; i < cfg_.dim; ++i) {
            buf[i] = c ? x[i] - c[i] : x[i];
            nsq += buf[i] * buf[i];
        }
        std::memset(buf + cfg_.dim, 0, (D_ - cfg_.dim) * sizeof(float));
        std::memset(out, 0, code_size_);
        float f_mul = 0.0f, f_ct = 0.0f, f_qjl = 0.0f;
        if (nsq > 0.0f && std::isfinite(nsq)) {
            const float inv = 1.0f / std::sqrt(nsq);
            for (size_t i = 0; i < cfg_.dim; ++i) buf[i] *= inv;
            rot_.applyPadded(buf);  // buf = ru (unit norm)
            const float n = std::sqrt(nsq);
            if (cfg_.bits == 16) encodeHalf(buf, out, n, cid, f_mul, f_ct);
            else encodeIndex(buf, aux, out, n, cid, f_mul, f_ct, f_qjl);
        }
        // x == c (or zero with centering none): f_mul = 0, so d = ||q - c||^2 exactly.
        common::storeUnaligned<float>(out + off_sq_, nsq);
        common::storeUnaligned<float>(out + off_mul_, f_mul);
        if (cfg_.centering == Centering::Ivf) {
            common::storeUnaligned<float>(out + off_ct_, f_ct);
            common::storeUnaligned<uint16_t>(out + off_cid_, static_cast<uint16_t>(cid));
        }
        if (cfg_.qjl) common::storeUnaligned<float>(out + off_qjl_, f_qjl);
    }

    void encodeHalf(const float *ru, uint8_t *out, float n, uint32_t cid, float &f_mul,
                    float &f_ct) const {
        float ur = 0.0f;
        for (size_t i = 0; i < D_; ++i) {
            const uint16_t h = common::floatToHalf(ru[i]);
            common::storeUnaligned<uint16_t>(out + 2 * i, h);
            ur += ru[i] * common::halfToFloat(h);
        }
        const float s = (cfg_.estimator == Estimator::Corrected && ur > 1e-6f) ? 1.0f / ur : 1.0f;
        f_mul = n * s;
        if (cfg_.centering == Centering::Ivf) {
            const float *rc = rc_.data() + static_cast<size_t>(cid) * D_;
            float acc = 0.0f;
            for (size_t i = 0; i < D_; ++i)
                acc += rc[i] * common::halfToFloat(common::loadUnaligned<uint16_t>(out + 2 * i));
            f_ct = f_mul * acc;
        }
    }

    // ru: rotated unit residual; e: D-float scratch. Writes the payload.
    void encodeIndex(const float *ru, float *e, uint8_t *out, float n, uint32_t cid,
                     float &f_mul, float &f_ct, float &f_qjl) const {
        const LloydMaxQuantizer &lm = lloydMax(mse_bits_);
        const float *bnd = lm.boundaries.data();
        const float *cent = lm.centroids.data();
        static thread_local std::vector<uint8_t> units;
        if (units.size() < D_) units.resize(D_);
        float ur = 0.0f;
        for (size_t i = 0; i < D_; ++i) {
            const uint32_t idx = quantizeIndex(bnd, mse_bits_, ru[i] * sqrt_d_);
            units[i] = static_cast<uint8_t>(idx);
            ur += ru[i] * cent[idx];
        }
        ur /= sqrt_d_;
        const float s = (cfg_.estimator == Estimator::Corrected && ur > 1e-6f) ? 1.0f / ur : 1.0f;
        f_mul = n * s / sqrt_d_;

        if (cfg_.qjl) {  // unit = idx << 1 | sign(S (ru - rec))
            float gsq = 0.0f;
            for (size_t i = 0; i < D_; ++i) {
                e[i] = ru[i] - cent[units[i]] / sqrt_d_;
                gsq += e[i] * e[i];
            }
            qjl_rot_->applyPadded(e);
            for (size_t i = 0; i < D_; ++i)
                units[i] = static_cast<uint8_t>((units[i] << 1) | (e[i] >= 0.0f ? 1u : 0u));
            f_qjl = n * std::sqrt(static_cast<float>(common::kPi) / 2.0f) / sqrt_d_ * std::sqrt(gsq);
        }
        if (cfg_.bits == 4) {
            for (size_t i = 0; i < D_; i += 2)
                out[i >> 1] = static_cast<uint8_t>(units[i] | (units[i + 1] << 4));
        } else {
            std::memcpy(out, units.data(), D_);
        }
        if (cfg_.centering == Centering::Ivf) {
            const float *rc = rc_.data() + static_cast<size_t>(cid) * D_;
            const float *src = cfg_.qjl ? src_.data() + static_cast<size_t>(cid) * D_ : nullptr;
            float acc = 0.0f, accq = 0.0f;
            for (size_t i = 0; i < D_; ++i) {
                acc += rc[i] * table_a_[units[i]];
                if (src) accq += src[i] * table_b_[units[i]];
            }
            f_ct = f_mul * acc + f_qjl * accq;
        }
    }

    TurboQuantConfig cfg_;
    common::Rotation rot_;
    std::optional<common::Rotation> qjl_rot_;
    Centerer centerer_;
    size_t D_ = 0;
    float sqrt_d_ = 1.0f;
    common::Isa isa_ = common::Isa::Scalar;
    int num_threads_ = 1;
    int mse_bits_ = 0;
    kernels::Payload payload_ = kernels::Payload::Nibble;
    kernels::Kernels kernels_{};
    size_t payload_bytes_ = 0, code_size_ = 0;
    size_t off_sq_ = 0, off_mul_ = 0, off_ct_ = 0, off_qjl_ = 0, off_cid_ = 0;
    std::vector<float> table_a_, table_b_;  // decode tables (16 or 256 entries)
    size_t k_ = 0;
    std::vector<float> rc_;   // [k, D] R c_j            (ivf)
    std::vector<float> src_;  // [k, D] S R c_j          (ivf + qjl)
    std::vector<float> cc_;   // [k, k] ||c_a - c_b||^2  (ivf)
};

}  // namespace vsq::turboquant
