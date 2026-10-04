// C++ checks that the Python tests cannot express (review 2026-10-04):
//   1.4  spaces are copyable: a copy keeps working after the source dies
//   3.1  every SIMD kernel family agrees with the scalar reference
//   4    the kernel family actually dispatched on this CPU is reported
//
// Build: cmake -DVSQ_BUILD_TESTS=ON ... && ctest   (or run the binary)
// Exit code 0 on success; prints one line per check.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "rabitq/space.h"
#include "turboquant/fastscan.h"
#include "turboquant/space.h"

using vsq::common::Isa;
using vsq::common::detectIsa;
using vsq::rabitq::RaBitQSpace;
using namespace vsq::turboquant;

namespace {

int g_failures = 0;

void check(bool ok, const char *what) {
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what);
    if (!ok) ++g_failures;
}

std::vector<float> gaussian(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> N(0.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (float &x : v) x = N(rng) + 0.25f;
    return v;
}

bool close(float a, float b, float rel) {
    return std::fabs(a - b) <= rel * std::max(1.0f, std::fabs(b));
}

// ---- 1.4 copy safety -----------------------------------------------------------

void testCopySafety() {
    const size_t dim = 96, n = 400;
    const std::vector<float> X = gaussian(n, dim, 1);

    TurboQuantConfig cfg;
    cfg.dim = dim;
    cfg.n_clusters = 8;
    auto s1 = std::make_unique<TurboQuantSpace>(cfg);
    s1->train(X.data(), n);
    std::vector<uint8_t> c1(s1->codeSizeBytes());
    s1->encode(X.data(), c1.data());
    const float ref1 = s1->distance(X.data() + dim, c1.data());
    const TurboQuantSpace copy1 = *s1;
    s1.reset();  // destroy the source: the copy must not alias its tables
    check(close(copy1.distance(X.data() + dim, c1.data()), ref1, 1e-6f), "TurboQuant space copy outlives source");

    auto s2 = std::make_unique<RaBitQSpace>(dim, 3, nullptr, 4);
    std::vector<uint8_t> c2(s2->codeSizeBytes());
    s2->encode(X.data(), c2.data());
    const float ref2 = s2->distanceRaw(X.data() + dim, c2.data());
    const RaBitQSpace copy2 = *s2;
    s2.reset();
    check(close(copy2.distanceRaw(X.data() + dim, c2.data()), ref2, 1e-6f),
          "RaBitQ space copy outlives source");
}

// ---- 3.1 SIMD vs scalar --------------------------------------------------------

void testTurboQuantKernels() {
    const size_t dim = 200, n = 300, m = 6;
    const std::vector<float> X = gaussian(n, dim, 2), Q = gaussian(m, dim, 3);
    for (int bits : {4, 8, 16}) {
        for (bool qjl : {false, true}) {
            if (qjl && bits == 16) continue;
            TurboQuantConfig cfg;
            cfg.dim = dim;
            cfg.bits = bits;
            cfg.qjl = qjl;
            cfg.n_clusters = 8;
            TurboQuantConfig scfg = cfg;
            scfg.isa = Isa::Scalar;
            TurboQuantSpace fast(cfg), ref(scfg);
            fast.train(X.data(), n);
            ref.train(X.data(), n);
            std::vector<uint8_t> codes(n * fast.codeSizeBytes());
            fast.encodeBatch(X.data(), n, codes.data());
            std::vector<float> a(m * n), b(m * n), sa(n * 10), sb(n * 10);
            fast.distanceBatchMToN(Q.data(), m, codes.data(), n, a.data());
            ref.distanceBatchMToN(Q.data(), m, codes.data(), n, b.data());
            fast.distanceBatchMToNSymmetric(codes.data(), 10, codes.data(), n, sa.data());
            ref.distanceBatchMToNSymmetric(codes.data(), 10, codes.data(), n, sb.data());
            // d = base + f_sq - 2 ip cancels large terms, so reassociation noise
            // (FMA, two accumulators vs the serial scalar sum) is measured
            // against the typical distance, not against each small value.
            auto worstDiff = [](const std::vector<float> &x, const std::vector<float> &y) {
                double scale = 0.0;
                for (float v : y) scale += std::fabs(v);
                scale = std::max(1.0, scale / static_cast<double>(y.size()));
                double worst = 0.0;
                for (size_t i = 0; i < x.size(); ++i)
                    worst = std::max(worst, std::fabs(x[i] - y[i]) / std::max<double>(std::fabs(y[i]), scale));
                return static_cast<float>(worst);
            };
            const float worst = std::max(worstDiff(a, b), worstDiff(sa, sb));
            const bool ok = worst < 1e-4f;
            char msg[160];
            std::snprintf(msg, sizeof msg, "TurboQuant v2 bits=%d qjl=%d: %s == scalar (max rel diff %.2e)",
                          bits, static_cast<int>(qjl), isaName(fast.kernelIsa()), worst);
            check(ok, msg);
        }
    }
}

void testRaBitQKernels() {
    const size_t dim = 256, n = 200;
    const std::vector<float> X = gaussian(n, dim, 4), Q = gaussian(1, dim, 5);
    for (int bits : {1, 4, 8}) {
        RaBitQSpace s(dim, 7, nullptr, bits);
        std::vector<uint8_t> codes(n * s.codeSizeBytes());
        s.encodeBatch(X.data(), n, codes.data());
        std::vector<uint8_t> pq(s.querySizeBytes());
        s.prepareQuery(Q.data(), pq.data());
        bool ok = true;
        for (size_t i = 0; i < n; ++i) {
            const uint8_t *c = codes.data() + i * s.codeSizeBytes();
            ok &= close(s.distancePrepared(pq.data(), c), s.distancePreparedScalar(pq.data(), c), 1e-4f);
        }
        char msg[128];
        std::snprintf(msg, sizeof msg, "RaBitQ bits=%d kernel %s == scalar", bits,
                      s.distanceKernel().c_str());
        check(ok, msg);
    }
}

void testFastScanKernels() {
    const size_t dim = 128, n = 1000;
    const std::vector<float> X = gaussian(n, dim, 6), Q = gaussian(1, dim, 7);
    TurboQuantConfig cfg;
    cfg.dim = dim;
    cfg.centering = Centering::None;
    TurboQuantConfig scfg = cfg;
    scfg.isa = Isa::Scalar;
    TurboQuantSpace fast(cfg), ref(scfg);
    std::vector<uint8_t> codes(n * fast.codeSizeBytes());
    fast.encodeBatch(X.data(), n, codes.data());
    TurboQuantFastScan fa(fast, codes.data(), n), fb(ref, codes.data(), n);
    std::vector<float> a(n), b(n);
    fa.scan(Q.data(), a.data());
    fb.scan(Q.data(), b.data());
    bool ok = true;
    for (size_t i = 0; i < n; ++i) ok &= close(a[i], b[i], 1e-5f);  // integer sums: identical
    char msg[96];
    std::snprintf(msg, sizeof msg, "FastScan TQ4 %s scan == scalar scan", isaName(fast.kernelIsa()));
    check(ok, msg);
}

}  // namespace

int main() {
    std::printf("detected ISA: %s\n", isaName(detectIsa()));
    testCopySafety();
    testTurboQuantKernels();
    testRaBitQKernels();
    testFastScanKernels();
    std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
