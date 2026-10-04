// Python bindings for vsq (vector search quantization).
//
// Exposed classes
//   TurboQuantSpace      code format v2: bits 4 | 8 | 16, optional QJL,
//                        centering none | mean | ivf
//   RaBitQSpace          RaBitQ / Extended RaBitQ, quantized-query 1-bit
//   TurboQuantFastScan   flat 1-to-N FastScan over 4-bit TurboQuant codes
//   RaBitQFastScan       two-stage FastScan search over RaBitQ codes
//
// Input contract (review item 1.3): every array argument must be
// C-contiguous with the exact dtype (float32 vectors, uint8 codes); strided
// views raise ValueError instead of being read with the wrong layout.
// Heavy C++ work runs with the GIL released (review item 3.6).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include "rabitq/fastscan.h"
#include "rabitq/space.h"
#include "turboquant/fastscan.h"
#include "turboquant/space.h"

namespace py = pybind11;

using vsq::common::Isa;
using vsq::common::RotationKind;

using vsq::turboquant::Centering;
using vsq::turboquant::Estimator;
using vsq::turboquant::QjlCrossTerm;
using vsq::turboquant::TurboQuantConfig;
using vsq::turboquant::TurboQuantFastScan;
using vsq::turboquant::TurboQuantSpace;

using vsq::rabitq::RaBitQFastScan;
using vsq::rabitq::RaBitQSpace;

namespace {

// ---- buffer validation ----------------------------------------------------------

bool isCContiguous(const py::buffer_info &info) {
    ssize_t expected = info.itemsize;
    for (ssize_t d = info.ndim - 1; d >= 0; --d) {
        if (info.shape[d] > 1 && info.strides[d] != expected) return false;
        expected *= info.shape[d];
    }
    return true;
}

// float32 [cols] (ndim 1) or [rows, cols] (ndim 2), C-contiguous.
struct FloatArray {
    const float *ptr;
    size_t rows;
};

FloatArray requireFloat(const py::buffer &buf, size_t cols, const char *name, int ndim) {
    const py::buffer_info info = buf.request();
    if (info.format != py::format_descriptor<float>::format())
        throw py::value_error(std::string(name) + ": expected float32, got format '" +
                              info.format + "'");
    if (info.ndim != ndim)
        throw py::value_error(std::string(name) + ": expected a " + std::to_string(ndim) +
                              "-D array, got " + std::to_string(info.ndim) + "-D");
    if (static_cast<size_t>(info.shape[ndim - 1]) != cols)
        throw py::value_error(std::string(name) + ": expected last dimension " +
                              std::to_string(cols) + ", got " +
                              std::to_string(info.shape[ndim - 1]));
    if (!isCContiguous(info))
        throw py::value_error(std::string(name) +
                              ": must be C-contiguous (use numpy.ascontiguousarray)");
    return {static_cast<const float *>(info.ptr),
            ndim == 2 ? static_cast<size_t>(info.shape[0]) : 1};
}

// uint8 codes: total bytes must be a multiple of code_size (exactly code_size
// when `single`). Returns the code count.
struct CodeArray {
    uint8_t *ptr;
    size_t n;
};

CodeArray requireCodes(const py::buffer &buf, size_t code_size, const char *name, bool writable,
                       bool single) {
    const py::buffer_info info = buf.request(writable);
    if (info.itemsize != 1) throw py::value_error(std::string(name) + ": must be a uint8 array");
    if (!isCContiguous(info))
        throw py::value_error(std::string(name) +
                              ": must be C-contiguous (use numpy.ascontiguousarray)");
    size_t total = 1;
    for (auto s : info.shape) total *= static_cast<size_t>(s);
    if (single ? total != code_size : (total % code_size != 0))
        throw py::value_error(std::string(name) + ": " + std::to_string(total) + " bytes is not " +
                              (single ? "" : "a multiple of ") +
                              "code_size_bytes = " + std::to_string(code_size));
    return {static_cast<uint8_t *>(info.ptr), total / code_size};
}

py::array_t<uint8_t> newCodes(size_t n, size_t code_size) {
    return py::array_t<uint8_t>({static_cast<ssize_t>(n), static_cast<ssize_t>(code_size)});
}

py::array_t<float> newMatrix(size_t m, size_t n) {
    return py::array_t<float>({static_cast<ssize_t>(m), static_cast<ssize_t>(n)});
}

// ---- option parsing ---------------------------------------------------------------

Isa parseIsa(const std::string &s) {
    if (s == "auto") return vsq::common::detectIsa();
    if (s == "scalar") return Isa::Scalar;
    if (s == "neon") return Isa::Neon;
    if (s == "avx2") return Isa::Avx2;
    throw py::value_error("isa must be 'auto', 'scalar', 'neon' or 'avx2'");
}

Centering parseCentering(const std::string &s) {
    if (s == "none") return Centering::None;
    if (s == "mean") return Centering::Mean;
    if (s == "ivf") return Centering::Ivf;
    throw py::value_error("centering must be 'none', 'mean' or 'ivf'");
}

Estimator parseEstimator(const std::string &s) {
    if (s == "plain") return Estimator::Plain;
    if (s == "corrected") return Estimator::Corrected;
    throw py::value_error("estimator must be 'plain' or 'corrected'");
}

QjlCrossTerm parseQjlCross(const std::string &s) {
    if (s == "linear") return QjlCrossTerm::Linear;
    if (s == "arcsine") return QjlCrossTerm::Arcsine;
    throw py::value_error("qjl_cross must be 'linear' or 'arcsine'");
}

RotationKind parseRotation(const std::string &s) {
    if (s == "kac") return RotationKind::BlockKac;
    if (s == "legacy") return RotationKind::LegacySrht;
    throw py::value_error("rotation must be 'kac' or 'legacy'");
}

// ---- shared method shapes -----------------------------------------------------------
// Each helper binds one method shape for any space type S; the callable `fn`
// does the C++ work and runs with the GIL released.

template <class S, class Fn>
void defSymmetricBatch(py::class_<S> &cls, const char *name, Fn fn) {
    cls.def(name,
            [fn](const S &self, py::buffer A, py::buffer B) {
                const CodeArray ca = requireCodes(A, self.codeSizeBytes(), "codes_a", false, false);
                const CodeArray cb = requireCodes(B, self.codeSizeBytes(), "codes_b", false, false);
                py::array_t<float> out = newMatrix(ca.n, cb.n);
                float *dst = out.mutable_data();
                py::gil_scoped_release nogil;
                fn(self, ca.ptr, ca.n, cb.ptr, cb.n, dst);
                return out;
            },
            py::arg("codes_a"), py::arg("codes_b"));
}

template <class S, class Fn>
void defSymmetricPair(py::class_<S> &cls, const char *name, Fn fn) {
    cls.def(name,
            [fn](const S &self, py::buffer a, py::buffer b) {
                const CodeArray ca = requireCodes(a, self.codeSizeBytes(), "code_a", false, true);
                const CodeArray cb = requireCodes(b, self.codeSizeBytes(), "code_b", false, true);
                return fn(self, ca.ptr, cb.ptr);
            },
            py::arg("code_a"), py::arg("code_b"));
}

// distance(query, code), distance_1_to_n(query, codes), distance_m_to_n(Q, codes).
template <class S>
void defAsymmetric(py::class_<S> &cls) {
    cls.def("distance",
            [](const S &self, py::buffer q, py::buffer code) {
                const FloatArray a = requireFloat(q, self.dim(), "query", 1);
                const CodeArray c = requireCodes(code, self.codeSizeBytes(), "code", false, true);
                if constexpr (std::is_same_v<S, RaBitQSpace>) return self.distanceRaw(a.ptr, c.ptr);
                else return self.distance(a.ptr, c.ptr);
            },
            py::arg("query"), py::arg("code"))
        .def("distance_1_to_n",
             [](const S &self, py::buffer q, py::buffer codes) {
                 const FloatArray a = requireFloat(q, self.dim(), "query", 1);
                 const CodeArray c = requireCodes(codes, self.codeSizeBytes(), "codes", false, false);
                 py::array_t<float> out(static_cast<ssize_t>(c.n));
                 float *dst = out.mutable_data();
                 py::gil_scoped_release nogil;
                 self.distanceBatch1ToN(a.ptr, c.ptr, c.n, dst);
                 return out;
             },
             py::arg("query"), py::arg("codes"))
        .def("distance_m_to_n",
             [](const S &self, py::buffer Q, py::buffer codes) {
                 const FloatArray a = requireFloat(Q, self.dim(), "queries", 2);
                 const CodeArray c = requireCodes(codes, self.codeSizeBytes(), "codes", false, false);
                 py::array_t<float> out = newMatrix(a.rows, c.n);
                 float *dst = out.mutable_data();
                 py::gil_scoped_release nogil;
                 self.distanceBatchMToN(a.ptr, a.rows, c.ptr, c.n, dst);
                 return out;
             },
             py::arg("queries"), py::arg("codes"));
}

// encode(x), encode_into(x, out), encode_batch(X), encode_batch_into(X, out).
template <class S, class EncOne, class EncBatch>
void defEncode(py::class_<S> &cls, EncOne one, EncBatch batch) {
    cls.def("encode",
            [one](const S &self, py::buffer x) {
                const FloatArray a = requireFloat(x, self.dim(), "x", 1);
                py::array_t<uint8_t> code(static_cast<ssize_t>(self.codeSizeBytes()));
                uint8_t *dst = code.mutable_data();
                py::gil_scoped_release nogil;
                one(self, a.ptr, dst);
                return code;
            },
            py::arg("x"))
        .def("encode_into",
             [one](const S &self, py::buffer x, py::buffer out) {
                 const FloatArray a = requireFloat(x, self.dim(), "x", 1);
                 const CodeArray c = requireCodes(out, self.codeSizeBytes(), "out", true, true);
                 py::gil_scoped_release nogil;
                 one(self, a.ptr, c.ptr);
             },
             py::arg("x"), py::arg("out"))
        .def("encode_batch",
             [batch](const S &self, py::buffer X) {
                 const FloatArray a = requireFloat(X, self.dim(), "X", 2);
                 py::array_t<uint8_t> codes = newCodes(a.rows, self.codeSizeBytes());
                 uint8_t *dst = codes.mutable_data();
                 py::gil_scoped_release nogil;
                 batch(self, a.ptr, a.rows, dst);
                 return codes;
             },
             py::arg("X"))
        .def("encode_batch_into",
             [batch](const S &self, py::buffer X, py::buffer out) {
                 const FloatArray a = requireFloat(X, self.dim(), "X", 2);
                 const CodeArray c = requireCodes(out, self.codeSizeBytes(), "out", true, false);
                 if (c.n != a.rows) throw py::value_error("out: wrong number of codes");
                 py::gil_scoped_release nogil;
                 batch(self, a.ptr, a.rows, c.ptr);
             },
             py::arg("X"), py::arg("out"));
}

// ---- TurboQuant -------------------------------------------------------------------

void bindTurboQuant(py::module_ &m) {
    py::class_<TurboQuantSpace> cls(m, "TurboQuantSpace",
                                    "TurboQuant code format v2 (see include/turboquant/space.h).");
    cls.def(py::init([](size_t dim, int bits, bool qjl, const std::string &estimator,
                        const std::string &qjl_cross, const std::string &centering,
                        size_t n_clusters, int rotation_rounds, uint64_t rot_seed,
                        uint64_t qjl_seed, int num_threads, const std::string &isa) {
                TurboQuantConfig c;
                c.dim = dim;
                c.bits = bits;
                c.qjl = qjl;
                c.estimator = parseEstimator(estimator);
                c.qjl_cross = parseQjlCross(qjl_cross);
                c.centering = parseCentering(centering);
                c.n_clusters = n_clusters;
                c.rotation_rounds = rotation_rounds;
                c.rot_seed = rot_seed;
                c.qjl_seed = qjl_seed;
                c.num_threads = num_threads;
                c.isa = parseIsa(isa);
                return TurboQuantSpace(c);
            }),
            py::arg("dim"), py::arg("bits") = 4, py::kw_only(), py::arg("qjl") = false,
            py::arg("estimator") = "corrected", py::arg("qjl_cross") = "linear",
            py::arg("centering") = "ivf", py::arg("n_clusters") = 256,
            py::arg("rotation_rounds") = 3, py::arg("rot_seed") = 42ULL,
            py::arg("qjl_seed") = 137ULL, py::arg("num_threads") = 0, py::arg("isa") = "auto")
        .def("dim", &TurboQuantSpace::dim)
        .def("padded_dim", &TurboQuantSpace::paddedDim)
        .def("bits", &TurboQuantSpace::bits)
        .def("qjl", &TurboQuantSpace::qjl)
        .def("centering",
             [](const TurboQuantSpace &s) { return vsq::turboquant::centeringName(s.centering()); })
        .def("n_clusters", &TurboQuantSpace::numClusters)
        .def("trained", &TurboQuantSpace::trained)
        .def("code_size_bytes", &TurboQuantSpace::codeSizeBytes)
        .def("kernel_isa", [](const TurboQuantSpace &s) { return vsq::common::isaName(s.kernelIsa()); })
        .def("num_threads", &TurboQuantSpace::numThreads)
        .def("rotation_rounds", [](const TurboQuantSpace &s) { return s.rotation().rounds(); })
        .def_static("format_version", [] { return vsq::turboquant::kTurboQuantFormatVersion; })
        .def("train",
             [](TurboQuantSpace &self, py::buffer X, uint64_t seed, int iters) {
                 const FloatArray a = requireFloat(X, self.dim(), "X", 2);
                 py::gil_scoped_release nogil;
                 self.train(a.ptr, a.rows, seed, iters);
             },
             py::arg("X"), py::arg("seed") = 1234ULL, py::arg("iters") = 10,
             "Fit the centering (mean or k-means centroids) on X [n, dim].")
        .def("set_centroids",
             [](TurboQuantSpace &self, py::buffer C) {
                 const FloatArray a = requireFloat(C, self.dim(), "centroids", 2);
                 self.setCentroids(a.ptr, a.rows);
             },
             py::arg("centroids"))
        .def("centroids",
             [](const TurboQuantSpace &self) {
                 const auto &c = self.centroids();
                 py::array_t<float> out = newMatrix(self.numClusters(), self.dim());
                 std::copy(c.begin(), c.end(), out.mutable_data());
                 return out;
             })
        .def("decode",
             [](const TurboQuantSpace &self, py::buffer code) {
                 const CodeArray c = requireCodes(code, self.codeSizeBytes(), "code", false, true);
                 py::array_t<float> out(static_cast<ssize_t>(self.dim()));
                 self.decode(c.ptr, out.mutable_data());
                 return out;
             },
             py::arg("code"), "Approximate reconstruction of the encoded vector.")
        .def(py::pickle(
            [](const TurboQuantSpace &self) {
                const std::vector<uint8_t> s = self.serialize();
                return py::make_tuple(py::bytes(reinterpret_cast<const char *>(s.data()), s.size()),
                                      self.numThreads());
            },
            [](const py::tuple &t) {
                if (t.size() != 2) throw py::value_error("TurboQuantSpace: bad pickle state");
                const std::string s = t[0].cast<std::string>();
                return TurboQuantSpace::deserialize(reinterpret_cast<const uint8_t *>(s.data()),
                                                    s.size(), t[1].cast<int>());
            }));
    defEncode(
        cls, [](const TurboQuantSpace &s, const float *x, uint8_t *o) { s.encode(x, o); },
        [](const TurboQuantSpace &s, const float *X, size_t n, uint8_t *o) { s.encodeBatch(X, n, o); });
    defAsymmetric(cls);
    defSymmetricPair(cls, "distance_symmetric", [](const TurboQuantSpace &s, const uint8_t *a,
                                                   const uint8_t *b) { return s.distanceSymmetric(a, b); });
    defSymmetricPair(cls, "distance_symmetric_full",
                     [](const TurboQuantSpace &s, const uint8_t *a, const uint8_t *b) {
                         return s.distanceSymmetricFull(a, b);
                     });
    defSymmetricBatch(cls, "distance_m_to_n_symmetric",
                      [](const TurboQuantSpace &s, const uint8_t *A, size_t m, const uint8_t *B,
                         size_t n, float *o) { s.distanceBatchMToNSymmetric(A, m, B, n, o); });
    defSymmetricBatch(cls, "distance_m_to_n_symmetric_full",
                      [](const TurboQuantSpace &s, const uint8_t *A, size_t m, const uint8_t *B,
                         size_t n, float *o) { s.distanceBatchMToNSymmetricFull(A, m, B, n, o); });
}

// ---- RaBitQ -------------------------------------------------------------------------

void bindRaBitQ(py::module_ &m) {
    py::class_<RaBitQSpace> cls(m, "RaBitQSpace");
    cls.def(py::init([](size_t dim, uint64_t rot_seed, py::object centroid, int bits,
                        py::object encode_mode, const std::string &rotation, int rotation_rounds,
                        py::object query_bits, int num_threads, const std::string &isa) {
                // None is the C++ sentinel -1: 1-bit -> algorithm1,
                // 4/8-bit -> fixed_scale. A string selects the mode.
                int mode = -1;
                if (!encode_mode.is_none()) {
                    const std::string name = py::cast<std::string>(encode_mode);
                    if (name == "algorithm1") mode = 0;
                    else if (name == "fixed_scale") mode = 1;
                    else if (name == "windowed_scale") mode = 2;
                    else
                        throw py::value_error(
                            "encode_mode must be algorithm1, fixed_scale, or windowed_scale");
                }
                const int qbits = query_bits.is_none() ? -1 : py::cast<int>(query_bits);
                const float *cp = nullptr;
                if (!centroid.is_none())
                    cp = requireFloat(py::cast<py::buffer>(centroid), dim, "centroid", 1).ptr;
                return RaBitQSpace(dim, rot_seed, cp, bits, mode, parseRotation(rotation),
                                   rotation_rounds, qbits, num_threads, parseIsa(isa));
            }),
            py::arg("dim"), py::arg("rot_seed") = 42, py::arg("centroid") = py::none(),
            py::arg("bits") = 1, py::arg("encode_mode") = py::none(), py::kw_only(),
            py::arg("rotation") = "kac", py::arg("rotation_rounds") = 3,
            py::arg("query_bits") = py::none(), py::arg("num_threads") = 0, py::arg("isa") = "auto")
        .def("encode_mode", &RaBitQSpace::encodeModeName)
        .def("fixed_scale", &RaBitQSpace::fixedScale)
        .def("dim", &RaBitQSpace::dim)
        .def("padded_dim", &RaBitQSpace::paddedDim)
        .def("bits", &RaBitQSpace::bits)
        .def("query_bits", &RaBitQSpace::queryBits)
        .def("rotation",
             [](const RaBitQSpace &s) {
                 return s.rotationKind() == RotationKind::LegacySrht ? "legacy" : "kac";
             })
        .def("code_size_bytes", &RaBitQSpace::codeSizeBytes)
        .def("query_size_bytes", &RaBitQSpace::querySizeBytes)
        .def("distance_kernel", &RaBitQSpace::distanceKernel)
        .def("distance_scalar",
             [](const RaBitQSpace &self, py::buffer q, py::buffer code) {
                 const FloatArray a = requireFloat(q, self.dim(), "query", 1);
                 const CodeArray c = requireCodes(code, self.codeSizeBytes(), "code", false, true);
                 return self.distanceRawScalar(a.ptr, c.ptr);
             },
             py::arg("query"), py::arg("code"))
        .def("distance_bound",
             [](const RaBitQSpace &self, py::buffer q, py::buffer code, float eps0) {
                 const FloatArray a = requireFloat(q, self.dim(), "query", 1);
                 const CodeArray c = requireCodes(code, self.codeSizeBytes(), "code", false, true);
                 std::vector<uint8_t> pq(self.querySizeBytes());
                 self.prepareQuery(a.ptr, pq.data());
                 float lo = 0.f, hi = 0.f;
                 const float d = self.distanceBound(pq.data(), c.ptr, &lo, &hi, eps0);
                 return std::make_tuple(d, lo, hi);
             },
             py::arg("query"), py::arg("code"), py::arg("eps0") = RaBitQSpace::kDefaultEps0,
             "(estimate, lower, upper) of the squared distance from the RaBitQ error bound.");
    defEncode(
        cls, [](const RaBitQSpace &s, const float *x, uint8_t *o) { s.encode(x, o); },
        [](const RaBitQSpace &s, const float *X, size_t n, uint8_t *o) { s.encodeBatch(X, n, o); });
    defAsymmetric(cls);
}

// ---- FastScan -----------------------------------------------------------------------

template <class Index>
FloatArray requireQuery(const Index &self, const py::buffer &q) {
    return requireFloat(q, self.dim(), "query", 1);
}

void bindFastScan(py::module_ &m) {
    py::class_<TurboQuantFastScan>(m, "TurboQuantFastScan",
                                   "FastScan flat index over 4-bit TurboQuantSpace codes.")
        .def(py::init([](const TurboQuantSpace &space, py::buffer codes) {
                 const CodeArray c = requireCodes(codes, space.codeSizeBytes(), "codes", false, false);
                 py::gil_scoped_release nogil;
                 return TurboQuantFastScan(space, c.ptr, c.n);
             }),
             py::arg("space"), py::arg("codes"), py::keep_alive<1, 2>())
        .def("__len__", &TurboQuantFastScan::size)
        .def("scan",
             [](const TurboQuantFastScan &self, py::buffer q) {
                 const FloatArray a = requireQuery(self, q);
                 py::array_t<float> out(static_cast<ssize_t>(self.size()));
                 float *dst = out.mutable_data();
                 py::gil_scoped_release nogil;
                 self.scan(a.ptr, dst);
                 return out;
             },
             py::arg("query"), "Approximate (LUT-quantized) distances to every code.")
        .def("search",
             [](const TurboQuantFastScan &self, py::buffer q, size_t k, size_t rerank) {
                 const FloatArray a = requireQuery(self, q);
                 const size_t kk = std::min(k, self.size());
                 py::array_t<uint32_t> ids(static_cast<ssize_t>(kk));
                 py::array_t<float> dists(static_cast<ssize_t>(kk));
                 uint32_t *pi = ids.mutable_data();
                 float *pd = dists.mutable_data();
                 {
                     py::gil_scoped_release nogil;
                     self.search(a.ptr, kk, rerank, pi, pd);
                 }
                 return std::make_tuple(ids, dists);
             },
             py::arg("query"), py::arg("k"), py::arg("rerank") = 4,
             "(ids, dists) of the k nearest codes; k * rerank candidates are re-scored.");

    py::class_<RaBitQFastScan>(m, "RaBitQFastScan",
                               "Two-stage FastScan search over RaBitQ codes (built from vectors).")
        .def(py::init([](const RaBitQSpace &space, py::buffer X, float eps0) {
                 const FloatArray a = requireFloat(X, space.dim(), "X", 2);
                 py::gil_scoped_release nogil;
                 return RaBitQFastScan(space, a.ptr, a.rows, eps0);
             }),
             py::arg("space"), py::arg("X"), py::arg("eps0") = RaBitQSpace::kDefaultEps0,
             py::keep_alive<1, 2>())
        .def("__len__", &RaBitQFastScan::size)
        .def("codes",
             [](const RaBitQFastScan &self) {
                 const auto &c = self.codes();
                 py::array_t<uint8_t> out(static_cast<ssize_t>(c.size()));
                 std::copy(c.begin(), c.end(), out.mutable_data());
                 return out;
             })
        .def("scan",
             [](const RaBitQFastScan &self, py::buffer q) {
                 const FloatArray a = requireQuery(self, q);
                 py::array_t<float> est(static_cast<ssize_t>(self.size()));
                 py::array_t<float> lower(static_cast<ssize_t>(self.size()));
                 float *pe = est.mutable_data();
                 float *pl = lower.mutable_data();
                 {
                     py::gil_scoped_release nogil;
                     self.scan(a.ptr, pe, pl);
                 }
                 return std::make_tuple(est, lower);
             },
             py::arg("query"), "(stage-1 estimate, lower bound) for every code.")
        .def("search",
             [](const RaBitQFastScan &self, py::buffer q, size_t k) {
                 const FloatArray a = requireQuery(self, q);
                 const size_t kk = std::min(k, self.size());
                 py::array_t<uint32_t> ids(static_cast<ssize_t>(kk));
                 py::array_t<float> dists(static_cast<ssize_t>(kk));
                 uint32_t *pi = ids.mutable_data();
                 float *pd = dists.mutable_data();
                 size_t refined = 0;
                 {
                     py::gil_scoped_release nogil;
                     refined = self.search(a.ptr, kk, pi, pd);
                 }
                 return std::make_tuple(ids, dists, refined);
             },
             py::arg("query"), py::arg("k"),
             "(ids, dists, n_refined): two-stage search; n_refined codes used the full code.");
}

}  // namespace

PYBIND11_MODULE(_vsq, m) {
    m.doc() = "TurboQuant / RaBitQ vector quantization for ANN search";
    m.def("detected_isa", [] { return vsq::common::isaName(vsq::common::detectIsa()); },
          "Kernel family this CPU/build dispatches to: 'avx2', 'neon' or 'scalar'.");
    bindTurboQuant(m);
    bindRaBitQ(m);
    bindFastScan(m);
}
