#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "turboquant/space_turbo_quant.h"

namespace py = pybind11;
using turboquant::TurboQuantSpace;
using turboquant::TurboQuantPreparedQuery;
using turboquant::TurboQuantPreparedSymCode;

namespace {

const float *as_float_ptr(py::buffer buf, ssize_t expected_last,
                          const char *name, int expected_ndim,
                          ssize_t *out_rows = nullptr) {
    auto info = buf.request();
    if (info.format != py::format_descriptor<float>::format())
        throw py::value_error(std::string(name) + ": expected float32");
    if (info.ndim != expected_ndim)
        throw py::value_error(std::string(name) + ": wrong ndim");
    if (expected_ndim == 1) {
        if (info.shape[0] != expected_last)
            throw py::value_error(std::string(name) + ": wrong length");
    } else {
        if (info.shape[1] != expected_last)
            throw py::value_error(std::string(name) + ": wrong dim");
        if (out_rows)
            *out_rows = info.shape[0];
    }
    // require C-contiguous
    ssize_t stride_last = info.strides[info.ndim - 1];
    if (stride_last != static_cast<ssize_t>(sizeof(float)))
        throw py::value_error(std::string(name) + ": must be C-contiguous float32");
    return static_cast<const float *>(info.ptr);
}

void *as_bytes_ptr(py::buffer buf, ssize_t expected_bytes,
                   const char *name, bool writable) {
    auto info = buf.request(writable);
    if (info.itemsize != 1)
        throw py::value_error(std::string(name) + ": must be uint8");
    ssize_t total = 1;
    for (auto s : info.shape)
        total *= s;
    if (total != expected_bytes)
        throw py::value_error(std::string(name) + ": wrong byte size");
    return info.ptr;
}

} // namespace

PYBIND11_MODULE(_turboquant, m) {
    m.doc() = "TurboQuant — SIMD-accelerated 4/8-bit quantization for ANN";

    py::class_<TurboQuantPreparedQuery>(m, "PreparedQuery");
    py::class_<TurboQuantPreparedSymCode>(m, "PreparedSymCode");

    py::class_<TurboQuantSpace>(m, "TurboQuantSpace")
        .def(py::init<size_t, int, uint64_t, uint64_t, int>(),
             py::arg("dim"),
             py::arg("bits_per_coord") = 4,
             py::arg("rot_seed") = 42ULL,
             py::arg("qjl_seed") = 137ULL,
             py::arg("num_threads") = 0)

        .def("dim", &TurboQuantSpace::dim)
        .def("padded_dim", &TurboQuantSpace::paddedDim)
        .def("padded", &TurboQuantSpace::padded)
        .def("num_threads", &TurboQuantSpace::numThreads)
        .def("code_size_bytes", &TurboQuantSpace::codeSizeBytes)
        .def("bits_per_coord", &TurboQuantSpace::bitsPerCoord)

        // === single encode ===
        .def("encode",
             [](const TurboQuantSpace &self, py::buffer input) {
                 const float *data = as_float_ptr(input, self.dim(), "input", 1);
                 auto code = py::array_t<uint8_t>(self.codeSizeBytes());
                 self.encodeVector(data, code.mutable_data());
                 return code;
             },
             py::arg("x"))

        .def("encode_into",
             [](const TurboQuantSpace &self, py::buffer input, py::buffer code) {
                 const float *idata = as_float_ptr(input, self.dim(), "input", 1);
                 void *cdata = as_bytes_ptr(code, self.codeSizeBytes(), "code", true);
                 self.encodeVector(idata, cdata);
             },
             py::arg("x"), py::arg("out"))

        // === batch encode (m × dim) → (m * code_size) bytes ===
        .def("encode_batch",
             [](const TurboQuantSpace &self, py::buffer input) {
                 ssize_t rows = 0;
                 const float *data = as_float_ptr(input, self.dim(), "input", 2, &rows);
                 const size_t n = static_cast<size_t>(rows);
                 auto codes = py::array_t<uint8_t>(
                     {static_cast<ssize_t>(n),
                      static_cast<ssize_t>(self.codeSizeBytes())});
                 self.encodeBatch(data, n, codes.mutable_data());
                 return codes;
             },
             py::arg("X"))

        .def("encode_batch_into",
             [](const TurboQuantSpace &self, py::buffer input, py::buffer codes) {
                 ssize_t rows = 0;
                 const float *data = as_float_ptr(input, self.dim(), "input", 2, &rows);
                 const size_t n = static_cast<size_t>(rows);
                 void *cdata = as_bytes_ptr(
                     codes, static_cast<ssize_t>(n * self.codeSizeBytes()),
                     "codes", true);
                 self.encodeBatch(data, n, cdata);
             },
             py::arg("X"), py::arg("out"))

        // === diagnostic: σ-normalised post-WHT coordinates (n × padded_dim) ===
        .def("rotated_coords_batch",
             [](const TurboQuantSpace &self, py::buffer input) {
                 ssize_t rows = 0;
                 const float *data = as_float_ptr(input, self.dim(), "input", 2, &rows);
                 const size_t n = static_cast<size_t>(rows);
                 const size_t d = self.paddedDim();
                 auto out = py::array_t<float>({static_cast<ssize_t>(n),
                                                static_cast<ssize_t>(d)});
                 self.rotatedCoordsBatch(data, n, out.mutable_data());
                 return out;
             },
             py::arg("X"),
             "Return (n, padded_dim) array of σ-normalised WHT coords. "
             "Mirrors the first two steps of encodeVector for distribution analysis.")

        // === single distance (asymmetric) ===
        .def("distance",
             [](const TurboQuantSpace &self, py::buffer query, py::buffer code) {
                 const float *q = as_float_ptr(query, self.dim(), "query", 1);
                 const void *c = as_bytes_ptr(code, self.codeSizeBytes(),
                                              "code", false);
                 return self.distance(q, c);
             },
             py::arg("query"), py::arg("code"))

        // === single distance (symmetric) ===
        .def("distance_symmetric",
             [](const TurboQuantSpace &self, py::buffer a, py::buffer b) {
                 const void *ca = as_bytes_ptr(a, self.codeSizeBytes(),
                                               "code_a", false);
                 const void *cb = as_bytes_ptr(b, self.codeSizeBytes(),
                                               "code_b", false);
                 return self.distanceSymmetric(ca, cb);
             },
             py::arg("code_a"), py::arg("code_b"))

        // === 1-to-N batch ===
        .def("distance_1_to_n",
             [](const TurboQuantSpace &self, py::buffer query, py::buffer codes) {
                 const float *q = as_float_ptr(query, self.dim(), "query", 1);
                 auto cinfo = codes.request();
                 if (cinfo.itemsize != 1)
                     throw py::value_error("codes must be uint8");
                 const size_t cs = self.codeSizeBytes();
                 ssize_t total = 1;
                 for (auto s : cinfo.shape) total *= s;
                 if (total % static_cast<ssize_t>(cs) != 0)
                     throw py::value_error("codes size not a multiple of code_size_bytes");
                 const size_t n = total / cs;
                 auto out = py::array_t<float>(static_cast<ssize_t>(n));
                 self.distanceBatch1ToN(q, cinfo.ptr, n, out.mutable_data());
                 return out;
             },
             py::arg("query"), py::arg("codes"))

        // === M-to-N batch (asymmetric) ===
        .def("distance_m_to_n",
             [](const TurboQuantSpace &self, py::buffer queries, py::buffer codes) {
                 ssize_t m = 0;
                 const float *q = as_float_ptr(queries, self.dim(), "queries", 2, &m);
                 auto cinfo = codes.request();
                 if (cinfo.itemsize != 1)
                     throw py::value_error("codes must be uint8");
                 const size_t cs = self.codeSizeBytes();
                 ssize_t total = 1;
                 for (auto s : cinfo.shape) total *= s;
                 if (total % static_cast<ssize_t>(cs) != 0)
                     throw py::value_error("codes size not a multiple of code_size_bytes");
                 const size_t n = total / cs;
                 auto out = py::array_t<float>({static_cast<ssize_t>(m),
                                                static_cast<ssize_t>(n)});
                 self.distanceBatchMToN(q, static_cast<size_t>(m),
                                        cinfo.ptr, n, out.mutable_data());
                 return out;
             },
             py::arg("queries"), py::arg("codes"))

        // === M-to-N batch (symmetric) ===
        .def("distance_m_to_n_symmetric",
             [](const TurboQuantSpace &self, py::buffer codes_a, py::buffer codes_b) {
                 const size_t cs = self.codeSizeBytes();
                 auto ai = codes_a.request();
                 auto bi = codes_b.request();
                 if (ai.itemsize != 1 || bi.itemsize != 1)
                     throw py::value_error("codes must be uint8");
                 ssize_t ta = 1, tb = 1;
                 for (auto s : ai.shape) ta *= s;
                 for (auto s : bi.shape) tb *= s;
                 if (ta % static_cast<ssize_t>(cs) != 0 ||
                     tb % static_cast<ssize_t>(cs) != 0)
                     throw py::value_error("codes size not a multiple of code_size_bytes");
                 const size_t m = ta / cs;
                 const size_t n = tb / cs;
                 auto out = py::array_t<float>({static_cast<ssize_t>(m),
                                                static_cast<ssize_t>(n)});
                 self.distanceBatchMToNSymmetric(ai.ptr, m, bi.ptr, n,
                                                 out.mutable_data());
                 return out;
             },
             py::arg("codes_a"), py::arg("codes_b"))

        // === M-to-N symmetric with full QJL correction ===
        .def("distance_m_to_n_symmetric_full",
             [](const TurboQuantSpace &self, py::buffer codes_a, py::buffer codes_b) {
                 const size_t cs = self.codeSizeBytes();
                 auto ai = codes_a.request();
                 auto bi = codes_b.request();
                 if (ai.itemsize != 1 || bi.itemsize != 1)
                     throw py::value_error("codes must be uint8");
                 ssize_t ta = 1, tb = 1;
                 for (auto s : ai.shape) ta *= s;
                 for (auto s : bi.shape) tb *= s;
                 if (ta % static_cast<ssize_t>(cs) != 0 ||
                     tb % static_cast<ssize_t>(cs) != 0)
                     throw py::value_error("codes size not a multiple of code_size_bytes");
                 const size_t m = ta / cs;
                 const size_t n = tb / cs;
                 auto out = py::array_t<float>({static_cast<ssize_t>(m),
                                                 static_cast<ssize_t>(n)});
                 self.distanceBatchMToNSymmetricFull(ai.ptr, m, bi.ptr, n,
                                                     out.mutable_data());
                 return out;
             },
             py::arg("codes_a"), py::arg("codes_b"))

        // === M-to-N symmetric with light QJL correction ===
        .def("distance_m_to_n_symmetric_light",
             [](const TurboQuantSpace &self, py::buffer codes_a, py::buffer codes_b) {
                 const size_t cs = self.codeSizeBytes();
                 auto ai = codes_a.request();
                 auto bi = codes_b.request();
                 if (ai.itemsize != 1 || bi.itemsize != 1)
                     throw py::value_error("codes must be uint8");
                 ssize_t ta = 1, tb = 1;
                 for (auto s : ai.shape) ta *= s;
                 for (auto s : bi.shape) tb *= s;
                 if (ta % static_cast<ssize_t>(cs) != 0 ||
                     tb % static_cast<ssize_t>(cs) != 0)
                     throw py::value_error("codes size not a multiple of code_size_bytes");
                 const size_t m = ta / cs;
                 const size_t n = tb / cs;
                 auto out = py::array_t<float>({static_cast<ssize_t>(m),
                                                 static_cast<ssize_t>(n)});
                 self.distanceBatchMToNSymmetricLight(ai.ptr, m, bi.ptr, n,
                                                      out.mutable_data());
                 return out;
             },
             py::arg("codes_a"), py::arg("codes_b"))

        .def("prepare_query",
             [](const TurboQuantSpace &self, py::buffer query) {
                 const float *q = as_float_ptr(query, self.dim(), "query", 1);
                 return self.prepareQuery(q);
             },
             py::arg("query"))

        // === prepared symmetric (full QJL) ===
        .def("prepare_symmetric_query",
             [](const TurboQuantSpace &self, py::buffer code) {
                 const void *c = as_bytes_ptr(code, self.codeSizeBytes(),
                                              "code", false);
                 return self.prepareSymmetricQuery(c);
             },
             py::arg("code"))

        .def("distance_symmetric_full_prepared",
             [](const TurboQuantSpace &self,
                const TurboQuantPreparedSymCode &pq, py::buffer code_b) {
                 const void *cb = as_bytes_ptr(code_b, self.codeSizeBytes(),
                                               "code_b", false);
                 return self.distanceSymmetricFullPrepared(pq, cb);
             },
             py::arg("pq"), py::arg("code_b"))

        .def("distance_1_to_n_symmetric_full",
             [](const TurboQuantSpace &self,
                const TurboQuantPreparedSymCode &pq, py::buffer codes) {
                 auto cinfo = codes.request();
                 if (cinfo.itemsize != 1)
                     throw py::value_error("codes must be uint8");
                 const size_t cs = self.codeSizeBytes();
                 ssize_t total = 1;
                 for (auto s : cinfo.shape) total *= s;
                 if (total % static_cast<ssize_t>(cs) != 0)
                     throw py::value_error("codes size not a multiple of code_size_bytes");
                 const size_t n = total / cs;
                 auto out = py::array_t<float>(static_cast<ssize_t>(n));
                 self.distanceBatch1ToNSymmetricFull(pq, cinfo.ptr, n,
                                                     out.mutable_data());
                 return out;
             },
             py::arg("pq"), py::arg("codes"))

        .def("distance_m_to_n_symmetric_full_prepared",
             [](const TurboQuantSpace &self, py::buffer codes_a,
                py::buffer codes_b) {
                 const size_t cs = self.codeSizeBytes();
                 auto ai = codes_a.request();
                 auto bi = codes_b.request();
                 if (ai.itemsize != 1 || bi.itemsize != 1)
                     throw py::value_error("codes must be uint8");
                 ssize_t ta = 1, tb = 1;
                 for (auto s : ai.shape) ta *= s;
                 for (auto s : bi.shape) tb *= s;
                 if (ta % static_cast<ssize_t>(cs) != 0 ||
                     tb % static_cast<ssize_t>(cs) != 0)
                     throw py::value_error("codes size not a multiple of code_size_bytes");
                 const size_t m = ta / cs;
                 const size_t n = tb / cs;
                 auto out = py::array_t<float>({static_cast<ssize_t>(m),
                                                 static_cast<ssize_t>(n)});
                 self.distanceBatchMToNSymmetricFullPrepared(
                     ai.ptr, m, bi.ptr, n, out.mutable_data());
                 return out;
             },
             py::arg("codes_a"), py::arg("codes_b"));
}
