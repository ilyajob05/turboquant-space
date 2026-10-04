#pragma once
// Build-wide configuration shared by every space header.
//
//   * architecture / ISA detection        VSQ_X86, VSQ_NEON
//   * runtime-dispatched AVX2 kernels     VSQ_HAVE_AVX2_KERNELS,
//                                         VSQ_TARGET_AVX2
//   * OpenMP helpers                      VSQ_OMP_PARALLEL_FOR
//
// AVX2 policy (same idea as hnswlib's SIMD dispatch): on GCC/Clang the AVX2
// kernels carry __attribute__((target("avx2,fma,..."))), so a translation unit
// compiled for the x86-64 baseline (SSE2) still contains them. A space picks a
// kernel once, at construction, from detectIsa(). On MSVC the kernels exist
// only when the TU itself is compiled with /arch:AVX2. NEON is the aarch64
// baseline and needs no dispatch.
//
// Every macro is prefixed VSQ_ so the headers can be copied next to
// hnswlib (which defines USE_SSE / USE_AVX / PORTABLE_ALIGN32) without clashes.

#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define VSQ_X86 1
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#  define VSQ_NEON 1
#  include <arm_neon.h>
#endif

#if defined(VSQ_X86) && (defined(__GNUC__) || defined(__clang__))
#  define VSQ_HAVE_AVX2_KERNELS 1
#  define VSQ_TARGET_AVX2 \
      __attribute__((target("avx2,fma,bmi,bmi2,popcnt,f16c")))
// Helpers called from AVX2 kernels: same target, always inlined.
#  define VSQ_AVX2_INLINE \
      __attribute__((target("avx2,fma,bmi,bmi2,popcnt,f16c"), always_inline)) inline
#  include <cpuid.h>
#  include <immintrin.h>
#elif defined(VSQ_X86) && defined(_MSC_VER) && defined(__AVX2__)
#  define VSQ_HAVE_AVX2_KERNELS 1
#  define VSQ_TARGET_AVX2
#  define VSQ_AVX2_INLINE __forceinline
#  include <immintrin.h>
#  include <intrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define VSQ_RESTRICT __restrict__
#  define VSQ_ALWAYS_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#  define VSQ_RESTRICT __restrict
#  define VSQ_ALWAYS_INLINE __forceinline
#else
#  define VSQ_RESTRICT
#  define VSQ_ALWAYS_INLINE inline
#endif

// OpenMP is optional. Batch helpers parallelise only above a size threshold;
// per-pair kernels never spawn threads (hnswlib owns its threading).
#if defined(VSQ_HAVE_OPENMP)
#  include <omp.h>
#  define VSQ_OMP_STRINGIFY(x) #x
#  define VSQ_OMP_PRAGMA(x) _Pragma(VSQ_OMP_STRINGIFY(x))
#  define VSQ_OMP_PARALLEL_FOR(nt, n)                                   \
      VSQ_OMP_PRAGMA(omp parallel for schedule(static)                  \
                            num_threads(nt) if ((n) > 64))
#  define VSQ_OMP_PARALLEL_FOR_DYNAMIC(nt, n)                           \
      VSQ_OMP_PRAGMA(omp parallel for schedule(dynamic, 1)              \
                            num_threads(nt) if ((n) > 1))
#else
#  define VSQ_OMP_PARALLEL_FOR(nt, n)
#  define VSQ_OMP_PARALLEL_FOR_DYNAMIC(nt, n)
#endif

namespace vsq::common {

constexpr double kPi = 3.14159265358979323846;

// Kernel family a space dispatches to.
enum class Isa : uint8_t { Scalar = 0, Neon = 1, Avx2 = 2 };

inline const char *isaName(Isa isa) {
    switch (isa) {
    case Isa::Neon: return "neon";
    case Isa::Avx2: return "avx2";
    default: return "scalar";
    }
}

namespace detail {

#if defined(VSQ_HAVE_AVX2_KERNELS)
// AVX2 + FMA + F16C + BMI1/2 + POPCNT, and the OS saves YMM state (XCR0 bits 1,2).
inline bool cpuSupportsAvx2() {
#  if defined(_MSC_VER) && !defined(__clang__)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const unsigned ecx1 = static_cast<unsigned>(r[2]);
    __cpuidex(r, 7, 0);
    const unsigned ebx7 = static_cast<unsigned>(r[1]);
    const bool os_ymm = (ecx1 & (1u << 27)) && ((_xgetbv(0) & 0x6) == 0x6);
#  else
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(1, 0, eax, ebx, ecx, edx);
    const unsigned ecx1 = ecx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    const unsigned ebx7 = ebx;
    bool os_ymm = false;
    if (ecx1 & (1u << 27)) {  // OSXSAVE
        unsigned lo = 0, hi = 0;
        __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        os_ymm = (lo & 0x6) == 0x6;
    }
#  endif
    const bool fma = ecx1 & (1u << 12);
    const bool popcnt = ecx1 & (1u << 23);
    const bool f16c = ecx1 & (1u << 29);
    const bool avx2 = ebx7 & (1u << 5);
    const bool bmi1 = ebx7 & (1u << 3);
    const bool bmi2 = ebx7 & (1u << 8);
    return os_ymm && fma && popcnt && f16c && avx2 && bmi1 && bmi2;
}
#endif

}  // namespace detail

// Best kernel family for this CPU and build. Computed once per process.
inline Isa detectIsa() {
    static const Isa isa = [] {
#if defined(VSQ_NEON)
        return Isa::Neon;
#elif defined(VSQ_HAVE_AVX2_KERNELS)
        return detail::cpuSupportsAvx2() ? Isa::Avx2 : Isa::Scalar;
#else
        return Isa::Scalar;
#endif
    }();
    return isa;
}

// `requested` if this CPU/build supports it, otherwise the best available.
// Scalar is always honoured so tests can compare SIMD kernels against it.
inline Isa resolveIsa(Isa requested) {
    const Isa best = detectIsa();
    return (requested == Isa::Scalar || requested == best) ? requested : best;
}

// OpenMP threads for batch helpers: requested > 0 wins, else the runtime
// default; always 1 without OpenMP.
inline int resolveNumThreads(int requested) {
#if defined(VSQ_HAVE_OPENMP)
    return requested > 0 ? requested : omp_get_max_threads();
#else
    (void)requested;
    return 1;
#endif
}

// Population count of a 64-bit word (GCC/Clang builtin, MSVC intrinsic).
VSQ_ALWAYS_INLINE int popcount64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#elif defined(_MSC_VER) && defined(_M_X64)
    return static_cast<int>(__popcnt64(x));
#else
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return static_cast<int>((x * 0x0101010101010101ULL) >> 56);
#endif
}

// Unaligned scalar access. hnswlib slots carry no alignment guarantee, so
// code metadata is always read and written through memcpy (one mov on x86/arm).
template <typename T>
VSQ_ALWAYS_INLINE T loadUnaligned(const void *p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

template <typename T>
VSQ_ALWAYS_INLINE void storeUnaligned(void *p, T v) {
    std::memcpy(p, &v, sizeof(T));
}

}  // namespace vsq::common
