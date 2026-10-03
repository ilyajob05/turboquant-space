import time
import numpy as np
from turboquant import TurboQuantSpace as TQS


def _time_it(fn, min_time=0.05, max_repeats=200):
    """Run fn() enough times to accumulate >= min_time seconds, return mean seconds per call."""
    # one untimed warm-up
    fn()
    repeats = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        dt = time.perf_counter() - t0
        if dt >= min_time or repeats >= max_repeats:
            return dt / repeats
        # grow repeats geometrically toward the target
        repeats = min(max_repeats, max(repeats * 2, int(repeats * min_time / max(dt, 1e-9))))


def run_benchmark():
    """Kernel throughput microbenchmark.

    All rows report a single unit in the rightmost column: distances/sec for
    distance kernels, vectors/sec for the encoder. This makes asymmetric
    (1-to-N, float query vs code DB) and symmetric (M-to-N, code vs code)
    paths directly comparable per distance computed.
    """
    dims = [128, 512, 768, 1024, 2048, 4096]
    batch_sizes = [1, 50, 100, 1000, 5000, 10000]
    bits_list = [4, 8]

    header = (
        f"{'Dim':>6} | {'Batch':>7} | {'Task':>20} | {'Time (ms)':>10} | "
        f"{'Throughput':>15} | {'Unit':>8}"
    )

    for bits in bits_list:
        print()
        print(f"=== bits_per_coord = {bits} ===")
        print(header)
        print("-" * len(header))

        for dim in dims:
            space = TQS(dim, bits_per_coord=bits)
            for n in batch_sizes:
                X = np.random.randn(n, dim).astype(np.float32)
                Q = np.random.randn(n, dim).astype(np.float32)
                q_single = Q[0]

                # 1. Encoder: vectors/sec
                dt = _time_it(lambda: space.encode_batch(X))
                print(f"{dim:6} | {n:7} | {'Encoding':>20} | {dt * 1000:10.2f} | "
                      f"{int(n / dt):15,} | {'vec/s':>8}")

                codes = space.encode_batch(X)

                # 2. Asymmetric 1-to-N: n distances per call → distances/sec
                dt = _time_it(lambda: space.distance_1_to_n(q_single, codes))
                print(f"{dim:6} | {n:7} | {'1-to-N (asym)':>20} | {dt * 1000:10.2f} | "
                      f"{int(n / dt):15,} | {'dist/s':>8}")

                # 3. Symmetric M-to-N: n*n distances per call → distances/sec
                if n > 1:
                    codes_q = space.encode_batch(Q)
                    dt = _time_it(lambda: space.distance_m_to_n_symmetric(codes_q, codes))
                    total_pairs = n * n
                    print(f"{dim:6} | {n:7} | {'M-to-N (sym)':>20} | {dt * 1000:10.2f} | "
                          f"{int(total_pairs / dt):15,} | {'dist/s':>8}")
            print("-" * len(header))


def run_threading_benchmark():
    dim = 512
    n_codes = 50_000
    n_queries = 128
    bits = 8
    thread_counts = [1, 2, 4, 8]
    repeats = 5

    print()
    print(f"Threading benchmark: dim={dim}, codes={n_codes}, queries={n_queries}, bits={bits}")
    print(f"{'Threads':>8} | {'Reported':>8} | {'Encode (ms)':>12} | "
          f"{'1-to-N (ms)':>12} | {'M-to-N (ms)':>12} | {'Speedup':>8}")
    print("-" * 78)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_codes, dim)).astype(np.float32)
    Q = rng.standard_normal((n_queries, dim)).astype(np.float32)

    baseline_mton = None
    for nt in thread_counts:
        space = TQS(dim, bits_per_coord=bits, num_threads=nt)

        # warm-up (JIT/page faults)
        codes = space.encode_batch(X[:1024])
        _ = space.distance_1_to_n(Q[0], codes)

        # encode_batch
        t0 = time.perf_counter()
        for _ in range(repeats):
            codes = space.encode_batch(X)
        t_enc = (time.perf_counter() - t0) / repeats * 1000

        # 1-to-N
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = space.distance_1_to_n(Q[0], codes)
        t_1ton = (time.perf_counter() - t0) / repeats * 1000

        # M-to-N asymmetric
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = space.distance_m_to_n(Q, codes)
        t_mton = (time.perf_counter() - t0) / repeats * 1000

        if baseline_mton is None:
            baseline_mton = t_mton
        speedup = baseline_mton / t_mton

        print(f"{nt:8} | {space.num_threads():8} | {t_enc:12.2f} | "
              f"{t_1ton:12.2f} | {t_mton:12.2f} | {speedup:7.2f}x")


if __name__ == "__main__":
    run_benchmark()
    run_threading_benchmark()
