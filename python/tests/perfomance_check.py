import time
import numpy as np
from turboquant import TurboQuantSpace as TQS


def run_benchmark():
    dims = [128, 768, 4096]
    batch_sizes = [1, 100, 10000]
    bits = 8

    print(f"{'Dim':>6} | {'Batch':>7} | {'Task':>15} | {'Time (ms)':>10} | {'Ops/sec':>12}")
    print("-" * 60)

    for dim in dims:
        space = TQS(dim, bits_per_coord=bits)
        for n in batch_sizes:
            # Data
            X = np.random.randn(n, dim).astype(np.float32)
            Q = np.random.randn(n, dim).astype(np.float32)

            # 1. Coding test
            start = time.perf_counter()
            codes = space.encode_batch(X)
            end = time.perf_counter()

            dt = (end - start)
            print(f"{dim:6} | {n:7} | {'Encoding':>15} | {dt * 1000:10.2f} | {int(n / dt):12,}")

            # 2. Тест Query-to-Batch Distance (Asymmetric)
            # Get 1 request and find of N codes (from ANN)
            q_single = Q[0]
            start = time.perf_counter()
            _ = space.distance_1_to_n(q_single, codes)
            end = time.perf_counter()

            dt = (end - start)
            print(f"{dim:6} | {n:7} | {'1-to-N Dist':>15} | {dt * 1000:10.2f} | {int(n / dt):12,}")

            # 3. Symmetric (Batch-to-Batch)
            if n > 1:
                codes_q = space.encode_batch(Q)
                start = time.perf_counter()
                _ = space.distance_m_to_n_symmetric(codes_q, codes)
                end = time.perf_counter()

                dt = (end - start)
                total_ops = n * n
                print(f"{dim:6} | {n:7} | {'M-to-N Sym':>15} | {dt * 1000:10.2f} | {int(total_ops / dt):12,}")
        print("-" * 60)


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
