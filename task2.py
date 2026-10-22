import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103240
N = 20_000_000
P = 0.95          # parallelizable fraction for Amdahl's Law
REPEATS = 5       # take the best run to reduce noise


def amdahl(p, par=P):
    return 1.0 / ((1.0 - par) + par / p)


def chunk_sum(data, start, end):
    return float(np.sum(data[start:end]))


def parallel_sum(data, workers, pool):
    bounds = np.linspace(0, len(data), workers + 1, dtype=np.int64)
    futures = [
        pool.submit(chunk_sum, data, int(bounds[i]), int(bounds[i + 1]))
        for i in range(workers)
    ]
    return sum(f.result() for f in futures)


def bench(data, workers):
    best = float("inf")
    result = 0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parallel_sum(data, workers, pool)  # warm-up
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            result = parallel_sum(data, workers, pool)
            best = min(best, time.perf_counter() - t0)
    return best * 1000.0, result


def main():
    max_cores = os.cpu_count()
    counts = sorted({1, 2, 4, max_cores, max_cores * 2})

    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS            : {platform.system()} {platform.release()}")
    print(f"Logical cores : {max_cores}")
    print(f"Array size    : {N:,} float64 | Amdahl P = {P}")

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(N, dtype=np.float64)

    serial_ref = float(np.sum(data))
    print(f"Serial sum    : {serial_ref:.6f}\n")

    header = f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup':>8} | {'Amdahl Upper Bound':>18}"
    print(header)
    print("-" * len(header))

    t1 = None
    for p in counts:
        t_ms, result = bench(data, p)
        if p == 1:
            t1 = t_ms
        speedup = t1 / t_ms
        print(f"{p:>6} | {t_ms:>14.2f} | {speedup:>7.2f}x | {amdahl(p):>17.2f}x")
        assert abs(result - serial_ref) < 1e-3 * abs(serial_ref), "sum mismatch"

    print(f"\nAmdahl limit for p -> infinity: {1 / (1 - P):.1f}x")


if __name__ == "__main__":
    main()