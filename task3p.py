"""
CSS 314 - Task 3: Prefix Scan (Step Efficiency vs. Work Efficiency)
Student: Tolegen Aset (ID 230103240)

Serial inclusive scan:   out[i] = out[i-1] + in[i]
Parallel chunked scan (2 passes over the data):
  Pass 1 (parallel): each worker computes the sum of its chunk
  Offset reduction (serial, tiny): exclusive scan of the chunk sums -> chunk offsets
  Pass 2 (parallel): each worker scans its chunk starting from its offset
Kernels are Numba nogil loops so the threads really run in parallel.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

STUDENT_ID = 230103240
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
REPEATS = 5


@njit(nogil=True, cache=True)
def serial_scan(a, out):
    out[0] = a[0]
    for i in range(1, a.shape[0]):
        out[i] = out[i - 1] + a[i]


@njit(nogil=True, cache=True)
def chunk_sum(a, lo, hi):
    s = 0
    for i in range(lo, hi):
        s += a[i]
    return s


@njit(nogil=True, cache=True)
def chunk_scan(a, out, lo, hi, offset):
    s = offset
    for i in range(lo, hi):
        s += a[i]
        out[i] = s


def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except ImportError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def parallel_scan(a, out, workers, ex):
    n = a.shape[0]
    step = -(-n // workers)
    bounds = [(s, min(s + step, n)) for s in range(0, n, step)]

    # Pass 1: local chunk sums (parallel)
    sums = list(ex.map(lambda b: chunk_sum(a, b[0], b[1]), bounds))

    # Offset reduction: exclusive scan of chunk sums (serial, only `workers` values)
    offsets = [0] * len(bounds)
    for k in range(1, len(bounds)):
        offsets[k] = offsets[k - 1] + sums[k - 1]

    # Pass 2: scan each chunk starting at its offset (parallel)
    list(ex.map(lambda kb: chunk_scan(a, out, kb[1][0], kb[1][1], offsets[kb[0]]),
                enumerate(bounds)))


def best_ms(fn):
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def main():
    workers = physical_cores()
    rng = np.random.default_rng(STUDENT_ID)
    a = rng.integers(1, 11, size=N, dtype=np.int64)  # values 1..10

    s_out = np.empty(N, dtype=np.int64)
    p_out = np.empty(N, dtype=np.int64)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        # warm-up (compile outside timed region)
        serial_scan(a[:16], s_out[:16])
        parallel_scan(a[:64], p_out[:64], workers, ex)

        t_serial = best_ms(lambda: serial_scan(a, s_out))
        t_par = best_ms(lambda: parallel_scan(a, p_out, workers, ex))

    ok = bool(p_out[-1] == s_out[-1])
    status = "PASS" if ok else "FAIL"
    speedup = t_serial / t_par

    lines = [
        "=" * 58,
        "Task 3: Prefix Scan (Step vs. Work Efficiency)",
        "=" * 58,
        f"Student ID    : {STUDENT_ID}",
        f"Array size    : {N:,} ints (1..10), seed={STUDENT_ID}",
        f"Workers       : {workers} threads (physical cores)",
        f"Timing        : best of {REPEATS} runs",
        f"Last element  : serial={s_out[-1]}  parallel={p_out[-1]}",
        "-" * 58,
        f"Serial Time (ms)   : {t_serial:.3f}",
        f"Parallel Time (ms) : {t_par:.3f}",
        f"Speedup            : {speedup:.2f}x",
        f"Verification       : {status}",
    ]
    out = "\n".join(lines)
    print(out)
    with open("task3p_output.txt", "w", encoding="utf-8") as f:
        f.write(out + "\n")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()