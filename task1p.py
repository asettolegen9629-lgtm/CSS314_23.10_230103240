"""
CSS 314 - Task 1: Row-Major vs. Column-Major Parallel Cache Traversal
Student: Tolegen Aset (ID 230103240)

Both modes use the same explicit-loop kernel compiled by Numba (nogil), so the only
difference is the memory access pattern:
  Mode A: each worker owns a block of ROWS   -> inner loop walks a row (stride 8 B)
  Mode B: each worker owns a block of COLUMNS -> inner loop walks down a column
          (stride = 8000 * 8 B = 64,000 B -> a new cache line on every access)
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

STUDENT_ID = 230103240
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
REPEATS = 3


@njit(nogil=True, cache=True)
def sum_rows(a, r0, r1):
    """Mode A: rows r0..r1, linear walk through columns (cache friendly)."""
    s = 0.0
    for i in range(r0, r1):
        for j in range(a.shape[1]):
            s += a[i, j]
    return s


@njit(nogil=True, cache=True)
def sum_cols(a, c0, c1):
    """Mode B: columns c0..c1, walk across rows (strided, jumps cache lines)."""
    s = 0.0
    for j in range(c0, c1):
        for i in range(a.shape[0]):
            s += a[i, j]
    return s


def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except ImportError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)  # assume 2-way SMT


def chunks(n, parts):
    step = -(-n // parts)  # ceil
    return [(s, min(s + step, n)) for s in range(0, n, step)]


def run_parallel(kernel, a, workers):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(kernel, a, lo, hi) for lo, hi in chunks(a.shape[0], workers)]
        total = sum(f.result() for f in futs)
    return (time.perf_counter() - t0) * 1000.0, total


def best_of(kernel, a, workers):
    runs = [run_parallel(kernel, a, workers) for _ in range(REPEATS)]
    return min(r[0] for r in runs), runs[0][1]


def main():
    logical = os.cpu_count()
    workers = physical_cores()

    rng = np.random.default_rng(STUDENT_ID)
    a = rng.random((N, N), dtype=np.float64)  # C-contiguous (row-major) float64

    # warm-up: compile both kernels outside the timed region
    sum_rows(a[:8], 0, 8)
    sum_cols(a[:8], 0, 8)

    t_a, sum_a = best_of(sum_rows, a, workers)
    t_b, sum_b = best_of(sum_cols, a, workers)
    penalty = t_b / t_a

    lines = [
        "=" * 58,
        "Task 1: Row-Major vs Column-Major Parallel Cache Traversal",
        "=" * 58,
        f"Student ID          : {STUDENT_ID}",
        f"Logical cores       : {logical}",
        f"Physical cores used : {workers}",
        f"Matrix              : {N} x {N} float64 ({a.nbytes / 1e6:.0f} MB), seed={STUDENT_ID}",
        f"Timing              : best of {REPEATS} runs, wall-clock",
        "-" * 58,
        f"Mode A (row-major)    : {t_a:10.2f} ms   sum = {sum_a:.6f}",
        f"Mode B (column-major) : {t_b:10.2f} ms   sum = {sum_b:.6f}",
        f"Sums match            : {np.isclose(sum_a, sum_b, rtol=1e-9)}",
        "-" * 58,
        f"Cache penalty factor (B / A) : {penalty:.2f}x",
    ]
    out = "\n".join(lines)
    print(out)
    with open("task1p_output.txt", "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()