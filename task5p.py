"""
CSS 314 - Task 5: Thread Oversubscription Threshold
Student: Tolegen Aset (ID 230103240)

Batch of 10,000 small CPU-bound tasks: each task hashes a 1 KB buffer 500 times
(FNV-1a 64-bit, chained: the result of round k seeds round k+1). The hash kernel is compiled
with Numba (nogil), so threads run truly in parallel instead of being serialized by the GIL.
The whole batch is run on thread pools of size [1, C/2, C, 2C, 4C, 16C, 256], C = logical cores.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

STUDENT_ID = 230103240
TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
ROUNDS = 500
REPEATS = 3
FNV_OFFSET = np.uint64(14695981039346656037)
FNV_PRIME = np.uint64(1099511628211)


@njit(nogil=True, cache=True)
def hash_task(buf, task_id):
    h = np.uint64(14695981039346656037) ^ np.uint64(task_id)
    prime = np.uint64(1099511628211)
    for _ in range(500):
        for i in range(buf.shape[0]):
            h = (h ^ np.uint64(buf[i])) * prime
    return h


def run_batch(buf, pool_size):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=pool_size) as ex:
        results = list(ex.map(lambda k: hash_task(buf, k), range(TASKS)))
    ms = (time.perf_counter() - t0) * 1000.0
    checksum = 0
    for r in results:
        checksum ^= int(r)
    return ms, checksum


def main():
    cores = os.cpu_count() or 1
    sizes = sorted({1, max(1, cores // 2), cores, cores * 2, cores * 4, cores * 16, 256})

    rng = np.random.default_rng(STUDENT_ID)
    buf = rng.integers(0, 256, size=1024, dtype=np.uint8)   # the 1 KB "string"
    hash_task(buf, 0)                                        # compile outside timing

    times, checks = [], set()
    for p in sizes:
        runs = []
        for _ in range(REPEATS):
            ms, cs = run_batch(buf, p)
            runs.append(ms)
            checks.add(cs)
        times.append(min(runs))
        print(f"  measured pool={p:<4d} {times[-1]:10.1f} ms", file=sys.stderr, flush=True)

    best_i = min(range(len(times)), key=times.__getitem__)
    status = []
    for i, p in enumerate(sizes):
        if i < best_i:
            status.append("Improving")
        elif i == best_i:
            status.append("Optimal (fastest)")
        elif i == best_i + 1:
            status.append("WORSENING <-- threshold")
        else:
            status.append("Degraded")

    lines = [
        "=" * 74,
        "Task 5: Thread Oversubscription Threshold",
        "=" * 74,
        f"Student ID    : {STUDENT_ID}",
        f"Logical cores : {cores}   (Cores = logical cores)",
        f"Batch         : {TASKS:,} tasks x (1 KB hashed {ROUNDS} times), best of {REPEATS} runs",
        f"Checksums     : {'all pool sizes identical' if len(checks) == 1 else 'MISMATCH'}",
        "-" * 74,
        f"{'Pool Size':>9} | {'Execution Time (ms)':>19} | {'Speedup vs 1 thread':>19} | Status",
        "-" * 74,
    ]
    for p, t, s in zip(sizes, times, status):
        lines.append(f"{p:>9} | {t:>19.1f} | {times[0] / t:>18.2f}x | {s}")
    lines.append("-" * 74)
    if best_i + 1 < len(sizes):
        lines.append(f"Fastest pool size: {sizes[best_i]}. Execution time starts worsening at pool size "
                     f"{sizes[best_i + 1]} (+{(times[best_i + 1] / times[best_i] - 1) * 100:.1f}% vs best).")
    else:
        lines.append(f"Fastest pool size: {sizes[best_i]}; no worsening observed up to {sizes[-1]} threads.")
    out = "\n".join(lines)
    print(out)
    with open("task5p_output.txt", "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()