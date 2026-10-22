import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

N = 30_000_000
BLOCK = 1 << 16
REPEATS = 3
MAX_CORES = os.cpu_count()
COUNTS = sorted({1, 2, MAX_CORES})


def copy_chunk(a, b, lo, hi):
    np.copyto(b[lo:hi], a[lo:hi])


def compute_chunk(a, b, lo, hi):
    s = np.empty(BLOCK)
    c = np.empty(BLOCK)
    t = np.empty(BLOCK)
    for start in range(lo, hi, BLOCK):
        end = min(start + BLOCK, hi)
        m = end - start
        x = a[start:end]
        out = b[start:end]
        np.sin(x, out=s[:m])
        np.cos(x, out=c[:m])
        np.tan(x, out=t[:m])
        np.multiply(s[:m], c[:m], out=out)
        np.add(out, t[:m], out=out)


def run(kernel, a, b, workers, pool):
    bounds = np.linspace(0, len(a), workers + 1, dtype=np.int64).tolist()
    futures = [
        pool.submit(kernel, a, b, bounds[i], bounds[i + 1])
        for i in range(workers)
    ]
    for f in futures:
        f.result()


def bench(kernel, a, b, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        run(kernel, a, b, workers, pool)
        best = float("inf")
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            run(kernel, a, b, workers, pool)
            best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def scaling(name, kernel, a, b):
    print(f"{name}")
    header = f"{'Cores':>6} | {'Time (ms)':>10} | {'Speedup':>8} | {'Efficiency':>10}"
    print(header)
    print("-" * len(header))
    t1 = None
    last_eff = None
    for p in COUNTS:
        t = bench(kernel, a, b, p)
        if p == 1:
            t1 = t
        speedup = t1 / t
        eff = speedup / p
        last_eff = eff
        print(f"{p:>6} | {t:>10.2f} | {speedup:>7.2f}x | {eff:>9.1%}")
    print()
    return last_eff


def verify(a, b_copy, b_comp):
    idx = slice(0, 1000)
    x = a[idx]
    ok_copy = np.array_equal(b_copy[idx], x)
    ok_comp = np.allclose(b_comp[idx], np.sin(x) * np.cos(x) + np.tan(x))
    if not (ok_copy and ok_comp):
        raise RuntimeError("kernel result mismatch")


def main():
    print("Student ID    : 230103240")
    print(f"OS            : {platform.system()} {platform.release()}")
    print(f"Logical cores : {MAX_CORES}")
    print(f"Array size    : {N:,} float64 ({N * 8 / 1e6:.0f} MB per array)\n")

    a = np.random.default_rng(230103240).random(N)
    b_copy = np.empty(N)
    b_comp = np.empty(N)

    eff_a = scaling("Kernel A (memory-bound): B[i] = A[i]", copy_chunk, a, b_copy)
    eff_b = scaling(
        "Kernel B (compute-bound): B[i] = sin(A[i]) * cos(A[i]) + tan(A[i])",
        compute_chunk,
        a,
        b_comp,
    )
    verify(a, b_copy, b_comp)

    print(f"Efficiency at {MAX_CORES} cores: Kernel A = {eff_a:.1%}, Kernel B = {eff_b:.1%}")
    if eff_a < eff_b:
        print(
            "RESULT: Kernel A (vector copy) stalled in scaling due to memory channel "
            "saturation; Kernel B (sin/cos/tan) kept scaling because it is compute-bound."
        )
    else:
        print(
            "RESULT: Kernel B scaled worse than Kernel A on this run; memory saturation "
            "is not the dominant limit for the copy kernel here."
        )


if __name__ == "__main__":
    main()