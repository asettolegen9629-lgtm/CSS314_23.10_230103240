import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

A = 2.5
B = 1.5
WORKERS = os.cpu_count()
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]


def serial(x, y):
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def chunk(x, y, lo, hi):
    ys = y[lo:hi]
    np.multiply(x[lo:hi], A, out=ys)
    np.add(ys, B, out=ys)


def parallel(x, y, pool, bounds):
    futures = [
        pool.submit(chunk, x, y, bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
    ]
    for f in futures:
        f.result()


def best_ms(fn, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        best = min(best, time.perf_counter_ns() - t0)
    return best / 1e6


def measure(n, pool):
    x = np.random.default_rng(230103240).random(n)
    y_s = np.empty(n)
    y_p = np.empty(n)
    bounds = np.linspace(0, n, WORKERS + 1, dtype=np.int64).tolist()
    repeats = max(7, min(3000, 3_000_000 // n))

    serial(x, y_s)
    parallel(x, y_p, pool, bounds)
    if not np.allclose(y_s, y_p):
        raise RuntimeError(f"result mismatch at N={n}")

    t_s = best_ms(lambda: serial(x, y_s), repeats)
    t_p = best_ms(lambda: parallel(x, y_p, pool, bounds), repeats)
    return t_s, t_p


def find_crossover(rows):
    k = None
    for idx in range(len(rows) - 1, -1, -1):
        if rows[idx][2] < rows[idx][1]:
            k = idx
        else:
            break
    return k


def print_table(rows):
    header = (
        f"{'N':>12} | {'Serial (ms)':>12} | {'Parallel (ms)':>13} | "
        f"{'Speedup':>8} | {'Winner':>8}"
    )
    print(header)
    print("-" * len(header))
    for n, ts, tp in rows:
        winner = "PARALLEL" if tp < ts else "SERIAL"
        print(f"{n:>12,} | {ts:>12.4f} | {tp:>13.4f} | {ts / tp:>7.2f}x | {winner:>8}")


def main():
    print("Student ID    : 230103240")
    print(f"OS            : {platform.system()} {platform.release()}")
    print(f"Logical cores : {WORKERS}")
    print(f"Operation     : y[i] = {A} * x[i] + {B}")
    print(f"Pool          : ThreadPoolExecutor({WORKERS}) | best of several repeats\n")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = [(n, *measure(n, pool)) for n in SIZES]
        print("Coarse scan")
        print_table(rows)

        k = find_crossover(rows)
        print()
        if k is None:
            print("Parallel never beat serial in the tested range.")
            return
        if k == 0:
            print(f"Parallel was faster for every tested N (from N={rows[0][0]:,}).")
            return

        lo, hi = rows[k - 1][0], rows[k][0]
        print(f"Coarse crossover: SERIAL wins at N={lo:,}, PARALLEL wins at N={hi:,}\n")

        grid = sorted(set(int(v) for v in np.geomspace(lo, hi, 25)))
        fine = [(n, *measure(n, pool)) for n in grid]
        print("Fine scan between the bracketing sizes")
        print_table(fine)

        kf = find_crossover(fine)
        print()
        if kf is None or kf == 0:
            print(f"Crossover lies between N={lo:,} and N={hi:,}")
        else:
            print(
                f"EXACT RANGE: multi-threading becomes faster than serial between "
                f"N={fine[kf - 1][0]:,} (slower) and N={fine[kf][0]:,} (faster)"
            )


if __name__ == "__main__":
    main()