import time
from multiprocessing import Pool
from kf import print_table

def collatz_steps(n):
    steps = 0
    while n > 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps

def run_granularity(workers, total_items=1_000_000, chunk_size=1):
    numbers = range(1, total_items + 1)
    t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        results = pool.map(collatz_steps, numbers, chunksize=chunk_size)
    return time.perf_counter() - t0, sum(results)

def serial_baseline(total_items=1_000_000):
    t0 = time.perf_counter()
    s = sum(collatz_steps(n) for n in range(1, total_items + 1))
    return time.perf_counter() - t0, s

def measure(chunk_size, cores=(1, 2, 4, 8), total_items=1_000_000):
    runtimes, checksum = {}, None
    for p in cores:
        t, s = run_granularity(p, total_items, chunk_size)
        runtimes[p] = t
        checksum = s if checksum is None else checksum
        assert s == checksum, "результаты разных прогонов не совпали!"
    return runtimes

if __name__ == "__main__":
    ts, _ = serial_baseline()
    print(f"Serial (без Pool): {ts:.3f}s")
    for label, k in (("FINE chunksize=1", 1), ("COARSE chunksize=10000", 10_000)):
        rt = measure(k)
        print_table(label, rt)
        print("T:", {p: round(t, 3) for p, t in rt.items()})