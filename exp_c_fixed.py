import time
from multiprocessing import Pool
from kf import karp_flatt, speedup

def private_worker(iterations):
    local_sum = 0
    for _ in range(iterations):
        local_sum += 1
    return local_sum

def run_privatized(p, total_increments=300_000_000):
    increments_per_worker = total_increments // p
    tasks = [increments_per_worker] * p
    t0 = time.perf_counter()
    with Pool(processes=p) as pool:
        results = pool.map(private_worker, tasks)
    total = sum(results)
    t1 = time.perf_counter()
    assert total == increments_per_worker * p
    return t1 - t0

if __name__ == "__main__":
    rt = {p: run_privatized(p) for p in (1, 2, 4)}
    t1 = rt[1]
    for p in (2, 4):
        print(f"P={p} | T={rt[p]:.3f}s | Speedup={speedup(t1, rt[p]):.2f}x | "
              f"E={speedup(t1, rt[p])/p*100:.0f}% | e={karp_flatt(t1, rt[p], p):.3f}")
