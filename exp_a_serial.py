import time, math
from multiprocessing import Pool
from kf import karp_flatt, speedup

def heavy_math(n):
    return sum(math.sin(i) * math.cos(i) for i in range(1, n))

def run_experiment(workers, total_chunks=16, chunk_size=2_000_000):
    t0 = time.perf_counter()
    time.sleep(0.40)                      # намеренное последовательное «горлышко»
    with Pool(processes=workers) as pool:
        pool.map(heavy_math, [chunk_size] * total_chunks)
    return time.perf_counter() - t0 

if __name__ == "__main__":
    rt = {p: run_experiment(p) for p in (1, 2, 4)}
    t1 = rt[1]
    for p in (2, 4):
        print(f"P={p} | T={rt[p]:.3f}s | Speedup={speedup(t1, rt[p]):.2f}x | "
              f"e={karp_flatt(t1, rt[p], p):.3f}")