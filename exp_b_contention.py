import time
from multiprocessing import Process, Value, Lock
from kf import karp_flatt, speedup

def contended_worker(shared_counter, lock, iterations):
    for _ in range(iterations):
        with lock:
            shared_counter.value += 1

def run_contended(p, total_increments=2_000_000):
    increments_per_worker = total_increments // p
    counter = Value('i', 0)
    lock = Lock()
    procs = [Process(target=contended_worker, args=(counter, lock, increments_per_worker))
             for _ in range(p)]
    t0 = time.perf_counter()
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    t1 = time.perf_counter()
    assert counter.value == increments_per_worker * p
    return t1 - t0

if __name__ == "__main__":
    rt = {p: run_contended(p) for p in (1, 2, 4)}
    t1 = rt[1]
    for p in (2, 4):
        print(f"P={p} | Runtime={rt[p]:.3f}s | Speedup={speedup(t1, rt[p]):.2f}x | "
              f"KarpFlatt e={karp_flatt(t1, rt[p], p):.3f}")