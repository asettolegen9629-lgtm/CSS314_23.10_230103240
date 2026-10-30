import time
from multiprocessing import Process, Queue
from kf import print_table

def buffered_worker(out_queue, items_per_worker, batch_size=10_000):
    buffer = []
    for i in range(items_per_worker):
        buffer.append(i % 7)
        if len(buffer) >= batch_size:
            out_queue.put(sum(buffer))
            buffer.clear()
    if buffer:
        out_queue.put(sum(buffer))
    out_queue.put(None)               # sentinel

def run_buffered(p, total_items=10_000_000):
    q = Queue()
    items_per_worker = total_items // p
    workers = [Process(target=buffered_worker, args=(q, items_per_worker)) for _ in range(p)]
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    total_sum, active = 0, p
    while active > 0:
        val = q.get()
        if val is None:
            active -= 1
        else:
            total_sum += val
    for w in workers:
        w.join()
    return time.perf_counter() - t0, total_sum

if __name__ == "__main__":
    runtimes, sums = {}, {}
    for p in (1, 2, 4, 8):
        runtimes[p], sums[p] = run_buffered(p)
    print_table("Buffered Batch Karp-Flatt Analysis", runtimes)
    print("T:", {p: round(t, 3) for p, t in runtimes.items()})
    print("Суммы (для P=1,2,4,8 совпадут только если total_items делится на P):", sums)
    # Throughput: сравнение с Exp B
    print(f"Throughput C (P=8): {10_000_000 / runtimes[8]:,.0f} items/s")