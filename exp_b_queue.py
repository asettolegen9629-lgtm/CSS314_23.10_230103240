import time
from multiprocessing import Process, Queue
from kf import print_table

def queue_worker(out_queue, items_per_worker):
    for i in range(items_per_worker):
        out_queue.put(i % 7)          # каждая итерация пересекает IPC-границу

def run_queue_benchmark(p, total_items=400_000):
    q = Queue()
    items_per_worker = total_items // p
    workers = [Process(target=queue_worker, args=(q, items_per_worker)) for _ in range(p)]
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    collected = 0
    expected = items_per_worker * p
    while collected < expected:
        q.get()
        collected += 1
    for w in workers:
        w.join()
    return time.perf_counter() - t0

if __name__ == "__main__":
    runtimes = {p: run_queue_benchmark(p) for p in (1, 2, 4, 8)}
    print_table("Queue Contention Karp-Flatt Analysis", runtimes)
    print("T:", {p: round(t, 3) for p, t in runtimes.items()})