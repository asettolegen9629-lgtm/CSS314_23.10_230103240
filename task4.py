import queue
import threading
import time

N = 1000
WORKERS = 4
UNIT = 0.0001
REPEATS = 3


def cost(i):
    time.sleep(UNIT * (i % 20))


def static_target(w, barrier, finish):
    lo = w * N // WORKERS
    hi = (w + 1) * N // WORKERS
    barrier.wait()
    for i in range(lo, hi):
        cost(i)
    finish[w] = time.perf_counter()


def make_dynamic(chunk):
    q = queue.Queue()
    for s in range(0, N, chunk):
        q.put((s, min(s + chunk, N)))

    def target(w, barrier, finish):
        barrier.wait()
        while True:
            try:
                lo, hi = q.get_nowait()
            except queue.Empty:
                break
            for i in range(lo, hi):
                cost(i)
        finish[w] = time.perf_counter()

    return target


def execute(target):
    barrier = threading.Barrier(WORKERS + 1)
    finish = [0.0] * WORKERS
    threads = [
        threading.Thread(target=target, args=(w, barrier, finish))
        for w in range(WORKERS)
    ]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    total = (time.perf_counter() - t0) * 1000.0
    rel = [(f - t0) * 1000.0 for f in finish]
    return total, min(rel), max(rel)


def best_of(make_target):
    best = None
    for _ in range(REPEATS):
        res = execute(make_target())
        if best is None or res[0] < best[0]:
            best = res
    return best


def main():
    ideal = sum(i % 20 for i in range(N)) * UNIT * 1000.0 / WORKERS
    print("Student ID : 230103240")
    print(f"Items      : {N} | Workers: {WORKERS} | cost(i) = sleep(0.0001 * (i % 20))")
    print(f"Ideal time : {ideal:.2f} ms (total work / {WORKERS})\n")

    strategies = [
        ("A: Static blocks", lambda: static_target),
        ("B: Dynamic chunk=1", lambda: make_dynamic(1)),
        ("B: Dynamic chunk=5", lambda: make_dynamic(5)),
    ]

    header = (
        f"{'Strategy':<20} | {'Total (ms)':>10} | {'First done':>10} | "
        f"{'Last done':>10} | {'Delta (ms)':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, make in strategies:
        total, first, last = best_of(make)
        print(
            f"{name:<20} | {total:>10.2f} | {first:>10.2f} | "
            f"{last:>10.2f} | {last - first:>10.2f}"
        )


if __name__ == "__main__":
    main()