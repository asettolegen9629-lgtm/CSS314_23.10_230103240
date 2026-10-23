"""
CSS 314 - Task 2: Pseudo-Random Generator State Bottleneck
Student: Tolegen Aset (ID 230103240)

Monte Carlo pi with 40,000,000 points on 4 threads.
  Run 1: ONE shared random.Random instance, every draw guarded by a threading.Lock.
  Run 2: each thread owns its own random.Random(seed = STUDENT_ID * 10 + thread_index).
Both runs use the same loop and the same generator type; the only difference is the lock
around the shared generator state.
"""
import math
import random
import sys
import threading
import time

STUDENT_ID = 230103240
TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 40_000_000
THREADS = 4
PER_THREAD = TOTAL // THREADS


def worker_shared(rng, lock, n, out, idx):
    rnd = rng.random
    hits = 0
    for _ in range(n):
        with lock:                      # one critical section per sample point
            x = rnd()
            y = rnd()
        if x * x + y * y <= 1.0:
            hits += 1
    out[idx] = hits


def worker_independent(seed, n, out, idx):
    rnd = random.Random(seed).random    # private generator, no lock
    hits = 0
    for _ in range(n):
        x = rnd()
        y = rnd()
        if x * x + y * y <= 1.0:
            hits += 1
    out[idx] = hits


def run(targets_args):
    out = [0] * THREADS
    threads = [threading.Thread(target=t, args=a + (out, i))
               for i, (t, a) in enumerate(targets_args)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ms = (time.perf_counter() - t0) * 1000.0
    pi = 4.0 * sum(out) / (PER_THREAD * THREADS)
    return pi, abs(pi - math.pi), ms


def main():
    # Run 1: shared PRNG + mutex
    shared_rng = random.Random(STUDENT_ID * 10)
    lock = threading.Lock()
    pi_s, err_s, ms_s = run([(worker_shared, (shared_rng, lock, PER_THREAD))] * THREADS)

    # Run 2: independent PRNG per thread, seed = ID*10 + thread_index
    pi_i, err_i, ms_i = run([(worker_independent, (STUDENT_ID * 10 + i, PER_THREAD))
                             for i in range(THREADS)])

    speedup = ms_s / ms_i
    lines = [
        "=" * 58,
        "Task 2: Pseudo-Random Generator State Bottleneck",
        "=" * 58,
        f"Student ID   : {STUDENT_ID}",
        f"Samples      : {PER_THREAD * THREADS:,} ({THREADS} threads x {PER_THREAD:,})",
        f"Seeds (indep): {', '.join(str(STUDENT_ID * 10 + i) for i in range(THREADS))}",
        "-" * 58,
        f"Shared PRNG (mutex)  : pi = {pi_s:.8f} | abs error = {err_s:.8f} | time = {ms_s:12.2f} ms",
        f"Independent PRNGs    : pi = {pi_i:.8f} | abs error = {err_i:.8f} | time = {ms_i:12.2f} ms",
        "-" * 58,
        f"Speedup from removing generator lock (shared / independent): {speedup:.2f}x",
    ]
    out = "\n".join(lines)
    print(out)
    with open("task2p_output.txt", "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()