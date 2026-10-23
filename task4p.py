"""
CSS 314 - Task 4: Host-to-Device Transfer Penalty (CPU-only path: IPC overhead)
Student: Tolegen Aset (ID 230103240)

Moves a float64 array of N elements from this process to a persistent worker process.
  Path A: IPC pipe with pickle serialization (the array is serialized, copied through the
          pipe, and deserialized in the worker).
  Path B: zero-copy multiprocessing.shared_memory (only the block name/shape/dtype is sent;
          the worker maps the same physical memory).
The worker computes the array sum in both paths. Its own compute time is measured inside the
worker and subtracted, so  overhead = round-trip time - worker compute time.
Worker start-up is excluded (the worker is started and warmed up once).
"""
import multiprocessing as mp
import sys
import time
from multiprocessing import shared_memory

import numpy as np

STUDENT_ID = 230103240
SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 5


def _attach(name):
    try:
        return shared_memory.SharedMemory(name=name, track=False)  # Python 3.13+
    except TypeError:
        return shared_memory.SharedMemory(name=name)


def worker(conn):
    while True:
        msg = conn.recv()               # Path A: the unpickled array arrives here
        if msg is None:
            break
        if isinstance(msg, tuple):      # Path B: (shm_name, shape, dtype_str)
            name, shape, dtype = msg
            shm = _attach(name)
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            t0 = time.perf_counter()
            s = float(arr.sum())
            tc = time.perf_counter() - t0
            del arr
            shm.close()
        else:
            t0 = time.perf_counter()
            s = float(msg.sum())
            tc = time.perf_counter() - t0
        conn.send((s, tc))


def path_a(conn, arr):
    t0 = time.perf_counter()
    conn.send(arr)                      # pickle -> pipe -> unpickle in worker
    s, tc = conn.recv()
    total = time.perf_counter() - t0
    return (total - tc) * 1000.0, s


def path_b(conn, shm, shape, dtype):
    t0 = time.perf_counter()
    conn.send((shm.name, shape, dtype))  # only metadata crosses the process boundary
    s, tc = conn.recv()
    total = time.perf_counter() - t0
    return (total - tc) * 1000.0, s


def best(fn):
    runs = [fn() for _ in range(REPEATS)]
    return min(r[0] for r in runs), runs[0][1]


def main():
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=worker, args=(child,), daemon=True)
    proc.start()

    rng = np.random.default_rng(STUDENT_ID)
    rows = []
    for n in SIZES:
        shm = shared_memory.SharedMemory(create=True, size=n * 8)
        arr = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
        arr[:] = rng.random(n)           # same data for both paths (not timed)

        # warm-up (first message pays one-off costs)
        path_a(parent, arr[:1000].copy())
        path_b(parent, shm, (n,), "float64")

        a_ms, sum_a = best(lambda: path_a(parent, arr))
        b_ms, sum_b = best(lambda: path_b(parent, shm, (n,), "float64"))
        ok = abs(sum_a - sum_b) <= 1e-9 * abs(sum_a)
        rows.append((n, n * 8 / 1e6, a_ms, b_ms, a_ms / b_ms, ok))

        del arr
        shm.close()
        shm.unlink()

    parent.send(None)
    proc.join(timeout=5)

    lines = [
        "=" * 78,
        "Task 4: IPC Transfer Overhead - Pickle Pipe (A) vs Shared Memory (B)",
        "=" * 78,
        f"Student ID : {STUDENT_ID}   (data seed = {STUDENT_ID}, float64 arrays)",
        f"Timing     : best of {REPEATS} runs; worker compute time subtracted",
        "-" * 78,
        f"{'N':>12} | {'Size (MB)':>9} | {'Path A pipe (ms)':>16} | {'Path B shm (ms)':>15} | {'A / B':>8} | Sums",
        "-" * 78,
    ]
    for n, mb, a, b, r, ok in rows:
        lines.append(f"{n:>12,} | {mb:>9.1f} | {a:>16.3f} | {b:>15.3f} | {r:>7.1f}x | {'OK' if ok else 'MISMATCH'}")
    out = "\n".join(lines)
    print(out)
    with open("task4p_output.txt", "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()