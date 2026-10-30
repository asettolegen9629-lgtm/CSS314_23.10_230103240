from kf import karp_flatt, speedup
import exp_a_granularity as A, exp_b_queue as B, exp_c_buffered as C

def row(name, rt, verdict):
    t1 = rt[1]
    return (f"| {name} | {rt[1]:.3f} | {rt[2]:.3f} | {rt[4]:.3f} | {rt[8]:.3f} | "
            f"{speedup(t1, rt[8]):.2f} | {karp_flatt(t1, rt[2], 2):.3f} | "
            f"{karp_flatt(t1, rt[8], 8):.3f} | {verdict} |")

if __name__ == "__main__":
    a_fine = A.measure(1)
    a_coarse = A.measure(10_000)
    b = {p: B.run_queue_benchmark(p) for p in (1, 2, 4, 8)}
    c = {p: C.run_buffered(p)[0] for p in (1, 2, 4, 8)}

    lines = [
        "# RESULTS_B",
        "",
        "| Configuration | T1(s) | T2(s) | T4(s) | T8(s) | S8 | e(2) | e(8) | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
        row("A: Micro (k=1)", a_fine, "Master dispatch serialization"),
        row("A: Coarse (k=10000)", a_coarse, "Flat e: healthy strong scaling"),
        row("B: Queue Choke", b, "IPC lock contention explosion"),
        row("C: Buffered Batch", c, "Amortized synchronization"),
        "",
        "Hardware: AMD Ryzen 7 5825U (8C/16T), 15.4 GB RAM, Windows, Python 3.12.",
    ]
    open("RESULTS_B.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))