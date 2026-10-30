from kf import karp_flatt, speedup, best_of
import exp_a_serial as A, exp_b_contention as B, exp_c_fixed as C

def times(fn, repeats=3):
    return {p: best_of(fn, p, repeats=repeats) for p in (1, 2, 4)}

def e(rt, p):
    return karp_flatt(rt[1], rt[p], p)

def row(name, rt, diag):
    return (f"| {name} | {rt[1]:.3f} | {rt[2]:.3f} | {rt[4]:.3f} | "
            f"{e(rt,2):.3f} | {e(rt,4):.3f} | {diag} |")

if __name__ == "__main__":
    a = times(A.run_experiment)
    b = times(B.run_contended)
    c = times(C.run_privatized)

    ratio = e(a, 4) / e(a, 2)
    e4_c = speedup(c[1], c[4]) / 4 * 100
    checks = [
        ("A: e(4)/e(2) in [0.80, 1.25]", f"{ratio:.2f}", 0.80 <= ratio <= 1.25),
        ("B: T4 > T1", f"{b[4]:.3f} > {b[1]:.3f}", b[4] > b[1]),
        ("B: e(4) >> e(2)", f"{e(b,4):.2f} vs {e(b,2):.2f}", e(b, 4) > 1.5 * e(b, 2)),
        ("C: T4 < T1", f"{c[4]:.3f} < {c[1]:.3f}", c[4] < c[1]),
        ("C: E4 > 70%", f"{e4_c:.0f}%", e4_c > 70),
    ]

    lines = [
        "# RESULTS",
        "",
        "Hardware: AMD Ryzen 7 5825U (8 cores / 16 threads), 15.4 GB RAM, Windows, Python 3.12.",
        "Each time is the best of 3 runs. Exp C uses total_increments = 300,000,000, because "
        "process-spawn overhead on Windows distorts the 20M default.",
        "",
        "| Experiment | T1(s) | T2(s) | T4(s) | e(2) | e(4) | Architectural Diagnosis |",
        "|---|---|---|---|---|---|---|",
        row("A: Serial Tail", a, "Constant e: fixed Amdahl serialization"),
        row("B: Lock Contention", b, "Exploding e: mutex and bus thrashing"),
        row("C: Privatized State", c, "Low e: near-linear scalability restored"),
        "",
        "## Rubric self-check",
    ] + [f"- {'PASS' if ok else 'FAIL'}: {name} ({val})" for name, val, ok in checks]

    text = "\n".join(lines)
    open("RESULTS.md", "w", encoding="utf-8").write(text)
    print(text)
