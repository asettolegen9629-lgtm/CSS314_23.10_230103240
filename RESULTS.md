# RESULTS

Hardware: AMD Ryzen 7 5825U (8 cores / 16 threads), 15.4 GB RAM, Windows, Python 3.12.
Each time is the best of 3 runs. Exp C uses total_increments = 300,000,000, because process-spawn overhead on Windows distorts the 20M default.

| Experiment | T1(s) | T2(s) | T4(s) | e(2) | e(4) | Architectural Diagnosis |
|---|---|---|---|---|---|---|
| A: Serial Tail | 4.526 | 2.649 | 1.754 | 0.171 | 0.183 | Constant e: fixed Amdahl serialization |
| B: Lock Contention | 5.445 | 13.880 | 15.556 | 4.098 | 3.476 | Exploding e: mutex and bus thrashing |
| C: Privatized State | 9.110 | 4.938 | 2.874 | 0.084 | 0.087 | Low e: near-linear scalability restored |

## Rubric self-check
- PASS: A: e(4)/e(2) in [0.80, 1.25] (1.07)
- PASS: B: T4 > T1 (15.556 > 5.445)
- FAIL: B: e(4) >> e(2) (3.48 vs 4.10)
- PASS: C: T4 < T1 (2.874 < 9.110)
- PASS: C: E4 > 70% (79%)