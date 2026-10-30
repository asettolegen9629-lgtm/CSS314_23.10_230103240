# RESULTS_B

| Configuration | T1(s) | T2(s) | T4(s) | T8(s) | S8 | e(2) | e(8) | Verdict |
|---|---|---|---|---|---|---|---|---|
| A: Micro (k=1) | 38.699 | 46.403 | 52.462 | 68.480 | 0.57 | 1.398 | 1.880 | Master dispatch serialization |
| A: Coarse (k=10000) | 9.982 | 5.193 | 2.832 | 1.886 | 5.29 | 0.040 | 0.073 | Flat e: healthy strong scaling |
| B: Queue Choke | 3.376 | 6.385 | 6.208 | 6.396 | 0.53 | 2.783 | 2.022 | IPC lock contention explosion |
| C: Buffered Batch | 0.867 | 0.478 | 0.336 | 0.308 | 2.81 | 0.104 | 0.264 | Amortized synchronization |

Hardware: AMD Ryzen 7 5825U (8C/16T), 15.4 GB RAM, Windows, Python 3.12.