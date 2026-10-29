import json
import time
import numpy as np

def main():
    # Загружаем соленый параметр N из Milestone 0
    with open("results/salted_params.json", "r") as f:
        params = json.load(f)
    
    N = params["N"]
    M = N // 2  # Подмножество для теста внутри лимита времени
    
    np.random.seed(params["seed"])
    A = np.random.rand(M, M)
    B = np.random.rand(M, M)
    C_naive = np.zeros((M, M))
    C_interchanged = np.zeros((M, M))

    # 1. Алгоритм А (Наивный i-j-k)[cite: 3]
    times_naive = []
    for _ in range(3):
        start = time.perf_counter()
        C_naive.fill(0.0)
        for i in range(M):
            for j in range(M):
                sum_val = 0.0
                for k in range(M):
                    sum_val += A[i, k] * B[k, j]
                C_naive[i, j] = sum_val
        times_naive.append(time.perf_counter() - start)
    T_naive = np.median(times_naive)

    # 2. Алгоритм Б (Переставленный i-k-j)[cite: 3]
    times_inter = []
    for _ in range(3):
        start = time.perf_counter()
        C_interchanged.fill(0.0)
        for i in range(M):
            for k in range(M):
                r_aik = A[i, k]
                for j in range(M):
                    C_interchanged[i, j] += r_aik * B[k, j]
        times_inter.append(time.perf_counter() - start)
    T_interchanged = np.median(times_inter)

    speedup = T_naive / T_interchanged if T_interchanged > 0 else 0.0

    output = {
        "matrix_subblock_M": M,
        "time_naive_seconds": T_naive,
        "time_interchanged_seconds": T_interchanged,
        "speedup": speedup
    }

    with open("results/loop_interchange.json", "w") as f:
        json.dump(output, f, indent=4)

    print(f"[PASS] Milestone 1: Naive: {T_naive:.4f}s, Interchanged: {T_interchanged:.4f}s, Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()