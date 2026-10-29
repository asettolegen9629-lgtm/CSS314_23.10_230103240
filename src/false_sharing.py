import multiprocessing as mp
import ctypes
import time
import json

def worker_unpadded(shared_arr, worker_id, iterations):
    for _ in range(iterations):
        shared_arr[worker_id] += 1

def worker_padded(shared_arr, worker_id, iterations, stride):
    idx = worker_id * stride
    for _ in range(iterations):
        shared_arr[idx] += 1

def worker_privatized(shared_arr, worker_id, iterations, stride):
    local_val = 0
    for _ in range(iterations):
        local_val += 1
    shared_arr[worker_id * stride] = local_val

def measure_time(target_func, args, P):
    procs = []
    start = time.perf_counter()
    for p in range(P):
        p_args = args + (p,)
        proc = mp.Process(target=target_func, args=p_args)
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    end = time.perf_counter()
    return end - start

def main():
    iterations = 5_000_000
    p_values = [1, 2, 4, 8]
    results_rows = []

    # Базовые замеры для P=1 (необходимы для формулы Карпа-Флатта)
    arr_u1 = mp.Array(ctypes.c_int64, 8)
    T1_unpadded = measure_time(worker_unpadded, (arr_u1, iterations), 1)
    
    arr_p1 = mp.Array(ctypes.c_int64, 64)
    T1_padded = measure_time(worker_padded, (arr_p1, iterations, 8), 1)

    print("Running False Sharing & Karp-Flatt Analysis...")
    for P in p_values:
        # 1. Unpadded (False Sharing)[cite: 4]
        arr_u = mp.Array(ctypes.c_int64, 8)
        T_unpad = measure_time(worker_unpadded, (arr_u, iterations), P)
        e_unpad = 0.0 if P == 1 else ((P * T_unpad / T1_unpadded) - 1) / (P - 1)

        # 2. Padded (64-byte padding)[cite: 4]
        arr_p = mp.Array(ctypes.c_int64, 64)
        T_pad = measure_time(worker_padded, (arr_p, iterations, 8), P)
        e_pad = 0.0 if P == 1 else ((P * T_pad / T1_padded) - 1) / (P - 1)

        # 3. Privatized
        arr_priv = mp.Array(ctypes.c_int64, 64)
        T_priv = measure_time(worker_privatized, (arr_priv, iterations, 8), P)

        results_rows.append(f"{P}\t{T_unpad:.4f}\t{e_unpad:.4f}\t{T_pad:.4f}\t{e_pad:.4f}\t{T_priv:.4f}")
        print(f"P={P} | Unpadded T: {T_unpad:.4f}s (e={e_unpad:.2f}) | Padded T: {T_pad:.4f}s (e={e_pad:.2f})")

    # Сохраняем результаты в TSV
    with open("results/false_sharing_karp_flatt.tsv", "w") as f:
        f.write("P\tT_unpadded\te_unpadded\tT_padded\te_padded\tT_privatized\n")
        f.write("\n".join(results_rows) + "\n")

    print("[PASS] Milestone 2 completed successfully.")

if __name__ == "__main__":
    main()