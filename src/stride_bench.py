import time
import numpy as np

def main():
    # Выделяем плоский массив (~134 МБ), превышающий объем L3 кэша[cite: 2, 3]
    data = np.ones(16_777_216, dtype=np.float64)
    strides = [1, 2, 4, 8, 16, 32, 64, 128]
    results = []

    print("Running Stride Latency Cliff Benchmark...")
    for s in strides:
        # 5 прогонов разогрева
        for _ in range(5):
            val = 0.0
            for i in range(0, len(data), s):
                val += data[i]

        # 5 записанных прогонов для получения точной медианы
        times = []
        for _ in range(5):
            start = time.perf_counter_ns()
            val = 0.0
            elements_touched = 0
            for i in range(0, len(data), s):
                val += data[i]
                elements_touched += 1
            end = time.perf_counter_ns()
            elapsed_ns = end - start
            times.append(elapsed_ns / elements_touched)

        median_latency = np.median(times)
        results.append(f"{s}\t{median_latency:.2f}")
        print(f"Stride {s:3d}: {median_latency:.2f} ns/element")

    # Сохраняем результаты в TSV-файл
    with open("results/stride_results.tsv", "w") as f:
        f.write("Stride\tLatency_ns_per_element\n")
        f.write("\n".join(results) + "\n")

if __name__ == "__main__":
    main()