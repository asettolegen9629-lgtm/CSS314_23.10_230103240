import os
# Keep BLAS/OpenMP threads at 1 so the baseline is strictly serial
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

import platform
import time
import numpy as np

STUDENT_ID = 230103240
N = 20_000_000


def cpu_model():
    # Windows: read the real CPU name from the registry
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or platform.machine()


def main():
    print(f"Student ID     : {STUDENT_ID}")
    print(f"OS             : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"CPU            : {cpu_model()}")
    print(f"Logical cores  : {os.cpu_count()}")
    print(f"Python         : {platform.python_version()}")
    print(f"Array size     : {N:,} float64")

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(N, dtype=np.float64)

    t0 = time.perf_counter()
    total = float(np.sum(data))
    mean = total / N
    var = float(np.sum((data - mean) ** 2)) / N   # population std dev
    std = var ** 0.5
    t1 = time.perf_counter()

    print(f"Sum            : {total:.6f}")
    print(f"Std deviation  : {std:.10f}")
    print(f"Serial time    : {(t1 - t0) * 1000:.2f} ms")


if __name__ == "__main__":
    main()