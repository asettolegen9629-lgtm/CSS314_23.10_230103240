def karp_flatt(t1, tp, p):
    return ((p * tp / t1) - 1.0) / (p - 1.0)

def speedup(t1, tp):
    return t1 / tp

def print_table(title, runtimes):
    t1 = runtimes[1]
    print(f"\n--- {title} ---")
    for p in (2, 4, 8):
        tp = runtimes[p]
        print(f"P={p:2d} | Time={tp:7.3f}s | Speedup={speedup(t1, tp):5.2f}x | "
              f"Eff={speedup(t1, tp)/p*100:6.1f}% | Karp-Flatt e={karp_flatt(t1, tp, p):7.3f}")