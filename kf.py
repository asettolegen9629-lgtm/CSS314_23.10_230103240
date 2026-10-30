import time

def karp_flatt(t1, tp, p):
    return ((p * tp / t1) - 1.0) / (p - 1.0)

def speedup(t1, tp):
    return t1 / tp

def best_of(fn, *args, repeats=3):
    """Минимальное время из нескольких прогонов (снижает шум ноутбука)."""
    return min(fn(*args) for _ in range(repeats))