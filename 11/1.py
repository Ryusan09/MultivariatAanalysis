import time
import random
import multiprocessing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Serial reference implementation
# ---------------------------------------------------------------------------

def merge_sort_serial(data: list) -> list:
    """Pure-serial merge sort. Used as the worker function and as a baseline."""
    if len(data) <= 1:
        return data
    mid = len(data) // 2
    left = merge_sort_serial(data[:mid])
    right = merge_sort_serial(data[mid:])
    return _merge(left, right)


def _merge(left: list, right: list) -> list:
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# ---------------------------------------------------------------------------
# Parallel implementation
# ---------------------------------------------------------------------------

def merge_sort_parallel(data: list, num_workers: int) -> list:
    """
    Parallel merge sort using multiprocessing.Pool.

    Strategy:
      1. Split input into num_workers chunks via strided assignment.
      2. Sort all chunks in parallel with pool.map().
      3. Iteratively merge pairs of sorted chunks until one result remains.

    The split and merge steps are sequential; the sorting of chunks is parallel.
    This mirrors the pattern in later exercises: independent work units are
    dispatched to a pool, then results are combined.
    """
    if len(data) <= 1:
        return data

    # Step 1: Strided split — each worker gets every num_workers-th element
    chunks = [data[i::num_workers] for i in range(num_workers)]

    # Step 2: Parallel sort — each chunk is sorted independently
    with multiprocessing.Pool(processes=num_workers) as pool:
        sorted_chunks = pool.map(merge_sort_serial, chunks)

    # Step 3: Sequential merge — merge pairs iteratively until one list remains
    while len(sorted_chunks) > 1:
        next_level = []
        for i in range(0, len(sorted_chunks), 2):
            if i + 1 < len(sorted_chunks):
                next_level.append(_merge(sorted_chunks[i], sorted_chunks[i + 1]))
            else:
                next_level.append(sorted_chunks[i])
        sorted_chunks = next_level

    return sorted_chunks[0]

# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark(data: list, max_workers: int = 8) -> list:
    """
    Run merge_sort_parallel with worker counts 1..max_workers.

    Returns a list of (num_workers, elapsed_seconds) tuples.
    """
    results = []
    for w in range(1, max_workers + 1):
        start = time.perf_counter()
        merge_sort_parallel(data, num_workers=w)
        elapsed = time.perf_counter() - start
        results.append((w, elapsed))
        print(f"  workers={w:2d}  elapsed={elapsed:.4f}s")
    return results


def verify_correctness(data: list, num_workers: int) -> bool:
    """Verify that the parallel sort produces the same result as serial sort."""
    serial_result = merge_sort_serial(data)
    parallel_result = merge_sort_parallel(data, num_workers)
    return serial_result == parallel_result

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_timing(results: list, output_dir: str = ".") -> str:
    """Plot execution time vs. number of workers and save as PNG."""
    workers = [r[0] for r in results]
    times = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(workers, times, "bo-", markersize=8, linewidth=2)
    ax.set_xlabel("Number of Workers", fontsize=12)
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_title("Merge Sort: Parallel Speedup", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Annotate each point
    for w, t in zip(workers, times):
        ax.annotate(f"{t:.3f}s", (w, t), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)

    path = f"{output_dir}/merge_sort_timing.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Exercise 11-1: Merge Sort with multiprocessing")
    print("=" * 60)
    print()

    # Generate random data
    data_size = 10_000_000  # 1 million floats
    random.seed(42)
    data = [random.uniform(0.0, 100000.0) for _ in range(data_size)]
    print(f"Data size: {data_size} floats")
    print()

    # --- Correctness check ---
    print("[1] Correctness verification")
    for w in [1, 2, 4, 8]:
        ok = verify_correctness(data, w)
        status = "PASS" if ok else "FAIL"
        print(f"  workers={w}: {status}")
    print()

    # --- Benchmark ---
    print("[2] Timing benchmark (workers=1..8)")
    results = benchmark(data, max_workers=8)
    print()

    # --- Visualization ---
    print("[3] Generating timing plot ...")
    plot_path = plot_timing(results)
    print(f"  Saved: {plot_path}")
    print()

    # --- Bayes' theorem connection ---
    print("[4] Bayes' theorem connection")
    print("  This exercise demonstrates the parallelization pattern:")
    print("    pool.map(independent_function, list_of_inputs)")
    print()
    print("  In the following exercises, the same pattern is used to:")
    print("    Exercise 11-2: Parallelize particle prediction & weight updates")
    print("                  (each particle is independent → Bayes weight update)")
    print("    Exercise 12-1: Parallelize fitness evaluation in GA")
    print("                  (each individual's fitness is independent)")
    print("    Exercise 12-2: Parallelize likelihood evaluations in BO")
    print("                  (each candidate point's likelihood is independent)")
    print()
    print("  Bayes' theorem: p(θ|D) ∝ p(D|θ) · p(θ)")
    print("  The likelihood p(D|θ) is computed independently for each data point")
    print("  or each particle — exactly what pool.map() parallelizes.")
    print()
    print("Done.")

if __name__ == "__main__":
    main()