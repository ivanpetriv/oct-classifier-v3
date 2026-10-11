import os
import time
import tracemalloc
from contextlib import contextmanager

import psutil


@contextmanager
def track_step_resources(step_name: str):
    """
    Context manager to track wall-clock time and memory delta for a specific execution block.
    """
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    start_mem = process.memory_info().rss / (1024 * 1024)

    yield

    end_time = time.perf_counter()
    end_mem = process.memory_info().rss / (1024 * 1024)

    elapsed_time = end_time - start_time
    mem_delta = end_mem - start_mem

    # print(f"[{step_name}] Time: {elapsed_time:.8f}s | Mem Delta: {mem_delta:+.5f} MB")


@contextmanager
def track_trial_resources():
    """Tracks time and peak memory, yielding a dict to store the results."""
    tracemalloc.start()
    start_time = time.perf_counter()

    metrics = {}
    try:
        yield metrics
    finally:
        metrics["time"] = time.perf_counter() - start_time
        _, peak_mem = tracemalloc.get_traced_memory()
        metrics["peak_mem_mb"] = peak_mem / (1024 * 1024)
        tracemalloc.stop()
