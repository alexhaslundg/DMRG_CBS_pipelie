import contextlib
import time
import threading
import psutil
import os
import numpy as np




def convert_numpy(obj):
    """Convert numpy types and other non-serializable objects to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, '__class__') and obj.__class__.__module__.startswith('pyscf'):
        # Skip PySCF objects (like RHF, UHF, etc.) - not JSON serializable
        return f"<{obj.__class__.__name__} object - not serialized>"
    elif callable(obj):
        # Skip callable objects
        return None
    else:
        return obj

process = psutil.Process(os.getpid())

def get_mem_gb():
    """Return current memory usage in GB."""
    return process.memory_info().rss / 1024**3

def clean_scratch_dir(scratch_dir):
    """Remove scratch directory contents to free disk space."""
    import shutil
    if os.path.exists(scratch_dir):
        try:
            shutil.rmtree(scratch_dir)
            print(f"✅ Cleaned scratch directory: {scratch_dir}")
        except Exception as e:
            print(f"⚠️ Could not clean scratch: {e}")

def cleanup_memory():
    """Force garbage collection"""
    import gc
    gc.collect()

# -------------------------
# Adding in the memory monitor 
# -------------------------
# resource_monitor.py
"""
Resource monitoring utilities.

- check_memory(cap_gb): quick pre-check to decide if it's safe to start a job.
- resource_guard(cap_gb): context manager that raises MemoryError if memory exceeded at entry,
  and prints summary on exit.
- peak_memory_monitor(): context manager that samples process RSS periodically and reports peak.
"""


_SAMPLE_INTERVAL = 0.12  # seconds between memory samples


def _bytes_to_gb(x: int) -> float:
    return float(x) / 1024 ** 3


def check_memory(cap_gb: float) -> bool:
    """
    Quick check whether current used memory is below cap_gb.
    used = total - available
    """
    vm = psutil.virtual_memory()
    used_gb = _bytes_to_gb(vm.total - vm.available)
    return used_gb < cap_gb


@contextlib.contextmanager
def resource_guard(cap_gb: float):
    """
    Guard that checks memory before entering. Raises MemoryError if already above cap.
    On exit prints wall-clock time and approximate memory used.
    """
    if not check_memory(cap_gb):
        raise MemoryError(f"Memory usage already above cap of {cap_gb:.2f} GB.")

    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        vm = psutil.virtual_memory()
        used_gb = _bytes_to_gb(vm.total - vm.available)
        print(f"[resource_guard] Elapsed {end - start:.2f}s | system memory used ≈ {used_gb:.2f} GB")


@contextlib.contextmanager
def peak_memory_monitor(sample_interval: float = _SAMPLE_INTERVAL):
    """
    Monitor peak RSS of the current process while inside the context.
    Returns a dict-like object with 'peak_gb' on exit (use return value).
    Usage:
        with peak_memory_monitor() as monitor:
            do_work()
        peak = monitor['peak_gb']
    """
    proc = psutil.Process()
    peak_rss = 0
    running = True

    def _sampler():
        nonlocal peak_rss, running
        while running:
            try:
                rss = proc.memory_info().rss
                if rss > peak_rss:
                    peak_rss = rss
            except Exception:
                # process may have terminated
                pass
            time.sleep(sample_interval)

    thread = threading.Thread(target=_sampler, daemon=True)
    thread.start()
    try:
        monitor = {"peak_gb": None}
        yield monitor
    finally:
        running = False
        thread.join(timeout=1.0)
        monitor["peak_gb"] = _bytes_to_gb(peak_rss)
