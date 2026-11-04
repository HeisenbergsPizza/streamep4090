"""Nsight Systems options for one measured window in each Ray worker."""

import math
from pathlib import Path
import shutil
import time


def make_nsight_config(cpu_sampling: bool = False) -> dict[str, str]:
    """Build Ray's ``runtime_env['nsight']`` without CUDA or Ray imports.

    Workers must call cudaProfilerStart/Stop around the measured workload.
    ``stop-shutdown`` ends the collection session at cudaProfilerStop; ``kill``
    is explicitly ``none`` so session shutdown does not terminate the worker.
    CPU sampling requires host/container perf permissions.
    NVTX and CUDA API ranges remain available when CPU sampling is disabled.

    Return a new dictionary because Ray 2.43 adds the output path in place.
    Leave that path to Ray so every worker gets its own PID-named report.
    """
    if not isinstance(cpu_sampling, bool):
        raise TypeError("cpu_sampling must be a bool")
    cpu_scope = "process-tree" if cpu_sampling else "none"
    return {
        "t": "cuda,nvtx,osrt",
        "sample": cpu_scope,
        "cpuctxsw": cpu_scope,
        "capture-range": "cudaProfilerApi",
        "capture-range-end": "stop-shutdown",
        "kill": "none",
        # Ray launches nsys as the worker wrapper; retain that wrapper until
        # its primary actor process exits, even after collection completes.
        "wait": "primary",
        "stop-on-exit": "true",
        # Capture individual kernels inside CUDA graphs for operator breakdowns.
        # This costs more than graph-level tracing; use short measured windows.
        "cuda-graph-trace": "node",
    }


def collect_nsight_reports(ray_root, output_dir, expected_workers=4, timeout_s=60):
    """Wait for worker reports and copy them into a new output directory.

    Nonempty files must have unchanged size and modification time across two
    polling intervals. Resolve paths to avoid counting Ray's
    ``session_latest`` symlink twice. Waiting is bounded by ``timeout_s`` (at
    most 60 seconds), excluding the subsequent file copy.

    File stability is only a coarse export-completion check. Run ``nsys stats``
    on the copied reports to verify they are readable and contain the expected
    CUDA activity. Call this before Ray shutdown so report export can finish.
    The output directory must be new; original Ray files are never removed.
    """
    if (isinstance(expected_workers, bool)
            or not isinstance(expected_workers, int) or expected_workers < 1):
        raise ValueError("expected_workers must be a positive integer")
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or not 0 <= timeout_s <= 60):
        raise ValueError("timeout_s must be a finite number between 0 and 60")
    root = Path(ray_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Nsight output directory already exists; choose a new run directory: {output}")

    deadline = time.monotonic() + timeout_s
    previous = {}
    stable_intervals = {}
    while True:
        current = {}
        for candidate in root.glob("session_*/logs/nsight/*.nsys-rep"):
            try:
                source = candidate.resolve(strict=True)
                stat = source.stat()
            except FileNotFoundError:
                # The profiler or Ray may still be changing temporary files.
                continue
            if source.is_file() and stat.st_size > 0:
                current[source] = (stat.st_size, stat.st_mtime_ns)

        stable_intervals = {
            source: stable_intervals.get(source, 0) + 1
            if previous.get(source) == signature else 0
            for source, signature in current.items()
        }
        stable = sorted(source for source in current if stable_intervals[source] >= 2)
        if len(stable) >= expected_workers:
            names = [source.name for source in stable]
            if len(names) != len(set(names)):
                raise RuntimeError(f"Duplicate Nsight report names under {root}; refusing to overwrite reports")
            output.mkdir(parents=True, exist_ok=False)
            copied = []
            for source in stable:
                destination = output / source.name
                shutil.copy2(source, destination)
                after = source.stat()
                if ((after.st_size, after.st_mtime_ns) != current[source]
                        or destination.stat().st_size != current[source][0]):
                    raise RuntimeError(
                        f"Nsight report changed during copy: {source}; "
                        f"output directory may be incomplete: {output}. "
                        f"Original reports remain under {root}"
                    )
                copied.append(str(destination))
            return copied

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout_s:g}s waiting for {expected_workers} stable "
                f"Nsight worker reports: found {len(current)} nonempty, {len(stable)} stable. "
                f"Original Ray report root: {root}; inspect session_*/logs/nsight/ "
                "and worker/runtime_env logs. No reports were copied or removed."
            )
        previous = current
        time.sleep(min(1.0, remaining))
