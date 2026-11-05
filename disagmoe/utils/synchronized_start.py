"""One-shot CPU start gate for prepared batches on a single host.

The shared target uses ``perf_counter_ns`` like the benchmark clock. It does
not synchronize GPU clocks or promise simultaneous CUDA kernel launches.
This module deliberately has no Ray, CUDA, or PyTorch imports.
"""

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid
from collections.abc import Mapping
from numbers import Integral, Real


_MARKER = "marker.json"
_RELEASE = "release.json"
_KIND = "streamep-synchronized-start-v1"
_POLL_SECONDS = 0.001


def _positive_seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid synchronized-start data in {path}")
    return value


def create_gate():
    """Create a private directory which every local worker must be able to read."""
    directory = tempfile.mkdtemp(prefix="streamep-start-")
    run_id = uuid.uuid4().hex
    try:
        with open(Path(directory) / _MARKER, "x", encoding="utf-8") as handle:
            json.dump({"kind": _KIND, "run_id": run_id}, handle)
    except BaseException:
        shutil.rmtree(directory)
        raise
    return {"directory": directory, "run_id": run_id}


def validate_gate(directory, run_id):
    """Verify the run identity and that this process sees the shared directory."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Synchronized start requires a nonempty run_id")
    marker = _read_json(Path(directory) / _MARKER)
    if marker.get("kind") != _KIND or marker.get("run_id") != run_id:
        raise ValueError("Synchronized-start gate run_id does not match")


def publish_gate(directory, run_id, delay_s=1.0):
    """Atomically publish one shared future target after every worker is ready."""
    delay_s = _positive_seconds(delay_s, "delay_s")
    validate_gate(directory, run_id)
    directory = Path(directory)
    # Retain this exclusive claim after publication: the gate cannot be rearmed.
    claim = directory / "publish.lock"
    claim_fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(claim_fd)
    temporary = None
    try:
        if (directory / _RELEASE).exists():
            raise FileExistsError("Synchronized-start target was already published")
        release = {
            "run_id": run_id,
            "release_ns": time.perf_counter_ns() + int(delay_s * 1_000_000_000),
        }
        fd, temporary = tempfile.mkstemp(prefix="release-", suffix=".tmp", dir=directory)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(release, handle)
        os.replace(temporary, directory / _RELEASE)
        temporary = None
        return release
    finally:
        if temporary is not None:
            os.unlink(temporary)


def wait_for_release(directory, run_id, timeout_s=180.0, cancelled=lambda: False):
    """Wait for publication and its deadline, checking cancellation throughout.

    A worker that first observes an expired target fails instead of silently
    starting late. OS scheduling can still delay a worker after it has armed;
    callers must record their actual execution timestamp to measure that skew.
    """
    timeout_s = _positive_seconds(timeout_s, "timeout_s")
    validate_gate(directory, run_id)
    timeout_ns = time.perf_counter_ns() + int(timeout_s * 1_000_000_000)
    release = None
    while True:
        if cancelled():
            raise RuntimeError("Synchronized start cancelled")
        now = time.perf_counter_ns()
        if now >= timeout_ns:
            raise TimeoutError("Timed out waiting for synchronized-start release")
        if release is None:
            try:
                candidate = _read_json(Path(directory) / _RELEASE)
            except FileNotFoundError:
                candidate = None
            if candidate is not None:
                target = candidate.get("release_ns")
                if candidate.get("run_id") != run_id:
                    raise ValueError("Synchronized-start release run_id does not match")
                if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
                    raise ValueError("Invalid synchronized-start release_ns")
                if target <= time.perf_counter_ns():
                    raise RuntimeError("Synchronized-start target was already past when observed")
                release = candidate
        if release is not None:
            remaining_ns = release["release_ns"] - time.perf_counter_ns()
            if remaining_ns <= 0:
                return release
            sleep_seconds = min(_POLL_SECONDS, remaining_ns / 1_000_000_000)
        else:
            sleep_seconds = _POLL_SECONDS
        time.sleep(min(sleep_seconds, max(0, (timeout_ns - now) / 1_000_000_000)))


def cleanup_gate(directory):
    """Remove only a recognized gate directory; repeated cleanup is harmless."""
    directory = Path(directory)
    if not directory.exists():
        return
    if directory.is_symlink() or not directory.name.startswith("streamep-start-"):
        raise ValueError("Refusing to remove a path that is not a synchronized-start gate")
    marker = _read_json(directory / _MARKER)
    if marker.get("kind") != _KIND or not marker.get("run_id"):
        raise ValueError("Refusing to remove an unrecognized synchronized-start directory")
    shutil.rmtree(directory)


def validate_configuration(args):
    """Validate the optional fixed-cohort experiment without importing the runtime."""
    if not getattr(args, "synchronized_start", False):
        return
    if getattr(args, "num_nodes", None) != 1 or getattr(args, "placement", None) != "colocate":
        raise ValueError("--synchronized-start requires single-node colocate placement")
    if getattr(args, "tp_size", None) != 1:
        raise ValueError("--synchronized-start requires --tp-size 1")
    n_gpu = _positive_integer(getattr(args, "num_gpus", None), "num_gpus")
    if getattr(args, "dp_size", None) != n_gpu or getattr(args, "ep_size", None) != n_gpu:
        raise ValueError("--synchronized-start requires DP = EP = number of GPUs")
    if getattr(args, "generator_type", None) != "offline":
        raise ValueError("--synchronized-start requires --generator-type offline")
    n_requests = _positive_integer(getattr(args, "num_requests", None), "num_requests")
    if n_requests % n_gpu:
        raise ValueError("--synchronized-start requires num_requests divisible by DP")
    cap = _positive_integer(getattr(args, "max_batch_size_attn", None), "max_batch_size_attn")
    if n_requests // n_gpu > cap:
        raise ValueError("--synchronized-start initial batch per GPU exceeds max_batch_size_attn")
    for kind in ("input", "output"):
        minimum = _positive_integer(getattr(args, f"min_{kind}_len", None), f"min_{kind}_len")
        if getattr(args, f"max_{kind}_len", None) != minimum + 1:
            raise ValueError(f"--synchronized-start requires fixed {kind} length: max = min + 1")
    if (getattr(args, "mode", None) == "server" or getattr(args, "server", False)
            or getattr(args, "is_api_server", False)):
        raise ValueError("--synchronized-start does not support server mode")
    if getattr(args, "analyze_throughput", False):
        raise ValueError("--synchronized-start does not support --analyze-throughput")


def validate_capacity(kv_stats, blocks_per_request, batch_per_rank):
    """Check every rank before the caller mutates any KV reservation."""
    blocks_per_request = _positive_integer(blocks_per_request, "blocks_per_request")
    batch_per_rank = _positive_integer(batch_per_rank, "batch_per_rank")
    stats = dict(kv_stats) if isinstance(kv_stats, Mapping) else dict(enumerate(kv_stats))
    if not stats or set(stats) != set(range(len(stats))):
        raise ValueError("KV capacity must include contiguous DP ranks starting at zero")
    required = blocks_per_request * batch_per_rank
    for rank, available in sorted(stats.items()):
        if isinstance(available, bool) or not isinstance(available, Integral) or available < 0:
            raise ValueError(f"Invalid KV capacity for DP rank {rank}: {available}")
        if available < required:
            raise ValueError(
                f"Synchronized start needs {required} KV blocks on DP rank {rank} "
                f"({batch_per_rank} requests x {blocks_per_request}), but only {available} are free"
            )
