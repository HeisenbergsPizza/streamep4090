"""Versioned profile bundle and atomic per-stage publication."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile

FORMAT = "streaminfer.single_node_profile"
VERSION = 1
STAGES = ("expert", "attention", "grouped", "reformation", "network", "host")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(bundle):
    path = Path(bundle) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise ValueError(f"Unsupported profile bundle: {path}")
    model = manifest["model"]
    for key in ("hidden_size", "intermediate_size", "num_experts", "top_k", "num_attention_heads",
                "num_key_value_heads", "head_dim", "num_layers"):
        if type(model.get(key)) is not int or model[key] < 1:
            raise ValueError(f"Invalid model.{key} in {path}")
    if model["top_k"] > model["num_experts"] or model["num_attention_heads"] % model["num_key_value_heads"]:
        raise ValueError("Invalid expert or attention head configuration")
    devices = manifest["hardware"]["gpus"]
    n = len(devices)
    if (n < 2 or [x["logical_index"] for x in devices] != list(range(n))
            or manifest["hardware"]["interconnect"] not in ("pcie", "nvlink")
            or manifest["parallel"] != {"dp": n, "ep": n, "tp": 1}
            or model["num_experts"] % n):
        raise ValueError("Bundle requires one node, DP=EP=visible GPU count, attention TP=1 and even expert placement")
    if manifest["runtime"]["dtype"] != "bfloat16" or type(manifest["runtime"]["cuda_graph"]) is not bool:
        raise ValueError("Unsupported dtype or CUDA Graph setting in manifest")
    return manifest


@contextmanager
def StageWriter(bundle, stage):
    """Stages are independently collectable/retriable; consumers see only completed data."""
    bundle = Path(bundle)
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage}")
    load_manifest(bundle)
    manifest_hash = file_hash(bundle / "manifest.json")
    target = bundle / stage
    if target.exists():
        raise FileExistsError(f"Stage already exists: {target}")
    if stage != "host":
        from .manifest import operating_state
        before = operating_state()
    with tempfile.TemporaryDirectory(prefix=f".{stage}-", dir=bundle) as temporary:
        output = Path(temporary) / stage
        output.mkdir()
        yield output
        if file_hash(bundle / "manifest.json") != manifest_hash:
            raise ValueError("Profile manifest changed during collection")
        # Only simulator input and provenance files need to be rehashed on every import.
        hashes = {p.name: file_hash(p) for p in sorted(output.iterdir())
                  if p.is_file() and p.name in (stage + ".csv", "parameters.json", "details.json")}
        write_json(output / "stage.json", {"stage": stage, "manifest_sha256": manifest_hash,
                                          "finished_utc": datetime.now(timezone.utc).isoformat(),
                                          "gpu_before_after": [before, operating_state()] if stage != "host" else None,
                                          "sha256": hashes})
        output.rename(target)


def measure_cuda(callback, warmup, repeats):
    """Measure a native forward or native graph replay. Warmup precedes CUDA events."""
    import torch
    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    current = torch.cuda.current_stream()
    worker = torch.cuda.Stream()
    worker.wait_stream(current)
    with torch.cuda.stream(worker):
        for _ in range(warmup):
            callback()
        worker.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(repeats):
            start.record()
            callback()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    current.wait_stream(worker)
    if not all(math.isfinite(x) and x > 0 for x in samples):
        raise ValueError("CUDA events returned a nonpositive or nonfinite measurement")
    return samples
