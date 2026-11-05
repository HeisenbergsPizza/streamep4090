"""Rebuild the bundled Qwen3 TP1 decode Attention proxy, without network access.

This adapter is deliberately pinned to one upstream model/device bundle. It
composes that bundle's Qwen attention block; it does not calibrate A100 costs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from .costs import _bracket


UPSTREAM_COMMIT = "b09c7aaab4b072d73136641841cc2688616c49eb"
UPSTREAM_REPOSITORY = "https://github.com/casys-kaist/LLMServingSim"
UPSTREAM_BUNDLE = "profiler/perf/RTXPRO6000/Qwen/Qwen3-30B-A3B-Instruct-2507/bf16"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DENSE_MULTIPLIERS = {"qkv_proj": 1, "qk_norm": 1, "rotary_emb": 1,
                     "o_proj": 1, "layernorm": 2}
SOURCE_PATHS = {
    "attention.csv": f"{UPSTREAM_BUNDLE}/tp1/attention.csv",
    "dense.csv": f"{UPSTREAM_BUNDLE}/tp1/dense.csv",
    "meta.yaml": f"{UPSTREAM_BUNDLE}/meta.yaml",
    "qwen3.yaml": "profiler/models/qwen3.yaml",
    "timings.py": "profiler/core/hooks/timings.py",
    "LICENSE": "LICENSE",
}
SOURCE_SHA256 = {
    "attention.csv": "d62ea19809d848ced8e5a12b623c0a94491d69409284a13a5ff91975a42ccb30",
    "dense.csv": "b876a54dd1e05c0454295cb92469b07a47700e0db4768846b0722d630dcc574b",
    "meta.yaml": "a230fe396685686a9f0ea94edceec7b4b0f6666167bb691d4dabf5354f933bf8",
    "qwen3.yaml": "218a1f0dee83f973b14435e0170283b1e89bebcbd188f4c31f86a80489b73723",
    "timings.py": "5a83b045c4d1d72549cc9643832a6127c7ae2e40ad738ef84e58b323e13225aa",
    "LICENSE": "d74d0ee1a35975f6bd570bdcffbc5bf99fc3e808850da80b8aad8ab7308696f0",
}


def _require(row, fields):
    if not set(fields).issubset(row):
        raise ValueError(f"Missing source CSV fields: {sorted(set(fields) - set(row))}")


def _integer(value, label, minimum=0):
    try:
        # CSV coordinates are integers, not rounded continuous values.
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _time_us(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("time_us must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("time_us must be finite and nonnegative")
    return result


def compose_decode_rows(attention_rows, dense_rows):
    """Compose disjoint upstream components into the simulator's CSV contract.

    Keep all measured batch breakpoints from both tables in their common
    domain. Inserting interpolated core values at dense-only breakpoints
    preserves both original piecewise-linear functions exactly.
    """
    core = {}
    for row in attention_rows:
        _require(row, ("layer", "prefill_chunk", "prefill_key", "n_decode",
                       "kv_decode", "decode_q_len", "time_us"))
        if (row["layer"] != "attention"
                or _integer(row["prefill_chunk"], "prefill_chunk") != 0
                or float(row["prefill_key"]) != 0
                or _integer(row["decode_q_len"], "decode_q_len", 1) != 1):
            continue
        batch = _integer(row["n_decode"], "n_decode")
        if batch == 0:
            continue
        key = (batch, _integer(row["kv_decode"], "kv_decode"))
        if key in core:
            raise ValueError(f"Duplicate pure-decode attention coordinate {key}")
        core[key] = _time_us(row["time_us"])
    if not core:
        raise ValueError("No pure-decode q=1 Attention measurements")
    core_batches = sorted({b for b, _ in core})
    contexts = sorted({c for _, c in core})
    missing = [(b, c) for b in core_batches for c in contexts if (b, c) not in core]
    if missing:
        raise ValueError(f"Attention profile missing grid corners {missing[:8]}")

    dense = {name: {} for name in DENSE_MULTIPLIERS}
    for row in dense_rows:
        _require(row, ("layer", "tokens", "time_us"))
        name = row["layer"]
        if name not in dense:
            continue
        batch = _integer(row["tokens"], "tokens", 1)
        if batch in dense[name]:
            raise ValueError(f"Duplicate dense component/batch {name}/{batch}")
        dense[name][batch] = _time_us(row["time_us"])
    dense_batches = sorted(dense["qkv_proj"])
    if not dense_batches or any(sorted(values) != dense_batches for values in dense.values()):
        raise ValueError("Missing dense components or inconsistent component batch axes")
    low = max(core_batches[0], dense_batches[0])
    high = min(core_batches[-1], dense_batches[-1])
    if low > high:
        raise ValueError("Attention and dense batch ranges do not overlap")
    batches = sorted(b for b in set(core_batches) | set(dense_batches) if low <= b <= high)

    rows = []
    for batch in batches:
        dense_weights = _bracket(dense_batches, batch, "dense batch_size")
        values = {name: multiple * math.fsum(weight * dense[name][b]
                                             for b, weight in dense_weights)
                  for name, multiple in DENSE_MULTIPLIERS.items()}
        core_weights = _bracket(core_batches, batch, "attention batch_size")
        for context in contexts:
            components = {"attention_core": math.fsum(weight * core[b, context]
                                                      for b, weight in core_weights), **values}
            if not math.isfinite(math.fsum(components.values())) or math.fsum(components.values()) <= 0:
                raise ValueError("Composed attention total must be finite and positive")
            for name, time_us in components.items():
                rows.append({"component": name, "batch_size": batch,
                             "context_length": context, "time_ms": time_us / 1000})
    return rows


def convert_bundle(source_dir, output_path, *, overwrite=False):
    """Verify the pinned raw bundle and write a CSV plus reproducible manifest."""
    source_dir, output_path = Path(source_dir), Path(output_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    output_target, manifest_target = output_path.resolve(), manifest_path.resolve()
    if output_target == manifest_target:
        raise ValueError("Output CSV and manifest resolve to the same path")
    source_targets = {(source_dir / filename).resolve() for filename in SOURCE_PATHS}
    if output_target in source_targets or manifest_target in source_targets:
        raise ValueError("Output CSV or manifest would overwrite a pinned source file")
    if not overwrite and (output_path.exists() or manifest_path.exists()):
        raise ValueError("Output CSV or manifest already exists; use a new path or --overwrite")
    sources = {}
    for filename, upstream_path in SOURCE_PATHS.items():
        contents = (source_dir / filename).read_bytes()
        sha = hashlib.sha256(contents).hexdigest()
        if sha != SOURCE_SHA256[filename]:
            raise ValueError(f"Pinned source hash mismatch: {filename}")
        sources[filename] = {
            "upstream_path": upstream_path,
            "url": f"{UPSTREAM_REPOSITORY}/blob/{UPSTREAM_COMMIT}/{upstream_path}",
            "sha256": sha,
        }
    with (source_dir / "attention.csv").open(newline="") as handle:
        attention_rows = list(csv.DictReader(handle))
    with (source_dir / "dense.csv").open(newline="") as handle:
        dense_rows = list(csv.DictReader(handle))
    rows = compose_decode_rows(attention_rows, dense_rows)
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=("component", "batch_size", "context_length", "time_ms"),
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    contents = text.getvalue().encode("utf-8")
    batches = sorted({row["batch_size"] for row in rows})
    contexts = sorted({row["context_length"] for row in rows})
    manifest = {
        "schema_version": 1,
        "kind": "composed_operator_profile_proxy",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "model": MODEL_ID,
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "variant": "bf16", "tensor_parallel_size": 1,
        "vllm_version": "0.28.0", "cuda_version": "13.0",
        "source_measurement": {"load_format": "dummy", "enforce_eager": True,
                               "num_hidden_layers": 1, "measurement_iterations": 3},
        "sources": sources,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output": {"sha256": hashlib.sha256(contents).hexdigest(), "rows": len(rows),
                   "batch_sizes": batches, "context_lengths": contexts, "time_unit": "ms"},
        "composition": {"attention_core": 1, **DENSE_MULTIPLIERS},
        "context_semantics": "KV history before current query; one decode query per request",
        "selection": "layer=attention, prefill_chunk=0, prefill_key=0, decode_q_len=1, n_decode>0",
        "grid": "union of core/dense batch breakpoints in common range; measured core context axis",
        "interpolation": "linear per axis; preserves dense measurements; core values at added batch points are interpolated",
        "normalization": "layernorm is per-call mean used twice; qk_norm combines Q and K and is used once",
        "hardware_calibrated_for_current_simulation": False,
        "limitations": [
            "RTXPRO6000 Attention plus existing A100 expert profile is a cross-hardware proxy, without hardware scaling.",
            "Upstream Qwen includes QK norm; current StreamInfer Attention omits it. Routing/gate/top-k is not included here.",
            "No CPU scheduling, batch reformation, communication, or CUDA Graph padding costs are added.",
            "Summed operator costs are not a measured end-to-end StreamInfer Attention-stage latency.",
            "Existing expert CSV remains a grouped single-GEMM per-expert proxy, not a full MLP measurement.",
            "One table is reused for all simulated layers; no prefill, heterogeneous-length correction, or extrapolation.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(contents)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = convert_bundle(args.source_dir, args.output, overwrite=args.overwrite)
    print(json.dumps({"output": str(args.output), **result["output"]}, indent=2))


if __name__ == "__main__":
    main()
