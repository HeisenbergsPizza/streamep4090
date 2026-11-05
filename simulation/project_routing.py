"""Derive reproducible single-branch routing without changing simulator logic.

``uniform-from-topk`` selects one expert uniformly from every recorded Top-k
set. Rows are sorted by (rid, token_index, layer) before using the seeded RNG.
This is a counterfactual routing input, not a native Top-1 model's trace.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import random
import re
import tempfile

from simulation.router import ProfileDrivenRouter


KEY_COLUMNS = ["rid", "token_index", "layer"]
MODE = "uniform-from-topk"


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _empty_output(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output directory must be absent or empty: {path}")


def _expert_columns(names):
    if len(set(names)) != len(names):
        raise ValueError("Routing profile has duplicate column names")
    columns = []
    for name in names:
        if name.startswith("expert_logical_k"):
            match = re.fullmatch(r"expert_logical_k(0|[1-9]\d*)", name)
            if match is None:
                raise ValueError(f"Malformed expert column {name!r}")
            columns.append((int(match.group(1)), name))
    columns.sort()
    if not columns or [rank for rank, _ in columns] != list(range(len(columns))):
        raise ValueError("Expert columns must be contiguous expert_logical_k0..k(K-1)")
    return [name for _, name in columns]


def project_routing(input_path, output_dir, *, mode=MODE, seed=20261020,
                    experts, source_manifest=None):
    """Write a Top-1 Parquet and an auditable manifest, returning the manifest.

    The existing CPU router validates every source row, including layer
    coverage. Extra columns are recorded in the manifest and omitted from the
    simulator input. Keys and logical expert IDs are not renumbered.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if mode != MODE:
        raise ValueError(f"Unsupported mode {mode!r}; expected {MODE!r}")
    seed = _integer(seed, "seed")
    experts = _integer(experts, "experts", 1)
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    _empty_output(output_dir)
    raw = input_path.read_bytes()
    input_sha = hashlib.sha256(raw).hexdigest()
    try:
        table = pq.read_table(pa.BufferReader(raw))
    except Exception as exc:
        raise ValueError(f"Failed to parse Parquet routing profile: {exc}") from exc
    expert_columns = _expert_columns(table.column_names)
    source_k = len(expert_columns)
    router = ProfileDrivenRouter(raw, num_experts_expected=experts, top_k=source_k)
    source_reference = None
    if source_manifest is not None:
        source_manifest = Path(source_manifest).resolve()
        source_raw = source_manifest.read_bytes()
        source_data = json.loads(source_raw)
        if not isinstance(source_data, dict):
            raise ValueError("source manifest must be a JSON object")
        expected_hash = source_data.get("routing_sha256")
        if expected_hash is not None and expected_hash != input_sha:
            raise ValueError("source manifest routing_sha256 does not match input")
        source_reference = {
            "path": str(source_manifest),
            "sha256": hashlib.sha256(source_raw).hexdigest(),
            "routing_sha256_verified": expected_hash is not None,
            "scope": "reference only; model, workload and upstream provenance are not independently verified",
        }

    # Preserve key values and key Arrow types; canonical order makes choices
    # independent of the input Parquet row and row-group order.
    ordered = table.select(KEY_COLUMNS + expert_columns).sort_by(
        [(name, "ascending") for name in KEY_COLUMNS])
    rng = random.Random(seed)
    rank_counts = Counter()
    before, after = Counter(), Counter()
    chosen = []
    for row in ordered.select(expert_columns).to_pylist():
        candidates = [row[name] for name in expert_columns]
        rank = rng.randrange(source_k)
        expert = candidates[rank]
        chosen.append(expert)
        rank_counts[rank] += 1
        before.update(candidates)
        after[expert] += 1
    output = ordered.select(KEY_COLUMNS).append_column(
        "expert_logical_k0", pa.array(chosen, type=pa.int64()))
    output = output.replace_schema_metadata({
        b"routing_projection_mode": mode.encode(),
        b"routing_projection_seed": str(seed).encode(),
        b"source_routing_sha256": input_sha.encode(),
        b"source_top_k": str(source_k).encode(),
        b"top_k": b"1",
        b"synthetic": b"true",
        b"counterfactual_routing": b"true",
        b"native_top1_trace": b"false",
        b"hardware_calibrated": b"false",
    })
    manifest = {
        "schema_version": 1,
        "input_kind": "counterfactual_single_branch_routing",
        "mode": mode,
        "seed": seed,
        "source_top_k": source_k,
        "top_k": 1,
        "expert_count": experts,
        "row_count": output.num_rows,
        "request_count": router.num_profiled_requests,
        "layer_count": router.num_layers,
        "tokens_per_request": {
            str(rid): len(tokens)
            for rid, tokens in zip(router.profiled_request_ids, router.profiled_token_indices)
        },
        "sampling": {
            "rng": "python.random.Random",
            "operation": "randrange(source_top_k), once per sorted row",
            "row_order": KEY_COLUMNS,
            "replacement": "one independently selected rank per recorded row",
            "rank_semantics": "recorded column position; not assumed to be router-score rank",
        },
        "source_routing": {
            "path": str(input_path),
            "sha256": input_sha,
            "bytes": len(raw),
            "expert_columns": expert_columns,
            "schema_metadata": {
                key.decode("utf-8", errors="replace"): value.decode("utf-8", errors="replace")
                for key, value in (table.schema.metadata or {}).items()
            },
        },
        "source_manifest": source_reference,
        "preservation": {
            "source_files_modified": False,
            "keys_preserved": KEY_COLUMNS,
            "logical_expert_ids_preserved": True,
            "selection_is_source_topk_member": True,
            "dropped_columns": [
                name for name in table.column_names
                if name not in KEY_COLUMNS + expert_columns
            ],
        },
        "evidence_boundary": {
            "counterfactual_routing": True,
            "native_top1_trace": False,
            "hardware_calibrated": False,
            "description": (
                "Original later-layer routes are replayed after removing branches; "
                "no new model forward pass or compute profiling is performed. "
                "This derivation alone does not verify whether input rows are decoding."
            ),
        },
        "statistics": {
            "selected_source_rank_counts": {
                str(rank): rank_counts[rank] for rank in range(source_k)
            },
            "per_expert_assignment_counts": [
                {"expert": expert, "before": before[expert], "after": after[expert]}
                for expert in range(experts)
            ],
            "before_assignments": sum(before.values()),
            "after_assignments": sum(after.values()),
        },
        "routing_profile": str(output_dir / "routing.parquet"),
    }
    _empty_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".project-routing-", dir=output_dir) as temporary:
        temporary = Path(temporary)
        parquet_path = temporary / "routing.parquet"
        pq.write_table(output, parquet_path, compression="zstd", use_dictionary=False)
        manifest["routing_sha256"] = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        manifest_path = temporary / "projection_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        os.replace(parquet_path, output_dir / parquet_path.name)
        os.replace(manifest_path, output_dir / manifest_path.name)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=[MODE], default=MODE)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = project_routing(**vars(args))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: manifest[key] for key in
                      ("mode", "seed", "source_top_k", "top_k", "row_count",
                       "routing_profile", "routing_sha256")}, indent=2))


if __name__ == "__main__":
    main()
