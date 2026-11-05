"""Export an equal-subject, reproducible decode-trace sample for the simulator.

Reads a local ``decode_routes.npy`` cache and its metadata without downloading
or modifying source files. Output expert IDs and Top-k ordering are preserved;
selecting fewer layers exports a documented prefix of the source model.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import random
import re
import tempfile

import numpy as np


AXES = ["request", "decode_step", "moe_layer", "top_k_choice"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must contain a SHA256 digest")
    return value.lower()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _empty_output(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output directory must be absent or empty: {path}")


def _read_source_manifest(cache_dir, metadata):
    reference = Path(_text(metadata.get("source_manifest"), "source_manifest"))
    array_reference = metadata.get("array_path")
    roots = []
    if isinstance(array_reference, str) and not Path(array_reference).is_absolute():
        roots = [root for root in (cache_dir, *cache_dir.parents)
                 if (root / array_reference).resolve() == cache_dir / "decode_routes.npy"]
    if reference.is_absolute():
        path = reference.resolve()
        project_root = roots[0] if roots else path.parent
    elif len(roots) == 1:
        project_root = roots[0]
        path = (project_root / reference).resolve()
    else:
        candidates = [(root, (root / reference).resolve())
                      for root in (cache_dir, *cache_dir.parents)
                      if (root / reference).is_file()]
        if len(candidates) != 1:
            raise ValueError("cannot resolve source_manifest relative to a unique trace project root")
        project_root, path = candidates[0]
    if not path.is_file():
        raise ValueError(f"source_manifest does not exist: {path}")
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    expected_sha = metadata.get("source_manifest_sha256")
    if expected_sha is not None and _sha(expected_sha, "source_manifest_sha256") != actual_sha:
        raise ValueError("source manifest SHA256 mismatch")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("source manifest must be a JSON object")
    return project_root, path, manifest, actual_sha, expected_sha is not None


def _source_model(metadata, source_manifest):
    prefix = _text(source_manifest.get("model_prefix"), "source manifest model_prefix").strip("/")
    parts = prefix.split("/")
    if any(not part for part in parts):
        raise ValueError("source manifest model_prefix has empty path components")
    # Download manifests use <organization>/<model>/<workload>/... . Also
    # support a model-only prefix or an explicit model identity in the manifest.
    explicit = source_manifest.get("source_model") or source_manifest.get("model_id")
    if explicit is not None:
        model = _text(explicit, "source model")
        if prefix != model and not prefix.startswith(model + "/"):
            raise ValueError("source model conflicts with model_prefix")
    elif len(parts) >= 2 and parts[1].lower() not in ("mmlu", "sharegpt", "gsm8k"):
        model = "/".join(parts[:2])
    else:
        model = parts[0]
    cached_model = metadata.get("source_model") or metadata.get("model_id")
    if cached_model is not None and cached_model != model:
        raise ValueError("cache source model conflicts with source manifest")
    return model, prefix


def _validate_cache(array, metadata):
    if array.ndim != 4 or any(size <= 0 for size in array.shape):
        raise ValueError("cache must have nonempty request/decode/layer/top-k dimensions")
    shape = metadata.get("shape")
    if not isinstance(shape, list) or len(shape) != 4:
        raise ValueError("metadata shape must contain four dimensions")
    shape = tuple(_integer(value, "metadata shape", 1) for value in shape)
    if tuple(array.shape) != shape:
        raise ValueError(f"cache shape {array.shape} differs from metadata shape {shape}")
    for name, actual in zip(("request_count", "decode_steps", "layer_count", "top_k"), shape):
        if _integer(metadata.get(name), name, 1) != actual:
            raise ValueError(f"metadata {name} does not match cache shape")
    if metadata.get("axes") != AXES:
        raise ValueError(f"metadata axes must be {AXES}")
    if array.dtype.kind not in "iu" or str(array.dtype) != metadata.get("dtype"):
        raise ValueError("cache dtype must be integer and match metadata dtype")
    expert_count = _integer(metadata.get("expert_count"), "expert_count", 1)
    if shape[-1] > expert_count:
        raise ValueError("top-k exceeds expert_count")
    bounds = metadata.get("expert_id_range", [0, expert_count - 1])
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("metadata expert_id_range must contain lower and upper bounds")
    low, high = [_integer(value, "expert_id_range") for value in bounds]
    if not 0 <= low <= high < expert_count:
        raise ValueError("metadata expert_id_range conflicts with expert_count")

    layer_ids = metadata.get("layer_ids")
    if not isinstance(layer_ids, list) or len(layer_ids) != shape[2]:
        raise ValueError("metadata layer_ids do not match layer_count")
    layer_ids = [_integer(layer, "source layer ID") for layer in layer_ids]
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("metadata contains duplicate layer IDs")
    entries = metadata.get("decode_entry_indices")
    if not isinstance(entries, list) or len(entries) != shape[1]:
        raise ValueError("metadata decode_entry_indices do not match decode_steps")
    entries = [_integer(index, "decode entry index", 1) for index in entries]
    if entries != list(range(entries[0], entries[0] + len(entries))):
        raise ValueError("decode_entry_indices must be contiguous and exclude entry zero")

    records = metadata.get("requests")
    if not isinstance(records, list) or len(records) != shape[0]:
        raise ValueError("metadata requests do not match cache request_count")
    indexed, original_ids, subjects = {}, set(), defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("metadata request records must be JSON objects")
        index = _integer(record.get("request_index"), "request_index")
        if index >= shape[0] or index in indexed:
            raise ValueError("metadata request_index values must uniquely cover cache rows")
        request_id = _text(record.get("request_id"), "source request_id")
        if request_id in original_ids:
            raise ValueError("metadata contains duplicate source request IDs")
        subject = _text(record.get("subject"), "subject")
        _sha(record.get("source_sha256"), "request source_sha256")
        original_ids.add(request_id)
        indexed[index] = record
        subjects[subject].append(index)
    actual_subject_counts = dict(Counter(record["subject"] for record in records))
    for name in ("subject_counts", "subject_counts_after_filter"):
        if name in metadata and metadata[name] != actual_subject_counts:
            raise ValueError(f"metadata {name} does not match request records")
    # Validate the entire source cache, including rows not selected for export.
    # Bounded chunks avoid a second full-cache allocation for Top-k sorting.
    for start in range(0, shape[0], 32):
        block = array[start:start + 32]
        if int(block.min()) < low or int(block.max()) > high:
            raise ValueError(f"cache expert IDs outside [{low}, {high}] near request {start}")
        ordered = np.sort(block, axis=-1)
        if np.any(ordered[..., 1:] == ordered[..., :-1]):
            raise ValueError(f"cache contains duplicate experts within Top-k near request {start}")
    return indexed, {subject: sorted(indices) for subject, indices in subjects.items()}, layer_ids, entries


def export_sample(cache_dir, output_dir, requests=256, output_tokens=32,
                  layers=None, seed=20261005):
    """Return the written sample manifest; never modify the source project.

    ``layers`` is a positive prefix length, or None to export every source
    layer. Output request IDs are the shuffled sample ordinals. Original IDs,
    source hashes, decode-entry indices and layer mappings live in the manifest.
    """
    requests = _integer(requests, "requests", 1)
    output_tokens = _integer(output_tokens, "output_tokens", 1)
    if output_tokens > 100:
        raise ValueError("output_tokens must not exceed 100 decoded tokens")
    if layers is not None:
        layers = _integer(layers, "layers", 1)
    seed = _integer(seed, "seed")
    cache_dir, output_dir = Path(cache_dir).resolve(), Path(output_dir).resolve()
    _empty_output(output_dir)
    array_path = cache_dir / "decode_routes.npy"
    metadata_path = cache_dir / "decode_routes_metadata.json"
    metadata_raw = metadata_path.read_bytes()
    metadata = json.loads(metadata_raw)
    if not isinstance(metadata, dict):
        raise ValueError("cache metadata must be a JSON object")
    project_root, source_path, source, source_sha, source_verified = _read_source_manifest(cache_dir, metadata)
    if output_dir == project_root or project_root in output_dir.parents:
        raise ValueError("output directory must be outside the source trace project")
    if output_dir == cache_dir or cache_dir in output_dir.parents:
        raise ValueError("output directory must be outside the source cache")
    source_model, model_prefix = _source_model(metadata, source)
    dataset = _text(metadata.get("dataset"), "dataset")
    revision = _text(metadata.get("manifest_revision"), "manifest_revision")
    if source.get("dataset") != dataset or source.get("revision") != revision:
        raise ValueError("cache dataset/revision differs from source manifest")
    expected_cache_sha = _sha(metadata.get("array_sha256"), "array_sha256")
    actual_cache_sha = sha256_file(array_path)
    if actual_cache_sha != expected_cache_sha:
        raise ValueError("cache array SHA256 mismatch")
    if "array_bytes" in metadata and metadata["array_bytes"] != array_path.stat().st_size:
        raise ValueError("cache byte count differs from metadata")
    array = np.load(array_path, mmap_mode="r", allow_pickle=False)
    indexed, subjects, source_layers, source_entries = _validate_cache(array, metadata)
    layer_count = array.shape[2] if layers is None else layers
    if layer_count > array.shape[2] or output_tokens > array.shape[1]:
        raise ValueError("requested layers/output_tokens exceed cache dimensions")
    if requests % len(subjects):
        raise ValueError(f"requests must be divisible by the {len(subjects)} subjects")
    quota = requests // len(subjects)
    if any(len(indices) < quota for indices in subjects.values()):
        raise ValueError("insufficient requests in at least one subject for sampling without replacement")
    rng = random.Random(seed)
    selected = []
    for subject in sorted(subjects):
        selected.extend(rng.sample(subjects[subject], quota))
    rng.shuffle(selected)
    selected_routes = array[selected, :output_tokens, :layer_count, :]
    flat = selected_routes.reshape(-1, array.shape[3])

    import pyarrow as pa
    import pyarrow.parquet as pq

    # Vectorized columns preserve C-order request/token/layer rows and expert
    # rank order, without constructing a Python dictionary for every route.
    columns = {
        "rid": pa.array(np.repeat(np.arange(requests, dtype=np.int64), output_tokens * layer_count)),
        "token_index": pa.array(np.tile(np.repeat(np.arange(output_tokens, dtype=np.int32), layer_count), requests)),
        "layer": pa.array(np.tile(np.arange(layer_count, dtype=np.int32), requests * output_tokens)),
    }
    columns.update({f"expert_logical_k{k}": pa.array(flat[:, k]) for k in range(array.shape[3])})
    table = pa.table(columns).replace_schema_metadata({
        b"source_model": source_model.encode(), b"source_cache_sha256": actual_cache_sha.encode(),
        b"sampling_seed": str(seed).encode(), b"synthetic": b"false",
    })
    mappings = [{"rid": rid, "cache_request_index": index,
                 "original_request_id": indexed[index]["request_id"],
                 "subject": indexed[index]["subject"], "source_sha256": indexed[index]["source_sha256"],
                 "source_request": indexed[index]}
                for rid, index in enumerate(selected)]
    manifest = {
        "schema_version": 1, "input_kind": "sampled_decode_routing_trace", "synthetic": False,
        "source_model": source_model, "model_prefix": model_prefix,
        "dataset": dataset, "revision": revision, "seed": seed,
        "request_count": requests, "output_tokens": output_tokens, "layer_count": layer_count,
        "expert_count": metadata["expert_count"], "top_k": array.shape[3],
        "source_layer_count": array.shape[2], "source_layer_ids": source_layers[:layer_count],
        "layer_selection": "all" if layer_count == array.shape[2] else "prefix",
        "layer_mapping": [{"layer": layer, "source_layer_id": source_layers[layer]}
                          for layer in range(layer_count)],
        "source_decode_entry_indices": source_entries[:output_tokens],
        "scope": "contiguous decode prefix; source entry zero/prefill excluded; expert IDs/ranks unchanged",
        "sampling": {"method": "equal quota per sorted subject without replacement; then shuffle requests",
                     "rng": "python.random.Random", "subjects": sorted(subjects), "requests_per_subject": quota,
                     "available_subject_counts": {subject: len(subjects[subject]) for subject in sorted(subjects)}},
        "source_cache": {"path": str(array_path), "sha256": actual_cache_sha, "sha256_verified": True,
                         "shape": list(array.shape), "dtype": str(array.dtype),
                         "metadata_path": str(metadata_path), "metadata_sha256": hashlib.sha256(metadata_raw).hexdigest()},
        "source_manifest": {"path": str(source_path), "sha256": source_sha, "sha256_verified": source_verified},
        "source_request_hash_scope": "preserved from verified-cache metadata; raw request files not reread",
        "requests": mappings,
        "routing_profile": str(output_dir / "routing.parquet"),
        "row_count": table.num_rows,
    }
    _empty_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sample-trace-", dir=output_dir) as temporary:
        temporary = Path(temporary)
        parquet_path = temporary / "routing.parquet"
        pq.write_table(table, parquet_path, compression="zstd", use_dictionary=False)
        manifest["routing_sha256"] = sha256_file(parquet_path)
        manifest_path = temporary / "sample_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
        os.replace(parquet_path, output_dir / parquet_path.name)
        os.replace(manifest_path, output_dir / manifest_path.name)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--layers", type=int, help="explicit source layer prefix length; default all layers")
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args(argv)
    try:
        manifest = export_sample(**vars(args))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: manifest[key] for key in
                      ("source_model", "request_count", "output_tokens", "layer_count", "layer_selection",
                       "routing_profile", "routing_sha256", "row_count")}, indent=2))


if __name__ == "__main__":
    main()
