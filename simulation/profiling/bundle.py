"""Validate published simulator inputs and resolve them on a CPU-only simulation host."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from ..costs import AttentionCostModel, ExpertCostModel
from ..overheads import PIPELINE_DEFAULTS, STRUCTURAL_DEFAULTS
from .common import STAGES, file_hash, load_manifest

TABLES = {"expert": "expert.csv", "attention": "attention.csv", "grouped": "grouped.csv"}
PARAMETERS = {
    "network": {"network_latency_us", "network_bandwidth_gbps"},
    "host": {name for name in (*PIPELINE_DEFAULTS, *STRUCTURAL_DEFAULTS) if name.startswith("cpu_")},
    "reformation": {name for name in STRUCTURAL_DEFAULTS if name.startswith("reformation_")}
                   | {"reduction_fixed_us", "reduction_bandwidth_gb_s"},
}


def inspect(bundle):
    bundle = Path(bundle).resolve()
    manifest = load_manifest(bundle)
    digest = file_hash(bundle / "manifest.json")
    summary = {"manifest_sha256": digest, "stages": [], "missing_stages": [], "parameters": {},
               "tables": {}, "coverage": {}}
    for stage in STAGES:
        directory = bundle / stage
        if not directory.exists():
            summary["missing_stages"].append(stage)
            continue
        stamp = json.loads((directory / "stage.json").read_text())
        if stamp.get("stage") != stage or stamp.get("manifest_sha256") != digest:
            raise ValueError(f"{stage} was collected with a different bundle manifest")
        hashes = stamp["sha256"]
        for filename, checksum in hashes.items():
            if Path(filename).name != filename or file_hash(directory / filename) != checksum:
                raise ValueError(f"Changed profile input in {stage}: {filename}")
        summary["stages"].append(stage)
        if stage in TABLES:
            if TABLES[stage] not in hashes:
                raise ValueError(f"Missing published {stage} cost table")
            summary["tables"][stage] = str(directory / TABLES[stage])
        if stage in PARAMETERS and (directory / "parameters.json").exists():
            if "parameters.json" not in hashes:
                raise ValueError(f"Unpublished parameters in {stage}")
            params = json.loads((directory / "parameters.json").read_text())
            values = {}
            for flag, value in params.items():
                key = flag.removeprefix("--").replace("-", "_")
                if flag != "--" + key.replace("_", "-") or key not in PARAMETERS[stage]:
                    raise ValueError(f"Unexpected simulator parameter {flag!r} in {stage}")
                if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid numeric value for {flag}")
                if "bandwidth" in key and value <= 0:
                    raise ValueError(f"Expected positive bandwidth for {flag}")
                values[key] = value
            summary["parameters"][stage] = values
    tables, coverage = summary["tables"], summary["coverage"]
    if "expert" in tables:
        from ..util import expert_compute_time_lookup_table_from_profile
        lookup = expert_compute_time_lookup_table_from_profile(tables["expert"], max_batch_size=1)
        if len(lookup) != max(lookup):
            raise ValueError("Single-expert table must cover every integer from 1 to cap")
        coverage["expert_max_rows"] = max(lookup)
    if "attention" in tables:
        attn = AttentionCostModel(1, profile_path=tables["attention"])
        if len(attn.lookup) != len(attn.batch_sizes) * len(attn.context_lengths):
            raise ValueError("Attention table must be a complete batch × history grid")
        coverage["attention_batches"] = list(attn.batch_sizes)
        coverage["kv_histories"] = list(attn.context_lengths)
    if "grouped" in tables:
        grouped = ExpertCostModel({}, grouped_profile_path=tables["grouped"]).grouped_lookup
        width = manifest["model"]["num_experts"] // manifest["parallel"]["ep"]
        if any(len(vector) != width or not any(vector) for vector in grouped):
            raise ValueError(f"Grouped workloads need {width} local expert slots and a positive row count")
        coverage["grouped_exact_vectors"] = len(grouped)
    summary["parameters_without_estimate"] = {
        stage: sorted(keys - summary["parameters"].get(stage, {}).keys() - {"reformation_scale"})
        for stage, keys in PARAMETERS.items()}
    return manifest, summary


def validate(args):
    _, summary = inspect(args.bundle)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def add_commands(subparsers):
    parser = subparsers.add_parser("validate", help="Check bundle hashes, available parameters and table coverage")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.set_defaults(func=validate)


def add_bundle_input(parser, required=False):
    source = parser.add_mutually_exclusive_group(required=required)
    source.add_argument("--expert-profile", type=Path)
    source.add_argument("--profile-bundle", type=Path, help="Measured single-node profile from simulation.profiling")
    parser.add_argument("--use-bundle-grouped", action="store_true",
                        help="Use exact-vector grouped measurements; missing vectors will fail at evaluation")


def parse_simulator_args(parser, argv):
    """Bundle values supply parser defaults; explicit workload/cap/parameter arguments still win."""
    argv = list(sys.argv[1:] if argv is None else argv)
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument("--profile-bundle", type=Path)
    probe.add_argument("--mode", default="async")
    probe.add_argument("--overhead-model")
    probe.add_argument("--use-bundle-grouped", action="store_true")
    options, _ = probe.parse_known_args(argv)
    if options.profile_bundle is None:
        args = parser.parse_args(argv)
        if args.use_bundle_grouped or args.profile_bundle is not None:
            parser.error("Specify the full option --profile-bundle DIRECTORY")
        return args
    try:
        manifest, summary = inspect(options.profile_bundle)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    tables, coverage = summary["tables"], summary["coverage"]
    if "expert" not in tables:
        parser.error("Profile bundle requires a completed dense expert stage")
    gpus = manifest["hardware"]["gpus"]
    if len({(gpu["name"], gpu["sm"]) for gpu in gpus}) != 1:
        parser.error("Simulator uses one shared compute table and needs homogeneous GPU types")
    dims = {"gpus": len(gpus), "hidden_dim": manifest["model"]["hidden_size"],
            "experts": manifest["model"]["num_experts"], "top_k": manifest["model"]["top_k"],
            "layers": manifest["model"]["num_layers"], "bytes_per_element": 2}
    profiles = {"expert_profile": Path(tables["expert"])}
    defaults = {"expert_batch_cap": coverage["expert_max_rows"]}
    if "attention" in tables:
        profiles["attention_profile"] = Path(tables["attention"])
        defaults.update(attention_batch_cap=max(coverage["attention_batches"]),
                        initial_context=min(coverage["kv_histories"]))
    if options.use_bundle_grouped:
        if "grouped" not in tables:
            parser.error("Collect the grouped stage first or omit --use-bundle-grouped")
        profiles["grouped_profile"] = Path(tables["grouped"])
    mode = options.overhead_model or ("pipeline" if options.mode == "async" else "none")
    defaults["overhead_model"] = mode
    calibrated = {}
    for stage, values in summary["parameters"].items():
        if stage == "network" or mode == "pipeline":
            calibrated.update(values)
    parser.set_defaults(**dims, **profiles, **defaults, **calibrated)
    args = parser.parse_args(argv)
    for key, expected in {**dims, **profiles}.items():
        if getattr(args, key) != expected:
            parser.error(f"--{key.replace('_', '-')} conflicts with the bundle ({expected})")
    if args.expert_batch_cap > coverage["expert_max_rows"]:
        parser.error("Expert cap exceeds measured coverage")
    if "attention" in tables:
        batches, histories = coverage["attention_batches"], coverage["kv_histories"]
        if min(batches) > 1 or args.attention_batch_cap > max(batches):
            parser.error("Attention grid must cover 1..attention-batch-cap")
        if args.initial_context < min(histories) or args.initial_context + args.output_tokens - 1 > max(histories):
            parser.error("Attention grid does not cover initial-context through initial-context + output-tokens - 1")
    applied = {k: v for k, v in calibrated.items() if getattr(args, k) == v}
    args.profile_bundle_info = dict(manifest_sha256=summary["manifest_sha256"], stages=summary["stages"],
                                    coverage=coverage, applied_parameters=applied,
                                    parameters_without_estimate=summary["parameters_without_estimate"],
                                    interconnect=manifest["hardware"]["interconnect"],
                                    runtime=manifest["runtime"], gpu_names=[gpu["name"] for gpu in gpus],
                                    missing_stages=summary["missing_stages"])
    return args
