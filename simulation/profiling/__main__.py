"""Collect and validate a single-node hardware profile for the StreamInfer simulator."""
import argparse
from pathlib import Path
import subprocess
import sys

from . import bundle, compute, host, manifest, network, reformation
from .common import load_manifest

DEVICE_STAGES = ("expert", "attention", "grouped", "reformation", "network")


def collect(args):
    """Use fresh processes so graph pools and per-executor runtime settings stay local."""
    load_manifest(args.bundle)
    stages = args.stages or DEVICE_STAGES
    if len(set(stages)) != len(stages):
        raise ValueError("Choose each collection stage once")
    for stage in stages:
        if (args.bundle / stage).exists():
            raise FileExistsError(f"Stage {stage} already exists. Select remaining --stages or start a new bundle.")
    for stage in stages:
        cmd = [sys.executable, "-m", "simulation.profiling", stage, "--bundle", str(args.bundle),
               "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
        if stage != "network":
            cmd.extend(("--device", str(args.device)))
        if stage in ("expert", "grouped"):
            cmd.extend(("--max-batch-size", str(args.expert_cap)))
        if stage == "grouped" and args.vectors:
            cmd.extend(("--vectors", str(args.vectors)))
        if stage == "attention":
            cmd.extend(("--batch-sizes", args.attention_batches, "--contexts", args.contexts))
        if stage == "reformation":
            rows = sorted({1, args.expert_cap, *(n for n in (8, 32, 64, 128, 256, 512) if n < args.expert_cap)})
            cmd.extend(("--rows", *map(str, rows)))
        print(f"Collecting {stage}", flush=True)
        subprocess.run(cmd, check=True)
    bundle.validate(args)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="command", required=True)
    for module in (manifest, compute, reformation, network, host, bundle):
        module.add_commands(cmds)
    p = cmds.add_parser("collect", help="Run GPU stages sequentially in separate processes")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--stages", choices=DEVICE_STAGES, nargs="+",
                   help="Default: all five GPU stages; select a subset when resuming")
    p.add_argument("--device", type=int, default=0, help="Representative GPU for single-device operators")
    p.add_argument("--expert-cap", type=int, default=512)
    p.add_argument("--attention-batches", default="1,8,16,32,64,128")
    p.add_argument("--contexts", default="64,128,192", help="KV histories before the query")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--vectors", type=Path, help="Local grouped expert counts as JSONL or JSONL.gz")
    p.set_defaults(func=collect)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in DEVICE_STAGES:
            manifest.verify_machine(load_manifest(args.bundle), args.command)
        args.func(args)
    except subprocess.CalledProcessError as exc:
        parser.exit(exc.returncode if exc.returncode > 0 else 1, "Collection subprocess failed; previous stages remain available.\n")
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(1, f"Profile error: {exc}\n")


if __name__ == "__main__":
    main()
