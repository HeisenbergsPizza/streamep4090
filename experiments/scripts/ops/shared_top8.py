"""Prepare a shared MoE routing trace and verify the routes observed by both engines."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


TOP_K = 8
CHUNK_ROWS = 65536


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def draw_rows(rng, count, experts):
    """Ordered uniform draws without replacement via a partial Fisher-Yates shuffle."""
    rows = np.empty((count, TOP_K), dtype=np.uint8)
    for col in range(TOP_K):
        rows[:, col] = rng.integers(experts - col, size=count, dtype=np.uint8)
    # Undo each swap in reverse order to recover the expert at each drawn index.
    for col in reversed(range(TOP_K - 1)):
        later = rows[:, col + 1:]
        np.copyto(later, np.uint8(experts - 1 - col), where=later == rows[:, col:col + 1])
    return rows


def prepare(path, requests, steps, layers, experts, seed):
    if min(requests, steps, layers) < 1 or not TOP_K <= experts <= 256:
        raise ValueError("positive requests/steps/layers and 8..256 experts required")
    if path.suffix != ".npy":
        raise ValueError("output trace must end in .npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(seed))
    # Publish only a complete file. link() also protects an existing trace in concurrent runs.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".top8-", suffix=".npy") as tmp:
        shape = (requests, steps, layers, TOP_K)
        table = np.lib.format.open_memmap(tmp.name, mode="w+", dtype=np.uint8, shape=shape)
        rows = table.reshape(-1, TOP_K)
        for start in range(0, len(rows), CHUNK_ROWS):
            chunk = rows[start:start + CHUNK_ROWS]
            chunk[:] = draw_rows(rng, len(chunk), experts)
        table.flush()
        del table, rows, chunk
        os.link(tmp.name, path)
    return {"path": str(path.resolve()), "sha256": sha256(path), "shape": list(shape),
            "axes": ["request", "step", "layer", "expert_slot"], "dtype": "uint8",
            "experts": experts, "seed": seed, "algorithm": "pcg64-partial-fisher-yates-v1",
            "routing": "uniform ordered sample without replacement"}


def validate_audit(table, files, shape):
    seen = np.zeros(shape, dtype=bool)
    for path in files:
        with path.open() as f:
            for lineno, line in enumerate(f, 1):
                row = json.loads(line)
                phase = row.get("phase", "measured")
                if phase == "warmup":
                    continue
                where = f"{path}:{lineno}"
                if phase != "measured":
                    raise ValueError(f"{where}: unknown phase {phase!r}")
                coord = tuple(row.get(axis) for axis in ("request", "step", "layer"))
                if any(type(i) is not int or i < 0 for i in coord):
                    raise ValueError(f"{where}: invalid request/step/layer {coord}")
                if any(i >= bound for i, bound in zip(coord, table.shape)):
                    raise ValueError(f"{where}: coordinate outside trace {coord}")
                experts = row.get("experts")
                if not isinstance(experts, list) or any(type(i) is not int for i in experts):
                    raise ValueError(f"{where}: expected a list of integer expert IDs")
                expected = table[coord].tolist()
                if experts != expected:
                    raise ValueError(f"{where}: {coord} used {experts}, trace specifies {expected}")
                if any(i >= bound for i, bound in zip(coord, shape)):
                    continue
                if seen[coord]:
                    raise ValueError(f"{where}: duplicate measured coordinate {coord}")
                seen[coord] = True
    if not seen.all():
        first = tuple(int(i) for i in np.unravel_index(seen.argmin(), shape))
        raise ValueError(f"only {seen.sum()}/{seen.size} measured routes present; first missing {first}")
    return {"matched_unique_routes": int(seen.sum()), "files": [str(p) for p in files]}


def verify(trace_path, streaminfer, sglang, shape):
    table = np.load(trace_path, mmap_mode="r", allow_pickle=False)
    if table.ndim != 4 or table.shape[-1] != TOP_K or table.dtype != np.uint8:
        raise ValueError("expected a uint8 [request,step,layer,8] .npy trace")
    if any(n < 1 or n > bound for n, bound in zip(shape, table.shape[:3])):
        raise ValueError(f"invalid verification extent {shape} for trace shape {table.shape}")
    return {"trace_sha256": sha256(trace_path), "checked_extent": list(shape),
            "streaminfer": validate_audit(table, streaminfer, shape),
            "sglang": validate_audit(table, sglang, shape)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("prepare", help="write a new deterministic routing trace")
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--requests", type=int, default=1536)
    gen.add_argument("--steps", type=int, default=128)
    gen.add_argument("--layers", type=int, default=48)
    gen.add_argument("--experts", type=int, default=128)
    gen.add_argument("--seed", type=int, default=0)
    check = sub.add_parser("verify", help="check both engines against the same trace")
    check.add_argument("--trace", type=Path, required=True)
    check.add_argument("--streaminfer", type=Path, nargs="+", required=True)
    check.add_argument("--sglang", type=Path, nargs="+", required=True)
    for axis in ("requests", "steps", "layers"):
        check.add_argument(f"--{axis}", type=int, required=True,
                           help=f"require coverage of the first N {axis}")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output, args.requests, args.steps, args.layers, args.experts, args.seed)
    else:
        result = verify(args.trace, args.streaminfer, args.sglang, (args.requests, args.steps, args.layers))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
