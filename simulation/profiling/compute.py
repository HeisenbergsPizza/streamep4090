"""Time one native StreamInfer layer on a single GPU of the recorded node.

Import CUDA dependencies only when collecting: inspecting CLI options and trace
coverage is possible on a machine that only runs the simulator.
"""
import csv
import gzip
import json
import os
from collections import Counter
from pathlib import Path
from statistics import median

from .common import StageWriter, load_manifest, measure_cuda, write_json


def integer_grid(value, minimum=1):
    values = [int(item.strip()) for item in value.split(",")]
    if min(values) < minimum or len(set(values)) != len(values):
        raise ValueError(f"Expected distinct integers >= {minimum}: {value}")
    return sorted(values)


def validate_counts(counts, local_experts, cap):
    if (not isinstance(counts, list) or len(counts) != local_experts
            or any(type(n) is not int or n < 0 for n in counts)):
        raise ValueError(f"Expected {local_experts} nonnegative integer expert counts: {counts}")
    if not 1 <= sum(counts) <= cap:
        raise ValueError(f"Expected 1..{cap} packed rows: {counts}")
    return tuple(counts)


def vector_workload(path, sizes, local_experts, cap):
    """Return exact ordered load vectors and occurrences (trace coverage denominator)."""
    frequencies = Counter()
    if path is not None:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        if item.get("stage") in ("attention", "attn", "finalize"):
                            continue
                        item = item["expert_counts"]
                    frequencies[validate_counts(item, local_experts, cap)] += 1
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not frequencies:
            raise ValueError(f"No expert dispatches in {path}")
    else:
        sizes = integer_grid(sizes) if sizes else sorted({1, min(8, cap), min(32, cap),
                                                           min(128, cap), cap})
        for total in sizes:
            for active in {1, min(4, local_experts), local_experts}:
                quotient, remainder = divmod(total, active)
                counts = [quotient + int(i < remainder) for i in range(active)]
                counts.extend([0] * (local_experts - active))
                frequencies[validate_counts(counts, local_experts, cap)] = 1
    return frequencies


def blocks_per_seq(history, block_size):
    # The query writes its own slot after the KV history. At an exact block
    # boundary the query needs a fresh page.
    return history // block_size + 1


def validate_graph_expert_cap(cap):
    if cap > 128 and cap & (cap - 1):
        raise ValueError("Native expert graphs round caps >128 up to a power of two; "
                         "choose a power-of-two cap so the static input buffers cover the last bucket.")


def configure(args, manifest, attn_cap, expert_cap, max_seq_len):
    import torch
    from disagmoe.config import EngineConfig, ModelConfig
    from disagmoe.frontend.engine_utils import set_global_engine_config
    from disagmoe.models.distributed import set_tensor_model_parallel_config
    from disagmoe.utils.logger import initialize_logger

    gpu = args.device
    if not torch.cuda.is_available() or torch.cuda.device_count() != len(manifest["hardware"]["gpus"]):
        raise ValueError("Select the bundle's visible CUDA devices, in their original order")
    if not 0 <= gpu < torch.cuda.device_count():
        raise ValueError(f"Invalid CUDA device {gpu}")
    expected = manifest["hardware"]["gpus"][gpu]
    props = torch.cuda.get_device_properties(gpu)
    if props.name != expected["name"]:
        raise ValueError(f"GPU {gpu}: found {props.name}, expected {expected['name']}")
    if props.major < 8:
        raise ValueError("Native BF16 compute profiling requires SM80 or newer")
    if manifest["runtime"]["dtype"] != "bfloat16" or manifest["parallel"]["tp"] != 1:
        raise ValueError("Native collectors support BF16 and attention TP=1")
    if min(attn_cap, expert_cap, args.warmup, args.repeats) < 1:
        raise ValueError("Caps, warmup, and repeats must be positive")
    graph = manifest["runtime"]["cuda_graph"]
    d = manifest["model"]
    torch.cuda.set_device(gpu)
    torch.set_default_device(f"cuda:{gpu}")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(args.seed)
    initialize_logger("device-profile")
    model = ModelConfig(
        hidden_size=d["hidden_size"], num_layers=d["num_layers"], layer_ids=[0],
        head_dim=d["head_dim"], num_heads=d["num_attention_heads"], num_kv_heads=d["num_key_value_heads"],
        num_experts=d["num_experts"], intermediate_size=d["intermediate_size"], top_k=d["top_k"],
        dtype=torch.bfloat16, max_seq_len=max_seq_len, ep_size=manifest["parallel"]["ep"],
        dp_size=manifest["parallel"]["dp"], tp_size=1, rank=0,
    )
    set_tensor_model_parallel_config(model)
    set_global_engine_config(EngineConfig(
        unified_scheduler_type="defrag", defrag_weight_decay=0.0, defrag_lookahead_steps=0,
        defrag_lookback_steps=0, enable_cuda_graph_expert=graph, enable_cuda_graph_attn=graph,
        enable_grouped_gemm=True, less_than_sm90=props.major < 9,
        max_batch_size_expert=expert_cap, max_batch_size_attn=attn_cap, max_attn_graph_bsz=attn_cap,
    ))
    return model


def record_grid(output, stage, keys, cases, make_callback, args):
    """A grid is published only after every point has positive CUDA-event samples."""
    with (output / f"{stage}.csv").open("w", newline="", encoding="utf-8") as dest, (
            output / "raw.jsonl").open("w", encoding="utf-8") as raw:
        writer = csv.DictWriter(dest, fieldnames=[*keys, "time_ms"])
        writer.writeheader()
        for case in cases:
            callback, extra = make_callback(case)
            # In graph mode execute() calls the native bucketed graph executor.
            # Do not put that whole operation inside another graph.
            samples = measure_cuda(callback, args.warmup, args.repeats)
            values = {key: case[key] for key in keys}
            csv_values = {key: json.dumps(v) if isinstance(v, (list, tuple)) else v
                          for key, v in values.items()}
            writer.writerow({**csv_values, "time_ms": median(samples)})
            raw.write(json.dumps({**case, **extra, "samples_ms": samples}, allow_nan=False) + "\n")


def collection_details(args, manifest, scope, backend):
    return dict(timing_scope=scope, backend=backend, device=args.device, seed=args.seed,
                dtype=manifest["runtime"]["dtype"], cuda_graph=manifest["runtime"]["cuda_graph"],
                warmup=args.warmup, repeats=args.repeats, statistic="median", unit="ms",
                timer="CUDA-event stream span of one isolated forward, synchronized after end. "
                      "The span can include gaps between host submissions; it is not a sum of "
                      "GPU kernel durations. Do not charge those gaps a second time as CPU service.",
                weights="synthetic BF16, single resident layer, repeated inputs and weights")


def verify_group_boundaries(operator, batch, output, counts, deep_gemm):
    """Check first/last packed row per active expert against independent BF16 linear ops.

    In particular, installed DeepGEMM versions may impose M-alignment constraints
    on neighboring groups. Never publish a fast timing for silently mixed groups.
    """
    import torch
    import torch.nn.functional as F

    offset = 0
    errors = []
    for expert_id, count in enumerate(counts):
        if count == 0:
            continue
        rows = [offset] if count == 1 else [offset, offset + count - 1]
        if deep_gemm:
            w13, w2 = operator.w13[expert_id], operator.w2[expert_id]
        else:
            w13, w2 = operator.w13_weight[expert_id].T, operator.w2_weight[expert_id].T
        gate, up = F.linear(batch.data[rows], w13).chunk(2, dim=-1)
        expected = F.linear(F.silu(gate) * up, w2).float()
        rel_l2 = (output[rows].float() - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-6)
        errors.append(rel_l2.max())
        offset += count
    # Large enough for different BF16 reductions; intended to detect wrong
    # expert or invalid rows, not define a model accuracy tolerance.
    max_error = torch.stack(errors).max().item()
    if not 0 <= max_error < 0.08:
        raise ValueError(f"Native grouped output failed boundary-row check (relative L2 {max_error:.3g}) "
                         f"for counts={list(counts)}. Verify backend support for this packed layout.")
    return max_error


def collect_experts(args):
    import torch

    manifest = load_manifest(args.bundle)
    cap = args.max_batch_size
    if cap <= 0:
        raise ValueError("--max-batch-size must be positive")
    grouped = args.stage == "grouped"
    local = manifest["model"]["num_experts"] // manifest["parallel"]["ep"] if grouped else 1
    graph = manifest["runtime"]["cuda_graph"]
    if graph:
        validate_graph_expert_cap(cap)
    if grouped:
        freq = vector_workload(args.vectors, args.batch_sizes, local, cap)
        if args.max_vectors is not None and args.max_vectors < 1:
            raise ValueError("--max-vectors must be positive")
        selected = freq.most_common(args.max_vectors)
        cases = [dict(expert_counts=counts, occurrences=n) for counts, n in selected]
        keys = ["expert_counts"]
    else:
        cases = [dict(batch_size=n) for n in range(1, cap + 1)]
        keys = ["batch_size"]

    with StageWriter(args.bundle, args.stage) as output, torch.inference_mode():
        model = configure(args, manifest, 1, cap, 4096)
        from disagmoe.executor.executor import ExpertsExecutor
        from disagmoe.frontend.datatypes import ExpertForwardBatch
        local_ids = list(range(local))
        global_map = local_ids + [-1] * (model.num_experts - local)
        runner = ExpertsExecutor(model, local_ids, global_map)
        deep_gemm = runner.expert_cls.__name__ == "MoEExpertsDeepGemmBF16"
        if graph:
            # The native capture fixture floor-divides rows across its expert
            # slots; the DeepGEMM staging kernel reads one index per input row.
            # All later buckets above 128 are multiples of that first bucket.
            if deep_gemm and min(cap, 128) % local:
                raise ValueError("Native DeepGEMM graph capture needs its first bucket "
                                 f"({min(cap, 128)}) divisible by the local expert count ({local})")
            runner.build_cuda_graph_executor()
        hiddens = torch.randn(cap, model.hidden_size, device="cuda", dtype=model.dtype)

        def make_callback(case):
            counts = case["expert_counts"] if grouped else [case["batch_size"]]
            rows = sum(counts)
            counts_gpu = torch.tensor(counts, device="cuda", dtype=torch.int64)
            indices = torch.repeat_interleave(torch.arange(local, device="cuda", dtype=torch.int32),
                                              counts_gpu, output_size=rows)
            batch = ExpertForwardBatch(0, hiddens[:rows], rows, None, None, None, counts_gpu, indices)
            extra = {"executed_bucket": runner.cuda_graph_executor._get_bucket_by_num_tokens(rows)
                     if graph else rows}
            if grouped:
                extra["boundary_row_max_relative_l2"] = verify_group_boundaries(
                    runner.operators[0], batch, runner.execute(batch), counts, deep_gemm,
                )
            return lambda: runner.execute(batch), extra

        record_grid(output, args.stage, keys, cases, make_callback, args)
        details = collection_details(args, manifest,
            "Native ExpertsExecutor.execute after input packing, device counts and indices ready; "
            "includes up/gate/activation/down and native graph staging when enabled. Excludes routing, "
            "dispatch, and output combination.", runner.expert_cls.__name__)
        details.update(local_expert_slots=local, max_batch_size=cap)
        if grouped:
            details.update(source=str(args.vectors) if args.vectors else "synthetic concentrated and balanced",
                           unique_vectors=len(freq), measured_vectors=len(cases),
                           input_dispatches=sum(freq.values()), covered_dispatches=sum(n for _, n in selected))
        write_json(output / "details.json", details)


def attention_batch(model, rows, history, block_size):
    import torch
    from disagmoe.frontend.datatypes import AttentionForwardBatch
    from vllm.attention.backends.flash_attn import FlashAttentionMetadata

    pages = blocks_per_seq(history, block_size)
    # KV pool reserves page 0 for graph padding. Every active sequence gets its
    # own pages, so long-context loads cannot all hit one shared cache block.
    table = torch.arange(1, rows * pages + 1, device="cuda", dtype=torch.int32).reshape(rows, pages)
    seq = history + 1
    seq_lens = torch.full((rows,), seq, dtype=torch.int32, device="cuda")
    starts = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
    meta = FlashAttentionMetadata(
        num_prefills=0, num_prefill_tokens=0, num_decode_tokens=rows,
        slot_mapping=table[:, -1].long() * block_size + history % block_size,
        seq_lens=[seq] * rows, seq_lens_tensor=seq_lens, max_query_len=0,
        max_prefill_seq_len=0, max_decode_seq_len=seq, max_decode_query_len=1,
        query_start_loc=starts, seq_start_loc=starts * seq,
        context_lens_tensor=seq_lens - 1, block_tables=table, use_cuda_graph=False,
        multi_modal_placeholder_index_maps=None, enable_kv_scales_calculation=True,
    )
    # The synthetic engine uses the current sequence length for its rotary position.
    return AttentionForwardBatch(0, torch.randn(rows, model.hidden_size, device="cuda", dtype=model.dtype),
                                 rows, None, None, None, seq_lens.long(), meta)


def collect_attention(args):
    import torch

    manifest = load_manifest(args.bundle)
    batches = integer_grid(args.batch_sizes)
    histories = integer_grid(args.contexts, minimum=0)
    cap = args.graph_cap if args.graph_cap is not None else max(batches)
    if cap < max(batches) or (manifest["runtime"]["cuda_graph"] and cap > 1024):
        raise ValueError("Graph cap must cover the measured grid; native Attention graph cap is <=1024")
    if (args.block_size not in (16, 32) or args.max_seq_len < max(histories) + 1
            or args.max_seq_len % args.block_size or args.max_seq_len >= 4096 * 32):
        raise ValueError("Use block size 16/32 and a block-aligned max sequence length "
                         "covering history + one query within the native synthetic RoPE table")
    if os.environ.get("STREAMEP_SHARED_TOP8") or os.environ.get("DMOE_WEIGHTED_ROUTER_FILE"):
        raise ValueError("Attention profile uses the native synthetic random router; "
                         "unset STREAMEP_SHARED_TOP8 and DMOE_WEIGHTED_ROUTER_FILE")
    if os.environ.setdefault("VLLM_USE_V1", "0") != "0":
        raise ValueError("StreamInfer requires the patched vLLM V0 attention path (VLLM_USE_V1=0)")
    from disagmoe.config import CacheConfig
    from disagmoe.executor.executor import AttnExecutor
    graph = manifest["runtime"]["cuda_graph"]
    with StageWriter(args.bundle, "attention") as output, torch.inference_mode():
        model = configure(args, manifest, cap, 1, args.max_seq_len)
        cache = CacheConfig(block_size=args.block_size, gpu_memory_utilization=0.80,
                            swap_space=0, cache_dtype="auto")
        cache.num_gpu_blocks = max(batches) * blocks_per_seq(max(histories), args.block_size)
        runner = AttnExecutor(model, cache)
        # Initialize realistic, finite data even though no checkpoint is loaded.
        for parameter in runner.operators[0].parameters():
            if parameter.is_floating_point():
                if parameter.ndim >= 2:
                    parameter.normal_(0, 0.02)
                else:
                    parameter.fill_(1)
        runner.kv_cache.get_kv_buffer(0).normal_(0, 0.02)
        if graph:
            runner.build_cuda_graph_executor()

        def make_callback(case):
            rows = case["batch_size"]
            batch = attention_batch(model, rows, case["context_length"], args.block_size)
            bucket = runner.cuda_graph_executor._get_bucket_by_num_tokens(rows) if graph else rows
            return lambda: runner.execute(batch), {"executed_bucket": bucket}

        cases = (dict(batch_size=b, context_length=l) for b in batches for l in histories)
        record_grid(output, "attention", ["batch_size", "context_length"], cases, make_callback, args)
        impl = runner.operators[0].attn.impl
        details = collection_details(args, manifest,
            "One native synthetic StreamInfer AttnExecutor.execute, pure decode q=1; includes norms, "
            "QKV/O projections, RoPE, page writes/attention, gate/random Top-K; includes graph "
            "staging/replay/output copy when enabled. Input metadata built before timing.",
            type(impl).__module__ + "." + type(impl).__name__)
        details.update(context_axis="KV tokens before the current query; equal histories per batch",
                       block_size=args.block_size, max_seq_len=args.max_seq_len, graph_cap=cap,
                       router="native synthetic random logits", kv_layout="disjoint contiguous pages per request")
        write_json(output / "details.json", details)


def add_commands(subparsers):
    for name, function, help_text in (
        ("expert", collect_experts, "Dense single-expert BF16 native forward curve"),
        ("grouped", collect_experts, "Native grouped BF16 forward by ordered local branch counts"),
        ("attention", collect_attention, "Full native synthetic decode Attention on a batch/context grid"),
    ):
        parser = subparsers.add_parser(name, help=help_text)
        parser.add_argument("--bundle", type=Path, required=True)
        parser.add_argument("--device", type=int, default=0, help="Logical CUDA device in the manifest")
        parser.add_argument("--warmup", type=int, default=5)
        parser.add_argument("--repeats", type=int, default=20)
        parser.add_argument("--seed", type=int, default=0)
        if name == "attention":
            parser.add_argument("--batch-sizes", default="1,8,16,32,64,128")
            parser.add_argument("--contexts", default="64,128,192", help="KV history lengths before the query")
            parser.add_argument("--block-size", type=int, default=16)
            parser.add_argument("--max-seq-len", type=int, default=4096)
            parser.add_argument("--graph-cap", type=int, help="Use the serving graph cap; defaults to largest batch")
        else:
            parser.add_argument("--max-batch-size", type=int, default=512)
        if name == "grouped":
            source = parser.add_mutually_exclusive_group()
            source.add_argument("--vectors", type=Path,
                                help="JSONL[.gz]: local count lists or scheduler decision records with expert_counts")
            source.add_argument("--batch-sizes", help="Packed row totals for synthetic balanced/hotspot cases")
            parser.add_argument("--max-vectors", type=int, help="Measure the most frequent N distinct vectors")
        parser.set_defaults(func=function, stage=name)
