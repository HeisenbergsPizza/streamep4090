from __future__ import annotations

import itertools
import json
import hashlib
import multiprocessing as mp
import os
import random
import sys
import time
import atexit
from typing import Any, Dict, List
from types import SimpleNamespace
from collections.abc import Callable


# Ensure repo root is on sys.path so `disagmoe` imports work when this file is
# executed as a script (e.g. `python simulation/experiment.py sync`).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if __package__:
    from . import sim_sync, sim_tbo, sim_async
    from .overheads import BatchOverheadModel, overhead_config_from_args
else:
    import sim_sync
    import sim_tbo
    import sim_async
    from overheads import BatchOverheadModel, overhead_config_from_args


def _parse_int_list_env(name: str, default: List[int]) -> List[int]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return list(default)
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or list(default)


def _resolve_expert_cost_profile_path() -> str:
    profile = (os.environ.get("EXPERT_COST_PROFILE") or "").strip()
    if not profile:
        profile = "GPT-OSS-120B_A100.csv"
    if os.path.isabs(profile):
        return profile
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "expert_costs_profiles", profile)


def _resolve_gating_profile_path() -> str:
    profile = (os.environ.get("GATING_PROFILE") or "").strip()
    if not profile:
        profile = "gating_gptoss120b_200.parquet"
    if os.path.isabs(profile):
        return profile
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gating_profiles", profile)


# ============================================================
# Configuration macros (all tunables gathered in one place)
# ============================================================

# ATTN_SERVICE_T_VALUES = [1, 2, 4]
ATTN_SERVICE_T_VALUES = [float(x) for x in os.environ.get("ATTN_SERVICE_TIMES", "0.1,0.2,0.4").split(",")]
# EP_GROUP_SIZE_VALUES = [8, 16, 32]
EP_GROUP_SIZE_VALUES = _parse_int_list_env("EP_GROUP_SIZES", [4])
GLOBAL_BATCH_MULTIPLIERS = _parse_int_list_env("GLOBAL_BATCH_MULTIPLIERS", [128, 256])

# Simple topology model:
# - GPUs [0..N_GPU_PER_HOST-1] are on host 0, [N_GPU_PER_HOST..2*N_GPU_PER_HOST-1] on host 1, etc.
# - Network delay depends on whether src/dst GPUs are on the same host.
N_GPU_PER_HOST = int(os.environ.get("N_GPU_PER_HOST", "4"))
NET_DELAY_VALUES_INTRA_HOST = [0.04]
NET_DELAY_VALUES_INTER_HOST = [0.08]

# Toggle bandwidth-aware (congestion-modeled) network simulation.
# When True, transfer times are computed from data size and link bandwidth with
# receiver-side FIFO serialization.  When False, fixed per-hop delays are used.
# Both values are swept in the experiment grid so results can be compared.
ENBALE_BANDWIDTH_UNAWARE_COMPARISON = False
ENABLE_BANDWIDTH_AWARE_NETWORK_VALUES = (
    [True, False] if ENBALE_BANDWIDTH_UNAWARE_COMPARISON else [True]
)

# ------------------------------
# Model + cost-profile macros
# ------------------------------
MODEL_NUM_LAYERS = int(os.environ.get("MODEL_NUM_LAYERS", "36") or "36")
MODEL_ROUTING_TOPK = int(os.environ.get("MODEL_ROUTING_TOPK", "4") or "4")
MODEL_NUM_EXPERTS = int(os.environ.get("MODEL_NUM_EXPERTS", "128") or "128")

HIDDEN_DIM_VALUES = _parse_int_list_env("MODEL_HIDDEN_DIMS", default=[2880])  # bytes/token for comm model
BYTES_PER_ELEMENT = 2                      # FP16 = 2 bytes
INTRA_NODE_BANDWIDTH_GBPS = 800.0          # NVLink bandwidth (Gb/s)
INTER_NODE_BANDWIDTH_VALUES = _parse_int_list_env("INTER_NODE_BANDWIDTHS", [200])   # InfiniBand/RoCE bandwidth (Gb/s) to sweep
TICKS_PER_MILLISECOND = 10                 # Conversion factor for simulation time

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "1"))
TOTAL_REQUESTS = int(os.environ.get("TOTAL_REQUESTS", "8192"))
TOKENS_PER_REQUEST = int(os.environ.get("TOKENS_PER_REQUEST", "256"))
RNG_SEED = int(os.environ.get("RNG_SEED", "42"))
SCHEDULING_POLICY = os.environ.get("SCHEDULING_POLICY", "defrag")
MAX_ATTENTION_BATCH_SIZE = int(os.environ.get("MAX_ATTENTION_BATCH_SIZE", "160"))
EXPERT_COST_SCALE = float(os.environ.get("EXPERT_COST_SCALE", "1.0"))
ATTENTION_PROFILE = os.environ.get("ATTENTION_PROFILE") or None
GROUPED_PROFILE = os.environ.get("GROUPED_PROFILE") or None
INITIAL_CONTEXT_LENGTH = int(os.environ.get("INITIAL_CONTEXT_LENGTH", "0"))


def _optional_float_env(name):
    raw = (os.environ.get(name) or "").strip()
    return float(raw) if raw else None


# Resolve and validate before workers create output files. Empty optional values
# retain the same stage-specific/default assumptions as the newer CLI.
OVERHEAD_CONFIG = overhead_config_from_args(SimpleNamespace(
    overhead_model=os.environ.get("OVERHEAD_MODEL", "none"),
    cpu_schedule_scale=float(os.environ.get("CPU_SCHEDULE_SCALE", "1")),
    cpu_schedule_overlap=_optional_float_env("CPU_SCHEDULE_OVERLAP"),
    reformation_compute_ratio=_optional_float_env("REFORMATION_COMPUTE_RATIO"),
    reformation_overlap=float(os.environ.get("REFORMATION_OVERLAP", "0")),
))

# Enable early termination once max concurrent active requests is hit.
ENABLE_EARLY_TERMINATION_AFTER_MAX_BS = os.environ.get("EARLY_TERMINATION", "1") == "1"

# Once max concurrent active requests is first reached, let this many additional
# tokens complete before stopping and reporting metrics.
TOKENS__AFTER_REACHING_MAX_BS = int(os.environ.get("TOKENS_AFTER_SATURATION", "8000"))

# Optional: record a short post-saturation schedule timeline for the TBO simulator.
# Set env var `TBO_TIMELINE_CAPTURE_OPS` (e.g. "100") to enable.
try:
    TBO_TIMELINE_CAPTURE_OPS = int(os.environ.get("TBO_TIMELINE_CAPTURE_OPS", "0") or "0")
except ValueError:
    TBO_TIMELINE_CAPTURE_OPS = 0

# Async-only: after the system first reaches max concurrent active requests, keep
# updating a single defrag-v0 scheduler debug snapshot (queue lengths + score matrix),
# and write the *last* snapshot observed before simulation termination.
ENABLE_DEFRAG_V0_DEBUG_LOG = False
DEFRAG_V0_DEBUG_LOG_BASENAME = "experiment_defrag_v0_debug_async.txt"

# Per-token stats: sampled per-token latency ("ITL") logging.
ENABLE_PER_TOKEN_STATS = True
TOKEN_SAMPLING_RATE = 0.5


_PER_TOKEN_STATS_F = None
_PER_TOKEN_STATS_RNG: random.Random | None = None
_PER_TOKEN_STATS_SAMPLING_RATE: float = 0.0


def _results_dir() -> str:
    return os.path.abspath(os.environ.get("SIM_OUTPUT_DIR") or os.path.dirname(__file__))


def _per_token_stats_dir() -> str:
    # Resolve independently of cwd, with an optional SIM_OUTPUT_DIR override.
    return os.path.join(_results_dir(), "per-token-stats")


def _close_per_token_stats_file() -> None:
    global _PER_TOKEN_STATS_F
    f = _PER_TOKEN_STATS_F
    _PER_TOKEN_STATS_F = None
    if f is not None:
        try:
            f.close()
        except Exception:
            pass


def _init_worker_per_token_stats(mode: str, enabled: bool, sampling_rate: float) -> None:
    global _PER_TOKEN_STATS_F, _PER_TOKEN_STATS_RNG, _PER_TOKEN_STATS_SAMPLING_RATE

    if not enabled:
        _PER_TOKEN_STATS_F = None
        _PER_TOKEN_STATS_RNG = None
        _PER_TOKEN_STATS_SAMPLING_RATE = 0.0
        return

    sampling_rate = float(sampling_rate)
    if not (0.0 <= sampling_rate <= 1.0):
        raise ValueError("TOKEN_SAMPLING_RATE must be in [0.0, 1.0]")

    os.makedirs(_per_token_stats_dir(), exist_ok=True)
    worker_name = mp.current_process().name
    out_path = os.path.join(
        _per_token_stats_dir(),
        f"per_token_stats_{mode}_{worker_name}.csv",
    )

    # Overwrite per-worker log file if it exists.
    _PER_TOKEN_STATS_F = open(out_path, "w", encoding="utf-8", newline="", buffering=1)
    _PER_TOKEN_STATS_F.write(
        "mode,worker,pid,ep_group_size,global_request_max_batch_size,attn_service_t,"
        "n_gpu_per_host,net_delay_intra_host,net_delay_inter_host,"
        "hidden_dim,inter_node_bw_gbps,bw_aware,"
        "request_id,token_index,tid,birth_time,completion_time,latency_ticks,latency_ms\n"
    )
    _PER_TOKEN_STATS_F.flush()

    _PER_TOKEN_STATS_RNG = random.Random(RNG_SEED)
    _PER_TOKEN_STATS_SAMPLING_RATE = sampling_rate

    atexit.register(_close_per_token_stats_file)


def _per_token_stats_begin_sample(token: Any) -> None:
    # Per-token deterministic sampling is independent of process assignment.
    seed = (RNG_SEED << 64) ^ (int(token.request_id) << 32) ^ int(token.token_index)
    rng = random.Random(seed) if _PER_TOKEN_STATS_RNG is not None else None
    if rng is None:
        return
    rate = _PER_TOKEN_STATS_SAMPLING_RATE
    if rate <= 0.0:
        return
    sampled = True if rate >= 1.0 else (rng.random() < rate)
    try:
        setattr(token, "sampled_for_stats", bool(sampled))
    except Exception:
        pass


def _per_token_stats_log_if_sampled(
    *,
    mode: str,
    cfg: Dict[str, Any],
    token: Any,
    completion_time: float,
    ticks_per_ms: float,
) -> None:
    f = _PER_TOKEN_STATS_F
    if f is None:
        return

    if not bool(getattr(token, "sampled_for_stats", False)):
        return

    previous = getattr(token, "previous_completion_time", None)
    if previous is None:  # first output has no inter-token interval
        return
    latency_ticks = completion_time - float(previous)
    latency_ms = latency_ticks / float(ticks_per_ms) if ticks_per_ms else 0.0

    worker_name = mp.current_process().name
    pid = os.getpid()

    f.write(
        f"{mode},"
        f"{worker_name},"
        f"{pid},"
        f"{cfg.get('ep_group_size')},"
        f"{cfg.get('global_request_max_batch_size')},"
        f"{cfg.get('attn_service_t')},"
        f"{cfg.get('n_gpu_per_host')},"
        f"{cfg.get('net_delay_intra_host')},"
        f"{cfg.get('net_delay_inter_host')},"
        f"{cfg.get('hidden_dim', 0)},"
        f"{cfg.get('inter_node_bw_gbps', 0.0)},"
        f"{cfg.get('bw_aware', True)},"
        f"{getattr(token, 'request_id', '')},"
        f"{getattr(token, 'token_index', '')},"
        f"{getattr(token, 'tid', '')},"
        f"{getattr(token, 'birth_time', '')},"
        f"{completion_time},"
        f"{latency_ticks},"
        f"{latency_ms}\n"
    )


def _persist_run_result(mode, cfg, result):
    """Keep full research metrics alongside the historical plotting schema."""
    metadata = {
        "mode": mode, "config": cfg,
        "policy": SCHEDULING_POLICY if mode == "async" else mode,
        "seed": RNG_SEED, "total_requests": TOTAL_REQUESTS,
        "tokens_per_request": TOKENS_PER_REQUEST,
        "num_layers": MODEL_NUM_LAYERS, "num_experts": MODEL_NUM_EXPERTS,
        "top_k": MODEL_ROUTING_TOPK, "expert_cost_scale": EXPERT_COST_SCALE,
        "expert_profile": _resolve_expert_cost_profile_path(),
        "routing_profile": _resolve_gating_profile_path(),
        "attention_profile": ATTENTION_PROFILE, "grouped_profile": GROUPED_PROFILE,
        "initial_context_length": INITIAL_CONTEXT_LENGTH,
        "overhead_config": OVERHEAD_CONFIG,
        "cost_assumptions": {"batch_overhead": BatchOverheadModel(
            ticks_per_millisecond=TICKS_PER_MILLISECOND, **OVERHEAD_CONFIG).describe()},
        "max_attention_batch_size": MAX_ATTENTION_BATCH_SIZE,
        "early_termination": ENABLE_EARLY_TERMINATION_AFTER_MAX_BS,
        "tokens_after_saturation": TOKENS__AFTER_REACHING_MAX_BS,
        "network_model": "receive_fifo_bytes_over_bandwidth" if cfg["bw_aware"] else "fixed_remote_delay_no_contention",
        "network_note": "Fixed-delay columns are unused in bandwidth mode; fixed mode uses inter_host delay for all remote GPUs.",
        "throughput_note": "avg_throughput_req_per_sec counts completed tokens divided by output length; completed_requests_per_sec counts whole requests.",
    }
    for name in ("DEFRAG_LOOKAHEAD", "DEFRAG_LOOKBACK", "DEFRAG_WEIGHT_DECAY"):
        metadata[name] = os.environ.get(name, {"DEFRAG_LOOKAHEAD": "4", "DEFRAG_LOOKBACK": "4", "DEFRAG_WEIGHT_DECAY": "0.8"}[name])
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    folder = os.path.join(_results_dir(), "run-details")
    os.makedirs(folder, exist_ok=True)
    output = os.path.join(folder, f"{mode}_{metadata['policy']}_{digest}.json")

    def safe(value):
        if isinstance(value, float) and not (value == value and abs(value) != float("inf")):
            return None
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        return value

    with open(output, "w") as stream:
        json.dump(safe({"metadata": metadata, "result": {
            key: value for key, value in result.items() if key != "completion_times"
        }}), stream, indent=2, allow_nan=False)
    return output


def _build_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for ep_group_size, attn_t, net_delay_intra, net_delay_inter, mult, hidden_dim, inter_bw, bw_aware in itertools.product(
        EP_GROUP_SIZE_VALUES,
        ATTN_SERVICE_T_VALUES,
        NET_DELAY_VALUES_INTRA_HOST,
        NET_DELAY_VALUES_INTER_HOST,
        GLOBAL_BATCH_MULTIPLIERS,
        HIDDEN_DIM_VALUES,
        INTER_NODE_BANDWIDTH_VALUES,
        ENABLE_BANDWIDTH_AWARE_NETWORK_VALUES,
    ):
        global_batch = ep_group_size * mult
        configs.append(
            {
                "ep_group_size": ep_group_size,
                "attn_service_t": attn_t,
                "n_gpu_per_host": N_GPU_PER_HOST,
                "net_delay_intra_host": net_delay_intra,
                "net_delay_inter_host": net_delay_inter,
                "global_request_max_batch_size": global_batch,
                "hidden_dim": hidden_dim,
                "inter_node_bw_gbps": inter_bw,
                "bw_aware": bool(bw_aware),
            }
        )
    return configs


def _make_topology_net_delay_fn(
    *,
    n_gpu_per_host: int,
    net_delay_intra_host: float,
    net_delay_inter_host: float,
):
    n_gpu_per_host = max(1, int(n_gpu_per_host))
    net_delay_intra_host = float(net_delay_intra_host)
    net_delay_inter_host = float(net_delay_inter_host)

    def net_delay_fn(src_gpu: int, dst_gpu: int) -> float:
        if src_gpu == dst_gpu:
            return 0.0
        src_host = int(src_gpu) // n_gpu_per_host
        dst_host = int(dst_gpu) // n_gpu_per_host
        return net_delay_intra_host if src_host == dst_host else net_delay_inter_host

    return net_delay_fn


def _make_bandwidth_aware_delay_fn(
    *,
    n_gpu_per_host: int,
    hidden_dim: int,
    bytes_per_elem: int = BYTES_PER_ELEMENT,
    intra_node_bw_gbps: float = INTRA_NODE_BANDWIDTH_GBPS,
    inter_node_bw_gbps: float,
    ticks_per_ms: float = TICKS_PER_MILLISECOND,
):
    """
    Create a bandwidth-aware delay function that computes transfer time
    based on the amount of data being transferred.

    Args:
        n_gpu_per_host: Number of GPUs per host for topology
        hidden_dim: Hidden dimension of the model
        bytes_per_elem: Bytes per element (2 for FP16, 4 for FP32)
        intra_node_bw_gbps: Intra-node bandwidth in Gb/s
        inter_node_bw_gbps: Inter-node bandwidth in Gb/s
        ticks_per_ms: Conversion factor from ms to simulation ticks

    Returns:
        A function(src_gpu, dst_gpu, num_tokens=1) -> delay_ticks
    """
    n_gpu_per_host = max(1, int(n_gpu_per_host))
    bytes_per_token = int(hidden_dim) * int(bytes_per_elem)
    intra_bw_bytes_per_ms = float(intra_node_bw_gbps) * 1e9 / 8.0 / 1000.0  # Gb/s -> bytes/ms
    inter_bw_bytes_per_ms = float(inter_node_bw_gbps) * 1e9 / 8.0 / 1000.0  # Gb/s -> bytes/ms

    def net_delay_fn(src_gpu: int, dst_gpu: int, num_tokens: int = 1) -> float:
        if src_gpu == dst_gpu:
            return 0.0
        src_host = int(src_gpu) // n_gpu_per_host
        dst_host = int(dst_gpu) // n_gpu_per_host

        data_bytes = float(num_tokens) * float(bytes_per_token)
        bandwidth = intra_bw_bytes_per_ms if src_host == dst_host else inter_bw_bytes_per_ms

        transfer_time_ms = data_bytes / bandwidth
        transfer_time_ticks = transfer_time_ms * float(ticks_per_ms)
        return transfer_time_ticks

    return net_delay_fn


def _run_async_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if sim_async is None:
        raise RuntimeError(
            "Async simulator dependencies not available. "
            "Install `simpy` (and other requirements) or run `sync`/`tbo` modes instead."
        )
    # Configure per-run timing parameters via module globals,
    # then call the async simulator with explicit ep_group_size and batch size.
    sim_async.ATTN_SERVICE_T = cfg["attn_service_t"]
    # Keep module globals set for backward-compat / defaults; the actual delay
    # applied in the simulator is driven by `net_delay_fn` below.
    sim_async.NET_T_ATTN_TO_EXPERT = cfg["net_delay_inter_host"]
    sim_async.NET_T_EXPERT_TO_ATTN = cfg["net_delay_inter_host"]

    net_delay_fn = None
    if cfg.get("bw_aware", True):
        net_delay_fn = _make_bandwidth_aware_delay_fn(
            n_gpu_per_host=cfg["n_gpu_per_host"],
            hidden_dim=cfg["hidden_dim"],
            bytes_per_elem=BYTES_PER_ELEMENT,
            intra_node_bw_gbps=INTRA_NODE_BANDWIDTH_GBPS,
            inter_node_bw_gbps=cfg["inter_node_bw_gbps"],
            ticks_per_ms=sim_async.TICKS_PER_MILLISECOND,
        )

    tokens_after_reaching_max_bs = (
        TOKENS__AFTER_REACHING_MAX_BS if ENABLE_EARLY_TERMINATION_AFTER_MAX_BS else None
    )
    per_token_begin_cb: Callable[[Any], None] | None = None
    per_token_stats_cb: Callable[[Any, float], None] | None = None
    if ENABLE_PER_TOKEN_STATS:
        ticks_per_ms = sim_async.TICKS_PER_MILLISECOND
        per_token_begin_cb = _per_token_stats_begin_sample

        def _per_token_stats_cb(token: Any, t_complete: float) -> None:
            _per_token_stats_log_if_sampled(
                mode="async",
                cfg=cfg,
                token=token,
                completion_time=t_complete,
                ticks_per_ms=ticks_per_ms,
            )

        per_token_stats_cb = _per_token_stats_cb

    result = sim_async.run_simulation(
        ep_group_size=cfg["ep_group_size"],
        global_request_max_batch_size=cfg["global_request_max_batch_size"],
        attn_dp_group_size=cfg["ep_group_size"],
        total_expert_count=MODEL_NUM_EXPERTS,
        num_layers=MODEL_NUM_LAYERS,
        routing_top_k=MODEL_ROUTING_TOPK,
        total_requests=TOTAL_REQUESTS,
        tokens_per_request=TOKENS_PER_REQUEST,
        rng_seed=RNG_SEED,
        max_attention_batch_size=MAX_ATTENTION_BATCH_SIZE,
        expert_cost_scale=EXPERT_COST_SCALE,
        overhead_config=OVERHEAD_CONFIG,
        hidden_dim=cfg["hidden_dim"], bytes_per_element=BYTES_PER_ELEMENT,
        attention_profile_path=ATTENTION_PROFILE,
        grouped_profile_path=GROUPED_PROFILE,
        initial_context_length=INITIAL_CONTEXT_LENGTH,
        expert_profile_path=_resolve_expert_cost_profile_path(),
        profile_routing_path=_resolve_gating_profile_path(),
        tokens_after_reaching_max_bs=tokens_after_reaching_max_bs,
        enable_defrag_v0_debug_log=ENABLE_DEFRAG_V0_DEBUG_LOG,
        scheduling_policy=SCHEDULING_POLICY,
        defrag_lookahead_steps=int(os.environ.get("DEFRAG_LOOKAHEAD", "4")),
        defrag_lookback_steps=int(os.environ.get("DEFRAG_LOOKBACK", "4")),
        defrag_weight_decay=float(os.environ.get("DEFRAG_WEIGHT_DECAY", "0.8")),
        per_token_begin_cb=per_token_begin_cb,
        per_token_stats_cb=per_token_stats_cb,
        net_delay_fn=net_delay_fn,
        n_gpu_per_host=cfg["n_gpu_per_host"],
    )

    _persist_run_result("async", cfg, result)

    ticks_per_ms = sim_async.TICKS_PER_MILLISECOND
    avg_latency_ms = (
        result["avg_latency"] / ticks_per_ms if result["avg_latency"] else 0.0
    )
    makespan_ms = (
        result["makespan"] / ticks_per_ms if result["makespan"] else 0.0
    )

    avg_per_expert_batch_size = result.get("avg_per_expert_batch_size", 0.0)
    tail_throughput_req_per_sec = result.get(
        "tail_throughput_req_per_sec", float("nan")
    )

    debug_text = None
    if ENABLE_DEFRAG_V0_DEBUG_LOG and result.get("defrag_v0_debug_triggered"):
        entry = result.get("defrag_v0_debug_entry") or {}
        debug_text = json.dumps({
            "config": cfg, "scheduling_policy": SCHEDULING_POLICY,
            "last_schedule_after_saturation": entry,
        }, indent=2)

    summary: Dict[str, Any] = {
        "mode": "async",
        "ep_group_size": cfg["ep_group_size"],
        "global_request_max_batch_size": cfg["global_request_max_batch_size"],
        "attn_service_t": cfg["attn_service_t"],
        "n_gpu_per_host": cfg["n_gpu_per_host"],
        "net_delay_intra_host": cfg["net_delay_intra_host"],
        "net_delay_inter_host": cfg["net_delay_inter_host"],
        "hidden_dim": cfg.get("hidden_dim", 0),
        "inter_node_bw_gbps": cfg.get("inter_node_bw_gbps", 0),
        "bw_aware": cfg.get("bw_aware", True),
        "avg_token_latency_ms": avg_latency_ms,
        "makespan_ms": makespan_ms,
        "avg_throughput_req_per_sec": result.get("avg_throughput_req_per_sec", 0.0),
        "tail_throughput_req_per_sec": tail_throughput_req_per_sec,
        "avg_request_latency_ms": result.get("avg_request_latency_ms", 0.0),
        "p90_request_latency_ms": result.get("p90_request_latency_ms", 0.0),
        "p99_request_latency_ms": result.get("p99_request_latency_ms", 0.0),
        # Not applicable for async simulator; keep for uniform schema.
        "avg_layer_runtime_ms": float("nan"),
        "avg_layer_wait_imbalance_ms": float("nan"),
        "avg_per_expert_batch_size": avg_per_expert_batch_size,
        "avg_layer_worker_queue_stddev": float("nan"),
        "defrag_v0_debug_text": debug_text,
    }
    return summary


def _run_sync_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tokens_after_reaching_max_bs = (
        TOKENS__AFTER_REACHING_MAX_BS if ENABLE_EARLY_TERMINATION_AFTER_MAX_BS else None
    )
    per_token_begin_cb: Callable[[Any], None] | None = None
    per_token_stats_cb: Callable[[Any, float], None] | None = None
    if ENABLE_PER_TOKEN_STATS:
        ticks_per_ms = sim_sync.TICKS_PER_MILLISECOND
        per_token_begin_cb = _per_token_stats_begin_sample

        def _per_token_stats_cb(token: Any, t_complete: float) -> None:
            _per_token_stats_log_if_sampled(
                mode="sync",
                cfg=cfg,
                token=token,
                completion_time=t_complete,
                ticks_per_ms=ticks_per_ms,
            )

        per_token_stats_cb = _per_token_stats_cb

    net_delay_fn = None
    if cfg.get("bw_aware", True):
        net_delay_fn = _make_bandwidth_aware_delay_fn(
            n_gpu_per_host=cfg["n_gpu_per_host"],
            hidden_dim=cfg["hidden_dim"],
            bytes_per_elem=BYTES_PER_ELEMENT,
            intra_node_bw_gbps=INTRA_NODE_BANDWIDTH_GBPS,
            inter_node_bw_gbps=cfg["inter_node_bw_gbps"],
            ticks_per_ms=sim_sync.TICKS_PER_MILLISECOND,
        )

    result = sim_sync.run_simulation(
        ep_group_size=cfg["ep_group_size"],
        attn_service_t=cfg["attn_service_t"],
        attn_dp_group_size=cfg["ep_group_size"],
        net_t_attn_to_expert=cfg["net_delay_inter_host"],
        net_t_expert_to_attn=cfg["net_delay_inter_host"],
        total_expert_count=MODEL_NUM_EXPERTS,
        num_layers=MODEL_NUM_LAYERS,
        routing_top_k=MODEL_ROUTING_TOPK,
        total_requests=TOTAL_REQUESTS,
        tokens_per_request=TOKENS_PER_REQUEST,
        rng_seed=RNG_SEED,
        max_attention_batch_size=MAX_ATTENTION_BATCH_SIZE,
        expert_cost_scale=EXPERT_COST_SCALE,
        overhead_config=OVERHEAD_CONFIG,
        attention_profile_path=ATTENTION_PROFILE,
        grouped_profile_path=GROUPED_PROFILE,
        initial_context_length=INITIAL_CONTEXT_LENGTH,
        expert_profile_path=_resolve_expert_cost_profile_path(),
        profile_routing_path=_resolve_gating_profile_path(),
        net_delay_fn=net_delay_fn,
        n_gpu_per_host=cfg["n_gpu_per_host"],
        global_request_max_batch_size=cfg["global_request_max_batch_size"],
        tokens_after_reaching_max_bs=tokens_after_reaching_max_bs,
        per_token_begin_cb=per_token_begin_cb,
        per_token_stats_cb=per_token_stats_cb,
    )

    _persist_run_result("sync", cfg, result)

    ticks_per_ms = sim_sync.TICKS_PER_MILLISECOND
    avg_latency_ms = (
        result["avg_latency"] / ticks_per_ms if result["avg_latency"] else 0.0
    )
    makespan_ms = (
        result["makespan"] / ticks_per_ms if result["makespan"] else 0.0
    )

    avg_layer_runtime_ms = result.get("avg_layer_runtime", 0.0) / ticks_per_ms
    avg_layer_wait_imbalance_ms = (
        result.get("avg_layer_wait_imbalance", 0.0) / ticks_per_ms
    )

    avg_per_expert_batch_size = result.get("avg_per_expert_batch_size", 0.0)
    avg_layer_worker_queue_stddev = result.get(
        "avg_layer_worker_queue_stddev", 0.0
    )

    avg_throughput_req_per_sec = result.get("avg_throughput_req_per_sec", 0.0)
    tail_throughput_req_per_sec = result.get(
        "tail_throughput_req_per_sec", float("nan")
    )
    avg_request_latency_ms = result.get("avg_request_latency_ms", 0.0)
    p90_request_latency_ms = result.get("p90_request_latency_ms", 0.0)
    p99_request_latency_ms = result.get("p99_request_latency_ms", 0.0)

    summary: Dict[str, Any] = {
        "mode": "sync",
        "ep_group_size": cfg["ep_group_size"],
        "global_request_max_batch_size": cfg["global_request_max_batch_size"],
        "attn_service_t": cfg["attn_service_t"],
        "n_gpu_per_host": cfg["n_gpu_per_host"],
        "net_delay_intra_host": cfg["net_delay_intra_host"],
        "net_delay_inter_host": cfg["net_delay_inter_host"],
        "hidden_dim": cfg.get("hidden_dim", 0),
        "inter_node_bw_gbps": cfg.get("inter_node_bw_gbps", 0),
        "bw_aware": cfg.get("bw_aware", True),
        "avg_token_latency_ms": avg_latency_ms,
        "makespan_ms": makespan_ms,
        "avg_throughput_req_per_sec": avg_throughput_req_per_sec,
        "tail_throughput_req_per_sec": tail_throughput_req_per_sec,
        "avg_request_latency_ms": avg_request_latency_ms,
        "p90_request_latency_ms": p90_request_latency_ms,
        "p99_request_latency_ms": p99_request_latency_ms,
        "avg_layer_runtime_ms": avg_layer_runtime_ms,
        "avg_layer_wait_imbalance_ms": avg_layer_wait_imbalance_ms,
        "avg_per_expert_batch_size": avg_per_expert_batch_size,
        "avg_layer_worker_queue_stddev": avg_layer_worker_queue_stddev,
    }
    return summary


def _run_tbo_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tokens_after_reaching_max_bs = (
        TOKENS__AFTER_REACHING_MAX_BS if ENABLE_EARLY_TERMINATION_AFTER_MAX_BS else None
    )
    per_token_begin_cb: Callable[[Any], None] | None = None
    per_token_stats_cb: Callable[[Any, float], None] | None = None
    if ENABLE_PER_TOKEN_STATS:
        ticks_per_ms = sim_tbo.TICKS_PER_MILLISECOND
        per_token_begin_cb = _per_token_stats_begin_sample

        def _per_token_stats_cb(token: Any, t_complete: float) -> None:
            _per_token_stats_log_if_sampled(
                mode="tbo",
                cfg=cfg,
                token=token,
                completion_time=t_complete,
                ticks_per_ms=ticks_per_ms,
            )

        per_token_stats_cb = _per_token_stats_cb

    net_delay_fn = None
    if cfg.get("bw_aware", True):
        net_delay_fn = _make_bandwidth_aware_delay_fn(
            n_gpu_per_host=cfg["n_gpu_per_host"],
            hidden_dim=cfg["hidden_dim"],
            bytes_per_elem=BYTES_PER_ELEMENT,
            intra_node_bw_gbps=INTRA_NODE_BANDWIDTH_GBPS,
            inter_node_bw_gbps=cfg["inter_node_bw_gbps"],
            ticks_per_ms=sim_tbo.TICKS_PER_MILLISECOND,
        )

    result = sim_tbo.run_simulation(
        ep_group_size=cfg["ep_group_size"],
        attn_service_t=cfg["attn_service_t"],
        attn_dp_group_size=cfg["ep_group_size"],
        net_t_attn_to_expert=cfg["net_delay_inter_host"],
        net_t_expert_to_attn=cfg["net_delay_inter_host"],
        total_expert_count=MODEL_NUM_EXPERTS,
        num_layers=MODEL_NUM_LAYERS,
        routing_top_k=MODEL_ROUTING_TOPK,
        total_requests=TOTAL_REQUESTS,
        tokens_per_request=TOKENS_PER_REQUEST,
        rng_seed=RNG_SEED,
        max_attention_batch_size=MAX_ATTENTION_BATCH_SIZE,
        expert_cost_scale=EXPERT_COST_SCALE,
        overhead_config=OVERHEAD_CONFIG,
        attention_profile_path=ATTENTION_PROFILE,
        grouped_profile_path=GROUPED_PROFILE,
        initial_context_length=INITIAL_CONTEXT_LENGTH,
        expert_profile_path=_resolve_expert_cost_profile_path(),
        profile_routing_path=_resolve_gating_profile_path(),
        net_delay_fn=net_delay_fn,
        n_gpu_per_host=cfg["n_gpu_per_host"],
        global_request_max_batch_size=cfg["global_request_max_batch_size"],
        tokens_after_reaching_max_bs=tokens_after_reaching_max_bs,
        per_token_begin_cb=per_token_begin_cb,
        per_token_stats_cb=per_token_stats_cb,
        timeline_capture_ops=TBO_TIMELINE_CAPTURE_OPS,
        timeline_out_dir=os.path.join(_results_dir(), "tbo-timeline"),
        timeline_basename=(
            f"tbo_timeline_ep{cfg['ep_group_size']}"
            f"_gbs{cfg['global_request_max_batch_size']}"
            f"_attn{cfg['attn_service_t']}"
            f"_pid{os.getpid()}"
        ),
    )

    _persist_run_result("tbo", cfg, result)

    ticks_per_ms = sim_tbo.TICKS_PER_MILLISECOND
    avg_latency_ms = (
        result["avg_latency"] / ticks_per_ms if result["avg_latency"] else 0.0
    )
    makespan_ms = (
        result["makespan"] / ticks_per_ms if result["makespan"] else 0.0
    )

    avg_layer_runtime_ms = result.get("avg_layer_runtime", 0.0) / ticks_per_ms
    avg_layer_wait_imbalance_ms = (
        result.get("avg_layer_wait_imbalance", 0.0) / ticks_per_ms
    )

    avg_per_expert_batch_size = result.get("avg_per_expert_batch_size", 0.0)
    avg_layer_worker_queue_stddev = result.get(
        "avg_layer_worker_queue_stddev", 0.0
    )

    avg_throughput_req_per_sec = result.get("avg_throughput_req_per_sec", 0.0)
    tail_throughput_req_per_sec = result.get(
        "tail_throughput_req_per_sec", float("nan")
    )
    avg_request_latency_ms = result.get("avg_request_latency_ms", 0.0)
    p90_request_latency_ms = result.get("p90_request_latency_ms", 0.0)
    p99_request_latency_ms = result.get("p99_request_latency_ms", 0.0)

    summary: Dict[str, Any] = {
        "mode": "tbo",
        "ep_group_size": cfg["ep_group_size"],
        "global_request_max_batch_size": cfg["global_request_max_batch_size"],
        "attn_service_t": cfg["attn_service_t"],
        "n_gpu_per_host": cfg["n_gpu_per_host"],
        "net_delay_intra_host": cfg["net_delay_intra_host"],
        "net_delay_inter_host": cfg["net_delay_inter_host"],
        "hidden_dim": cfg.get("hidden_dim", 0),
        "inter_node_bw_gbps": cfg.get("inter_node_bw_gbps", 0),
        "bw_aware": cfg.get("bw_aware", True),
        "avg_token_latency_ms": avg_latency_ms,
        "makespan_ms": makespan_ms,
        "avg_throughput_req_per_sec": avg_throughput_req_per_sec,
        "tail_throughput_req_per_sec": tail_throughput_req_per_sec,
        "avg_request_latency_ms": avg_request_latency_ms,
        "p90_request_latency_ms": p90_request_latency_ms,
        "p99_request_latency_ms": p99_request_latency_ms,
        "avg_layer_runtime_ms": avg_layer_runtime_ms,
        "avg_layer_wait_imbalance_ms": avg_layer_wait_imbalance_ms,
        "avg_per_expert_batch_size": avg_per_expert_batch_size,
        "avg_layer_worker_queue_stddev": avg_layer_worker_queue_stddev,
    }
    return summary


def _write_results(mode: str, results: List[Dict[str, Any]]) -> None:
    results_dir = _results_dir()
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"experiment_results_{mode}.txt")

    header = (
        "mode,ep_group_size,global_request_max_batch_size,"
        "attn_service_t,n_gpu_per_host,net_delay_intra_host,net_delay_inter_host,"
        "hidden_dim,inter_node_bw_gbps,bw_aware,"
        "avg_token_latency_ms,makespan_ms,avg_throughput_req_per_sec,tail_throughput_req_per_sec,"
        "avg_request_latency_ms,p90_request_latency_ms,p99_request_latency_ms,"
        "avg_layer_runtime_ms,avg_layer_wait_imbalance_ms,"
        "avg_per_expert_batch_size,avg_layer_worker_queue_stddev\n"
    )

    # Sort results for reproducible ordering.
    results_sorted = sorted(
        results,
        key=lambda r: (
            r["ep_group_size"],
            r["global_request_max_batch_size"],
            r["attn_service_t"],
            r["n_gpu_per_host"],
            r["net_delay_intra_host"],
            r["net_delay_inter_host"],
            r.get("hidden_dim", 0),
            r.get("inter_node_bw_gbps", 0),
            0 if r.get("bw_aware", True) else 1,
        ),
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        for r in results_sorted:
            line = (
                f"{r['mode']},"
                f"{r['ep_group_size']},"
                f"{r['global_request_max_batch_size']},"
                f"{r['attn_service_t']},"
                f"{r['n_gpu_per_host']},"
                f"{r['net_delay_intra_host']},"
                f"{r['net_delay_inter_host']},"
                f"{r.get('hidden_dim', '')},"
                f"{r.get('inter_node_bw_gbps', '')},"
                f"{r.get('bw_aware', True)},"
                f"{r['avg_token_latency_ms']:.6f},"
                f"{r['makespan_ms']:.6f},"
                f"{r['avg_throughput_req_per_sec']:.6f},"
                f"{r.get('tail_throughput_req_per_sec', float('nan')):.6f},"
                f"{r['avg_request_latency_ms']:.6f},"
                f"{r['p90_request_latency_ms']:.6f},"
                f"{r['p99_request_latency_ms']:.6f},"
                f"{r['avg_layer_runtime_ms']:.6f},"
                f"{r['avg_layer_wait_imbalance_ms']:.6f},"
                f"{r['avg_per_expert_batch_size']:.6f},"
                f"{r['avg_layer_worker_queue_stddev']:.6f}\n"
            )
            f.write(line)

    print(f"Wrote results for mode='{mode}' to {out_path}; full metrics under {_results_dir()}/run-details")


def main(argv: List[str]) -> None:
    if len(argv) != 2 or argv[1] not in ("sync", "async", "tbo"):
        print("Usage: python experiment.py [sync|async|tbo]")
        raise SystemExit(1)

    mode = argv[1]

    # Print resolved macros once per invocation.
    profile_path = _resolve_expert_cost_profile_path()
    gating_profile_path = _resolve_gating_profile_path()
    print(
        "Experiment macros: "
        f"MODEL_NUM_EXPERTS={MODEL_NUM_EXPERTS} "
        f"MODEL_NUM_LAYERS={MODEL_NUM_LAYERS} "
        f"MODEL_ROUTING_TOPK={MODEL_ROUTING_TOPK} "
        f"ENBALE_BANDWIDTH_UNAWARE_COMPARISON={ENBALE_BANDWIDTH_UNAWARE_COMPARISON} "
        f"MODEL_HIDDEN_DIMS={HIDDEN_DIM_VALUES} "
        f"EXPERT_COST_PROFILE={profile_path} "
        f"GATING_PROFILE={gating_profile_path} "
        f"OVERHEAD_CONFIG={OVERHEAD_CONFIG}"
    )
    configs = _build_configs()
    if MAX_WORKERS < 1:
        raise ValueError("MAX_WORKERS must be >= 1")
    num_workers = min(MAX_WORKERS, len(configs))

    if mode == "async":
        worker_fn = _run_async_config
    elif mode == "sync":
        worker_fn = _run_sync_config
    else:
        worker_fn = _run_tbo_config

    results_dir = _results_dir()
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"experiment_results_{mode}.txt")
    debug_path = os.path.join(results_dir, DEFRAG_V0_DEBUG_LOG_BASENAME)

    header = (
        "mode,ep_group_size,global_request_max_batch_size,"
        "attn_service_t,n_gpu_per_host,net_delay_intra_host,net_delay_inter_host,"
        "hidden_dim,inter_node_bw_gbps,bw_aware,"
        "avg_token_latency_ms,makespan_ms,avg_throughput_req_per_sec,tail_throughput_req_per_sec,"
        "avg_request_latency_ms,p90_request_latency_ms,p99_request_latency_ms,"
        "avg_layer_runtime_ms,avg_layer_wait_imbalance_ms,"
        "avg_per_expert_batch_size,avg_layer_worker_queue_stddev\n"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.flush()

        debug_f = None
        try:
            if mode == "async" and ENABLE_DEFRAG_V0_DEBUG_LOG:
                debug_f = open(debug_path, "w", encoding="utf-8")
                debug_f.flush()

            if ENABLE_PER_TOKEN_STATS:
                os.makedirs(_per_token_stats_dir(), exist_ok=True)
                prefix = f"per_token_stats_{mode}_"
                for name in os.listdir(_per_token_stats_dir()):
                    if name.startswith(prefix):
                        try:
                            os.remove(os.path.join(_per_token_stats_dir(), name))
                        except FileNotFoundError:
                            pass

            pool_kwargs: Dict[str, Any] = {"processes": num_workers}
            if ENABLE_PER_TOKEN_STATS:
                pool_kwargs["initializer"] = _init_worker_per_token_stats
                pool_kwargs["initargs"] = (mode, True, TOKEN_SAMPLING_RATE)

            with mp.Pool(**pool_kwargs) as pool:
                for summary in pool.imap_unordered(worker_fn, configs):
                    line = (
                        f"{summary['mode']},"
                        f"{summary['ep_group_size']},"
                        f"{summary['global_request_max_batch_size']},"
                        f"{summary['attn_service_t']},"
                        f"{summary['n_gpu_per_host']},"
                        f"{summary['net_delay_intra_host']},"
                        f"{summary['net_delay_inter_host']},"
                        f"{summary.get('hidden_dim', '')},"
                        f"{summary.get('inter_node_bw_gbps', '')},"
                        f"{summary.get('bw_aware', True)},"
                        f"{summary['avg_token_latency_ms']:.6f},"
                        f"{summary['makespan_ms']:.6f},"
                        f"{summary['avg_throughput_req_per_sec']:.6f},"
                        f"{summary.get('tail_throughput_req_per_sec', float('nan')):.6f},"
                        f"{summary['avg_request_latency_ms']:.6f},"
                        f"{summary['p90_request_latency_ms']:.6f},"
                        f"{summary['p99_request_latency_ms']:.6f},"
                        f"{summary['avg_layer_runtime_ms']:.6f},"
                        f"{summary['avg_layer_wait_imbalance_ms']:.6f},"
                        f"{summary['avg_per_expert_batch_size']:.6f},"
                        f"{summary['avg_layer_worker_queue_stddev']:.6f}\n"
                    )
                    f.write(line)
                    f.flush()

                    if debug_f is not None:
                        dbg = summary.get("defrag_v0_debug_text")
                        if dbg:
                            debug_f.write(dbg + "\n")
                            debug_f.flush()
        finally:
            if debug_f is not None:
                debug_f.close()

    print(f"Wrote results for mode='{mode}' to {out_path}; full metrics under {_results_dir()}/run-details")
    if mode == "async" and ENABLE_DEFRAG_V0_DEBUG_LOG:
        print(f"Wrote defrag-v0 debug log (text) to {debug_path}")
    if ENABLE_PER_TOKEN_STATS:
        print(f"Wrote per-token sampled ITL logs under {_per_token_stats_dir()}")


if __name__ == "__main__":
    main(sys.argv)
