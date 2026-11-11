"""Single-process adapter for SGLang 0.5.9's unquantized Triton expert path.

The complete upstream forward is executed for every call. ``stage`` only selects
which launches receive the ExpertBench NVTX range when ``profile=True``; timing
the returned callback with CUDA events measures the complete forward.

No model, HTTP server, distributed group, or copied GPU kernel is needed. The
temporary hooks are intended for this standalone, single-threaded benchmark.
"""

from __future__ import annotations

import importlib
from importlib.metadata import version
from typing import Any, Callable


def build_sglang_runner(
    case: dict[str, Any], stage: str = "full"
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Return ``(run, metadata)`` for the local weighted expert contribution.

    Input layouts are x[T,H], w13[E,H,2I], w2[E,I,H], topk_ids[T,K], and
    topk_weights[T,K]. IDs refer to local experts 0..E-1; -1 means a remote
    expert. ``run(profile=False)`` returns the combined local result [T,H].
    All input tensors must already be on the same CUDA device.
    """
    if stage not in {"full", "up", "down", "activation"}:
        raise ValueError(f"Unsupported SGLang profiling stage: {stage}")

    installed_version = version("sglang")
    # Kernel signatures and the patched module symbols below are versioned.
    # Accept a local wheel suffix, but do not silently accept another release.
    if installed_version.split("+", 1)[0] != "0.5.9":
        raise RuntimeError(
            "This adapter supports sglang==0.5.9; installed version is "
            f"{installed_version}. Use the existing SGLang experiment environment."
        )

    import torch
    from sglang.srt.server_args import (
        ServerArgs,
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )

    moe = importlib.import_module(
        "sglang.srt.layers.moe.fused_moe_triton.fused_moe"
    )
    x = case["x"]
    canonical_w13 = case["w13"]
    canonical_w2 = case["w2"]
    ids = case["topk_ids"]
    weights = case["topk_weights"]
    tensors = (x, canonical_w13, canonical_w2, ids, weights)
    if x.device.type != "cuda" or any(t.device != x.device for t in tensors):
        raise ValueError("All SGLang adapter inputs must be on the same CUDA device")
    if any(t.dtype != torch.bfloat16 for t in tensors[:3]):
        raise ValueError("The expert comparison uses BF16 inputs and weights")
    if x.ndim != 2 or canonical_w13.ndim != 3 or canonical_w2.ndim != 3:
        raise ValueError("Expected x[T,H], w13[E,H,2I], and w2[E,I,H]")
    local_experts, hidden, twice_intermediate = canonical_w13.shape
    if twice_intermediate % 2 or tuple(canonical_w2.shape) != (
        local_experts, twice_intermediate // 2, hidden
    ):
        raise ValueError("Expert projection shapes do not match gated SiLU")
    if x.shape[1] != hidden or not 0 < x.shape[0] < 65536:
        raise ValueError("Expected matching hidden size and 1..65535 input tokens")
    if ids.ndim != 2 or ids.shape != weights.shape or ids.shape[0] != x.shape[0]:
        raise ValueError("Expected matching topk_ids/topk_weights of shape [T,K]")
    if ids.shape[1] < 1 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must contain at least one integral expert ID per token")
    if bool(((ids < -1) | (ids >= local_experts)).any().item()):
        raise ValueError("topk_ids must contain local IDs 0..E-1 or -1 for remote experts")

    # Transposition is a setup cost, outside warmup, timing, and NCU capture.
    w13 = canonical_w13.transpose(1, 2).contiguous()
    w2 = canonical_w2.transpose(1, 2).contiguous()
    x = x.contiguous()
    ids = ids.to(dtype=torch.int32).contiguous()
    weights = weights.to(dtype=torch.float32).contiguous()
    # "dummy" skips model-specific setup/downloads in ServerArgs.__post_init__.
    standalone_args = ServerArgs(
        model_path="dummy", device="cuda", dtype="bfloat16",
        moe_runner_backend="triton", moe_a2a_backend="none",
        enable_deterministic_inference=False,
    )
    metadata: dict[str, Any] = {
        "sglang_version": installed_version,
        "entrypoint": "outplace_fused_experts",
        "source_module": str(moe.__file__),
        "filter_expert": True,
        "local_experts": local_experts,
        "profile_stage": stage,
        "profile_range": f"ExpertBench.{stage}",
        "callback_timing_scope": "complete_local_expert_forward",
        "input_shape": list(x.shape),
        "w13_shape": list(w13.shape),
        "w2_shape": list(w2.shape),
        "topk_shape": list(ids.shape),
        "kernel_configs": {},
        "note": "No NCCL; output contains this EP rank's weighted contribution only.",
    }
    case_config = case.get("config", case)
    for key in ("global_experts", "ep_size", "local_rank"):
        if key in case_config:
            metadata[key] = case_config[key]

    running = False

    def run(*, profile: bool = False):
        nonlocal running
        if running:
            raise RuntimeError("The standalone SGLang runner cannot be called recursively")
        running = True
        original_invoke = moe.invoke_fused_moe_kernel
        original_activation = moe.act_and_mul_triton
        try:
            previous_args = get_global_server_args()
        except ValueError:
            previous_args = None
        seen = {"up": 0, "down": 0, "activation": 0}

        def marked(name: str, function: Callable[..., Any], *args, **kwargs):
            seen[name] += 1
            selected = profile and stage == name
            if selected:
                torch.cuda.nvtx.range_push(f"ExpertBench.{name}")
            try:
                return function(*args, **kwargs)
            finally:
                if selected:
                    torch.cuda.nvtx.range_pop()

        def invoke(*args, **kwargs):
            # The upstream 0.5.9 function receives A, B, bias, C, ..., config.
            # Custom-op dispatch may rewrap a Tensor's Python object, so compare
            # storage pointers rather than requiring Python object identity.
            if args[1].data_ptr() == w13.data_ptr():
                name = "up"
            elif args[1].data_ptr() == w2.data_ptr():
                name = "down"
            else:
                raise RuntimeError("Unexpected expert weight in SGLang kernel dispatch")
            metadata["kernel_configs"][name] = dict(args[14])
            return marked(name, original_invoke, *args, **kwargs)

        def activation(*args, **kwargs):
            return marked("activation", original_activation, *args, **kwargs)

        full_range = False
        try:
            set_global_server_args_for_scheduler(standalone_args)
            moe.invoke_fused_moe_kernel = invoke
            moe.act_and_mul_triton = activation
            if profile and stage == "full":
                torch.cuda.nvtx.range_push("ExpertBench.full")
                full_range = True
            with torch.inference_mode():
                output = moe.outplace_fused_experts(
                    x, w13, w2, weights, ids,
                    activation="silu", is_gated=True,
                    apply_router_weight_on_input=False,
                    no_combine=False, filter_expert=True,
                )
            if seen != {"up": 1, "down": 1, "activation": 1}:
                raise RuntimeError(
                    f"SGLang did not execute the expected BF16 EP path: {seen}"
                )
            return output
        finally:
            if full_range:
                torch.cuda.nvtx.range_pop()
            moe.invoke_fused_moe_kernel = original_invoke
            moe.act_and_mul_triton = original_activation
            set_global_server_args_for_scheduler(previous_args)
            running = False

    return run, metadata
