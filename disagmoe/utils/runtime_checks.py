"""Capacity checks that do not require importing CUDA or the native extension."""

import math


def kv_cache_block_budget(total_bytes, free_bytes, utilization, block_bytes):
    if not math.isfinite(utilization) or not 0 < utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if block_bytes <= 0 or total_bytes <= 0 or not 0 <= free_bytes <= total_bytes:
        raise ValueError("Invalid GPU memory or KV block size")
    # mem_get_info includes colocated expert weights, NCCL and other processes.
    resident_bytes = total_bytes - free_bytes
    budget = int(total_bytes * utilization - resident_bytes)
    blocks = budget // block_bytes
    if blocks < 1:
        gib = 1024 ** 3
        raise RuntimeError(
            f"No room for KV cache: {resident_bytes / gib:.2f} GiB already resident, "
            f"budget {total_bytes * utilization / gib:.2f} GiB "
            f"({utilization:.0%} of GPU memory). Free other GPU workloads, reduce "
            "--num-layers for a smoke test, or adjust --gpu-usage with runtime headroom."
        )
    return blocks


def validate_request_lengths(input_len, output_len, max_seq_len, block_size,
                             max_cache_blocks=None):
    if input_len < 1 or output_len < 1:
        raise ValueError("input_len and output_len must both be positive")
    if input_len + output_len > max_seq_len:
        raise ValueError(
            f"Request context {input_len + output_len} exceeds max_seq_len={max_seq_len}"
        )
    # Match DPScheduler.required_blocks(), which reserves each phase separately.
    needed = ((input_len + block_size - 1) // block_size
              + (output_len + block_size - 1) // block_size)
    if max_cache_blocks is not None and needed > max_cache_blocks:
        raise ValueError(
            f"Request needs {needed} KV blocks, but no attention rank has more than "
            f"{max_cache_blocks}; reduce request lengths or increase KV capacity."
        )
