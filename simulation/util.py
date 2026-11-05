import os
import csv
import math
from typing import Dict, List, Optional, Sequence

if __package__:
    from .router import ProfileDrivenRouter
else:  # direct invocation from simulation/
    from router import ProfileDrivenRouter


def expert_compute_time_lookup_table_from_profile(
    csv_path: str,
    max_batch_size: int,
    size_header_candidates: Optional[Sequence[str]] = None,
    time_header_candidates: Optional[Sequence[str]] = None,
    ticks_per_millisecond: float = 1.0,
) -> Dict[int, float]:
    """
    Load an expert compute time profile CSV and return a lookup table mapping
    batch_size -> compute_time (ticks). Validates that all sizes 1..max_batch_size
    are present to avoid runtime fallbacks.

    Args:
        csv_path: Absolute or relative path to the CSV file.
        max_batch_size: Maximum batch size the simulator may form. The CSV must
                        include entries for all 1..max_batch_size inclusive.
        size_header_candidates: Optional override list of header names for batch size.
        time_header_candidates: Optional override list of header names for time.
        ticks_per_millisecond: Conversion factor to map profile milliseconds into simulator ticks.

    Returns:
        Dict[int, float]: Mapping from batch size to compute time.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required headers are missing or coverage is incomplete.
    """
    if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
        raise ValueError("max_batch_size must be a positive integer")
    if not math.isfinite(ticks_per_millisecond) or ticks_per_millisecond <= 0:
        raise ValueError("ticks_per_millisecond must be finite and positive")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Expert compute profile not found: {csv_path}")

    size_headers = list(size_header_candidates) if size_header_candidates else [
        "batch_size", "batch", "size",
    ]
    time_headers = list(time_header_candidates) if time_header_candidates else [
        "avg_time_ms", "time_ms", "avg_time", "time",
    ]

    lookup: Dict[int, float] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        reader.fieldnames = headers
        if len(set(headers)) != len(headers):
            raise ValueError(f"Duplicate columns in expert profile {csv_path}")

        size_key: Optional[str] = next((h for h in size_headers if h in headers), None)
        time_key: Optional[str] = next((h for h in time_headers if h in headers), None)
        if size_key is None or time_key is None:
            raise ValueError(
                f"CSV must include batch size and time columns. "
                f"Found headers: {headers}; expected any of "
                f"size in {size_headers}, time in {time_headers}"
            )

        for line_number, row in enumerate(reader, 2):
            try:
                bs = int(row[size_key])
                t = float(row[time_key]) * ticks_per_millisecond
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Malformed expert profile row {line_number} in {csv_path}") from exc
            if bs < 1 or not math.isfinite(t) or t <= 0:
                raise ValueError(f"Expert profile row {line_number} requires positive batch size and finite positive time")
            if bs in lookup:
                raise ValueError(f"Duplicate batch size {bs} in expert profile {csv_path}")
            lookup[bs] = t

    # Validate coverage for 1..max_batch_size inclusive
    missing: List[int] = [s for s in range(1, max_batch_size + 1) if s not in lookup]
    if missing:
        sample = ", ".join(map(str, missing[:10]))
        more = "" if len(missing) <= 10 else f" (+{len(missing)-10} more)"
        raise ValueError(
            f"Expert profile {csv_path} missing batch sizes up to {max_batch_size}. "
            f"Missing: {sample}{more}"
        )

    return lookup


def build_profile_router(profile_path: str,
                         num_experts: int,
                         top_k: int) -> ProfileDrivenRouter:
    """
    Load a profile-driven router from disk. Raises if the profile cannot
    be read or parsed so simulation failures are loud and early.
    """
    if not profile_path:
        raise ValueError("profile_path must be provided for profile-driven routing.")

    abs_path = os.path.abspath(profile_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Profile file {abs_path} not found.")

    try:
        with open(abs_path, "rb") as fh:
            profile_bytes = fh.read()
    except OSError as exc:  # pragma: no cover - simple file IO guard
        raise RuntimeError(f"Failed to read profile file {abs_path}: {exc}") from exc

    return ProfileDrivenRouter(
        profile_bytes=profile_bytes,
        num_experts_expected=num_experts,
        top_k=top_k,
    )


__all__ = [
    "expert_compute_time_lookup_table_from_profile",
    "build_profile_router",
]
