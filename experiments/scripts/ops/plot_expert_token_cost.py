#!/usr/bin/env python3
"""Plot expert forward time per token from an existing batch latency CSV."""

import argparse
import csv
import math
from pathlib import Path

from sweep_expert_batch import plot_curve


def read_token_costs(path):
    costs = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        time_column = "time_ms" if "time_ms" in fields else "avg_time_ms"
        if "batch_size" not in fields or time_column not in fields:
            raise ValueError("CSV requires batch_size and time_ms (or avg_time_ms).")
        for row in reader:
            try:
                batch = int(row["batch_size"])
                time_ms = float(row[time_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid batch size or latency on CSV line {reader.line_num}.") from exc
            if batch <= 0 or time_ms <= 0 or not math.isfinite(time_ms):
                raise ValueError(f"Batch size and latency must be positive and finite on CSV line {reader.line_num}.")
            if batch in costs:
                raise ValueError(f"Duplicate batch_size {batch} on CSV line {reader.line_num}.")
            costs[batch] = time_ms / batch * 1000
    if not costs:
        raise ValueError("CSV contains no measurements.")
    return sorted(costs.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Expert latency CSV from the batch sweep.")
    parser.add_argument("--output", type=Path, help="Image path (defaults to expert_token_cost.png beside the CSV).")
    args = parser.parse_args()
    output = args.output or args.csv_path.with_name("expert_token_cost.png")
    try:
        costs = read_token_costs(args.csv_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        plot_curve(costs, "Single-expert forward cost per token", output,
                   ylabel="Forward time per token (µs/token)")
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(output)


if __name__ == "__main__":
    main()
