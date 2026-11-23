"""Address a shared synthetic decode route by request, decode step, and layer."""

import json
import os
from pathlib import Path

import numpy as np

WARMUP_REQUESTS = 10
WARMUP_PROMPT_LEN = 100


class SharedTop8Replay:
    def __init__(self, path, prompt_len, audit_dir=None):
        routes = np.load(path, mmap_mode="r", allow_pickle=False)
        if (routes.ndim != 4 or routes.shape[-1] != 8 or routes.dtype != np.uint8
                or min(routes.shape) < 1):
            raise ValueError("Shared Top-8 requires uint8 [request, step, layer, 8]")
        self.routes = routes
        self.requests, self.steps, self.layers, _ = routes.shape
        self.prompt_len = prompt_len
        self.audit_file = None
        if audit_dir:
            output = Path(audit_dir)
            output.mkdir(parents=True, exist_ok=True)
            self.audit_file = (output / f"streaminfer-{os.getpid()}.jsonl").open("x", buffering=1)

    def layer_table(self, layer_id, device):
        import torch

        # Keep each layer contiguous, so replay needs just one row gather.
        rows = self.routes[:, :, layer_id, :].reshape(-1, 8).copy()
        return torch.from_numpy(rows).to(device)

    def row_indices(self, request_ids, seq_lens):
        rows = []
        for rid, seq_len in zip(request_ids, seq_lens, strict=True):
            if 1 <= rid <= WARMUP_REQUESTS:
                # The ten warmups have their own 100-token prefixes. Small smoke
                # tables can be shorter than the warmup workload.
                request = (rid - 1) % self.requests
                step = (seq_len - WARMUP_PROMPT_LEN - 1) % self.steps
            else:
                request = rid - WARMUP_REQUESTS - 1
                step = seq_len - self.prompt_len - 1
                if not (0 <= request < self.requests and 0 <= step < self.steps):
                    raise ValueError(f"Shared route outside trace: req_id={rid}, seq_len={seq_len}, "
                                     f"request={request}, step={step}")
            rows.append(request * self.steps + step)
        return rows

    def audit_dispatch(self, layer, request_ids, seq_lens, expert_ids):
        # The native dispatcher already needs expert_ids on the host. Inspect
        # its actual rows here, without an additional device synchronization.
        for i, (rid, seq_len) in enumerate(zip(request_ids, seq_lens, strict=True)):
            request = rid - WARMUP_REQUESTS - 1
            step = seq_len - self.prompt_len - 1
            if 0 <= request < 4 and 0 <= step < 4:
                selected = expert_ids[i * 8:(i + 1) * 8]
                self.audit_file.write(json.dumps(dict(request=request, step=step,
                                                      layer=layer, experts=selected)) + "\n")
