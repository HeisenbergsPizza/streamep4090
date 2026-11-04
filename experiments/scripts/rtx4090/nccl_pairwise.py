#!/usr/bin/env python3
"""Run with torchrun: six two-rank NCCL communicators, BF16 values both ways."""
from datetime import timedelta
from itertools import combinations
import os

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 4
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=45))
    try:
        probe = torch.tensor([rank], device="cuda", dtype=torch.float32)
        dist.all_reduce(probe)
        assert probe.item() == 6
        for a, b in combinations(range(4), 2):
            group = dist.new_group([a, b], backend="nccl", timeout=timedelta(seconds=45))
            if rank in (a, b):
                peer = b if rank == a else a
                for size in (1, 4096, 1048576):
                    # Values <= 156 are exactly representable in BF16.
                    pattern = torch.arange(size, device="cuda", dtype=torch.int32) % 127
                    source = (pattern + rank * 10).to(torch.bfloat16)
                    received = torch.empty_like(source)
                    ops = [dist.P2POp(dist.isend, source, peer, group), dist.P2POp(dist.irecv, received, peer, group)]
                    for work in dist.batch_isend_irecv(ops):
                        work.wait()
                    torch.cuda.synchronize()
                    expected = (pattern + peer * 10).to(torch.bfloat16)
                    assert torch.equal(received, expected), f"NCCL mismatch {peer}->{rank}, elements={size}"
                if rank == a:
                    print(f"NCCL pair {a}<->{b}: three BF16 payload sizes verified", flush=True)
            dist.barrier()
        if rank == 0:
            print("PASS: all six GPU pairs exchanged correct BF16 values. This is a torch NCCL check, not the full StreamInfer runtime.", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
