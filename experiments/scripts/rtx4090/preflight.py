#!/usr/bin/env python3
"""Server checks. This file must run in the built Python 3.12 environment."""
import importlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    assert sys.version_info[:2] == (3, 12), "Expected Python 3.12"
    assert sys.platform == "linux", "This deployment targets Linux"
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    assert cpus >= 16, f"Need at least 16 available logical CPUs (12 workers + 4 tokenizer/detokenizer), got {cpus}"
    shm = shutil.disk_usage("/dev/shm")
    print(f"/dev/shm total={shm.total / 2**30:.2f} GiB free={shm.free / 2**30:.2f} GiB")
    assert shm.free >= 2**30, "Need at least 1 GiB free /dev/shm; for Docker use --shm-size=2g or larger"
    assert os.environ.get("NCCL_P2P_DISABLE") == "1", "This deployment uses the measured no-P2P topology"
    assert os.environ.get("NCCL_IB_DISABLE") == "1"
    assert os.environ.get("NCCL_SHM_DISABLE") != "1", "NCCL SHM must remain enabled"
    import torch
    assert torch.cuda.is_available(), "CUDA unavailable"
    assert torch.cuda.device_count() == 4, "Expose exactly four GPUs with CUDA_VISIBLE_DEVICES"
    assert torch.__version__.split("+")[0] == "2.6.0", torch.__version__
    assert importlib.metadata.version("vllm") == "0.8.2"
    print(f"Python={sys.version.split()[0]} torch={torch.__version__} CUDA={torch.version.cuda} NCCL={torch.cuda.nccl.version()}")
    min_free = float(os.environ.get("MIN_FREE_GIB", "20"))
    for gpu in range(4):
        with torch.cuda.device(gpu):
            props = torch.cuda.get_device_properties(gpu)
            free, total = torch.cuda.mem_get_info(gpu)
            print(f"GPU {gpu}: {props.name}, sm{props.major}{props.minor}, total={total / 2**30:.2f} GiB free={free / 2**30:.2f} GiB")
            assert "4090" in props.name, f"Expected RTX 4090, got {props.name}"
            assert (props.major, props.minor) == (8, 9)
            assert total >= 22 * 2**30, "Expected a 24 GB class card"
            assert free >= min_free * 2**30, f"GPU {gpu} needs at least {min_free} GiB free for the initial full-depth configuration"
            x = torch.ones((128, 128), dtype=torch.bfloat16, device=f"cuda:{gpu}")
            y = x @ x
            torch.cuda.synchronize(gpu)
            assert torch.equal(y, torch.full_like(y, 128)), "BF16 matmul value check failed"
            del x, y
            torch.cuda.empty_cache()
    print("CUDA peer-access matrix (capability only; NCCL value test follows separately):")
    for a in range(4):
        print(a, [None if a == b else torch.cuda.can_device_access_peer(a, b) for b in range(4)])
    for name in ("vllm", "vllm._C", "triton", "ray", "zmq", "disagmoe_c", "disagmoe.frontend.engine", "disagmoe.models.experts", "disagmoe.ops.grouped_gemm"):
        importlib.import_module(name)
        print("IMPORT OK", name)
    from vllm.attention import Attention
    assert "use_direct_call" in inspect.signature(Attention.__init__).parameters, "vLLM patch missing"
    assert "kv_cache" in inspect.signature(Attention.forward).parameters, "vLLM patch missing"
    import disagmoe_c as c
    assert not c.gdrcopy_compiled(), "Rebuild with ENABLE_GDRCOPY=0"
    assert not c.gdrcopy_enabled(), "Run with DMOE_USE_GDRCOPY=0"
    linked = subprocess.run(["ldd", str(Path(c.__file__).resolve())], check=True, capture_output=True, text=True).stdout
    print(linked)
    assert "not found" not in linked, "A linked library is missing"
    assert "libgdrapi" not in linked, "Extension still links GDRCopy"
    assert "libnixl" not in linked, "Extension still links NIXL; rebuild with ENABLE_NIXL=0"
    print("Hardware/import/BF16 preflight passed. No end-to-end serving claim is made by this check.")


if __name__ == "__main__":
    main()
