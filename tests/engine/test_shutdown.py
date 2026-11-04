"""Native idle-shutdown regression; run with the CUDA/NCCL extension built.

The subprocess deadline makes a stuck C++ join a test failure instead of
blocking the pytest runner. No model weights or multi-GPU cluster are needed.
"""

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device and native extension required")
def test_idle_pool_and_dispatcher_stop_without_another_message():
    program = textwrap.dedent("""
        import os
        import time
        import torch
        import disagmoe_c as native

        torch.cuda.set_device(0)
        native.select_transport("zmq")
        native.set_hosts(os.getpid(), {0: "127.0.0.1"})
        cfg = native.ParallelConfig()
        # An inbound local channel keeps the pool listening for metadata.
        # The dispatcher has no queued work and therefore waits on its CV.
        pool, scheduler, dispatcher = native.init_unified_engine(
            1, 0, 0, 8, True, True, False, cfg, [0], [0], [], {}, {}, [],
            "flfs", 0.8, 4, 0,
        )
        native.start_engine(scheduler, dispatcher)
        time.sleep(0.25)
        start = time.monotonic()
        dispatcher.terminate()
        pool.terminate()
        assert time.monotonic() - start < 5.0
        # A second stop must also be safe after the native thread was joined.
        dispatcher.terminate()
        pool.terminate()
        print("idle shutdown complete", flush=True)
    """)
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "idle shutdown complete" in result.stdout
