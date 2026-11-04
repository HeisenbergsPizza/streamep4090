"""CUDA tests for the GDRCopy-free host metadata path.

Run on the target server after building the extension:
    DMOE_USE_GDRCOPY=0 python -m pytest -q tests/ops/test_host_staging.py

These tests also exercise runtime disabling with a GDR-enabled extension.
They skip when CUDA or the compiled extension is unavailable.
"""

import ctypes
import gc

import pytest

torch = pytest.importorskip("torch")
native = pytest.importorskip("disagmoe_c")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def cuda_backend(monkeypatch):
    monkeypatch.setenv("DMOE_USE_GDRCOPY", "0")


def test_runtime_backend_selection(monkeypatch):
    assert isinstance(native.gdrcopy_compiled(), bool)
    assert native.gdrcopy_enabled() is False
    tensor = torch.empty(4, dtype=torch.int32, device="cuda")
    context = native.GdrContext(tensor)
    assert context.backend() == "cuda"
    # A runtime request cannot enable a dependency omitted from the build.
    monkeypatch.setenv("DMOE_USE_GDRCOPY", "1")
    assert native.gdrcopy_enabled() == native.gdrcopy_compiled()
    # Existing objects retain their backend selection.
    assert context.backend() == "cuda"


@pytest.mark.parametrize(
    "dtype,suffix,values",
    [
        (torch.int32, "int32", [0, -123, 2**30, 17]),
        (torch.int64, "int64", [0, -(2**40), 2**48, 17]),
        (torch.float32, "float", [0.0, -1.25, 0.5, 128.0]),
    ],
)
def test_vectors_on_nondefault_stream(dtype, suffix, values):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tensor = torch.zeros(len(values), dtype=dtype, device="cuda")
        context = native.GdrContext(tensor)
        # Previous writes, H2D, a subsequent kernel and D2H must keep this order.
        tensor.fill_(99)
        getattr(context, f"copy_from_host_{suffix}")(list(values))
        tensor.add_(1)
        result = getattr(context, f"copy_to_host_{suffix}")(len(values))
    # No caller-side synchronization is needed before reading returned values.
    assert result == [value + 1 for value in values]


@pytest.mark.parametrize("pinned", [False, True])
def test_host_tensor_lifetime_and_completion(pinned):
    stream = torch.cuda.Stream()
    src = torch.arange(4096, dtype=torch.int64, pin_memory=pinned, device="cpu")
    expected = src.clone()
    dst = torch.empty_like(src, pin_memory=pinned)
    with torch.cuda.stream(stream):
        tensor = torch.empty(src.numel(), dtype=src.dtype, device="cuda")
        context = native.GdrContext(tensor)
        context.copy_from_host_tensor(src)
        src.fill_(-1)  # Reuse host storage immediately after H2D returns.
        del src
        gc.collect()
        tensor.mul_(3)
        context.copy_to_host_tensor(dst)
    assert torch.equal(dst, expected * 3)


def test_raw_pointer_offsets_and_byte_fill():
    tensor = torch.empty(8, dtype=torch.int32, device="cuda")
    context = native.GdrContext(tensor)
    context.fill(0, tensor.numel() * tensor.element_size())
    src = (ctypes.c_int32 * 3)(7, -8, 9)
    context.copy_from_host(ctypes.addressof(src), ctypes.sizeof(src), 2 * 4)
    src[:] = [100, 100, 100]
    assert context.copy_to_host_int32(8) == [0, 0, 7, -8, 9, 0, 0, 0]
    dst = (ctypes.c_int32 * 3)()
    context.copy_to_host(ctypes.addressof(dst), ctypes.sizeof(dst), 2 * 4)
    assert list(dst) == [7, -8, 9]
    context.fill(0xFF, 4, 7 * 4)
    assert context.copy_to_host_int32(8)[-1] == -1


def test_empty_copies_and_bounds():
    empty = native.GdrContext(torch.empty(0, dtype=torch.int32, device="cuda"))
    empty.copy_from_host_int32([])
    empty.copy_from_host(0, 0)
    empty.copy_to_host(0, 0)
    empty.fill(0, 0)
    assert empty.copy_to_host_int32(0) == []

    context = native.GdrContext(torch.empty(4, dtype=torch.int32, device="cuda"))
    with pytest.raises(RuntimeError, match="bounds"):
        context.copy_from_host_int32([1] * 5)
    with pytest.raises(RuntimeError, match="bounds"):
        context.copy_to_host_int32(5)
    with pytest.raises(RuntimeError, match="nonnegative"):
        context.copy_to_host_int32(-1)
    with pytest.raises(RuntimeError, match="bounds"):
        context.fill(0, 1, 16)
    with pytest.raises(RuntimeError, match="null source"):
        context.copy_from_host(0, 4)
    with pytest.raises(RuntimeError, match="too small"):
        context.copy_from_host_tensor(torch.empty(1, dtype=torch.int32, device="cpu"), 8)
    with pytest.raises(RuntimeError, match="CPU"):
        context.copy_from_host_tensor(torch.empty(4, dtype=torch.int32, device="cuda"))


def test_double_buffer_wrapper():
    from disagmoe.utils.gdr_context import GdrDoubleBuffer

    buffers = GdrDoubleBuffer(4, torch.int64)
    first, second = buffers.get_one_handle(), buffers.get_one_handle()
    assert first.tensor.data_ptr() != second.tensor.data_ptr()
    assert first.backend == second.backend == "cuda"
    first.copy_from_host_int64([1, 2, 3, 4])
    second.copy_from_host_int64([8, 7, 6, 5])
    assert first.copy_to_host_int64(4) == [1, 2, 3, 4]
    assert second.copy_to_host_int64(4) == [8, 7, 6, 5]


def test_stage_outside_graph_then_replay():
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        tensor = torch.ones(8, device="cuda", dtype=torch.int32)
        context = native.GdrContext(tensor)
        output = tensor + 1  # Warm up the operation before capture.
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            output = tensor + 1
        context.copy_from_host_int32(list(range(8)))
        graph.replay()
        output_context = native.GdrContext(output)
        assert output_context.copy_to_host_int32(8) == list(range(1, 9))


def test_reject_host_copy_during_capture():
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        tensor = torch.ones(8, device="cuda", dtype=torch.int32)
        context = native.GdrContext(tensor)
        tensor.add_(1)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            tensor.add_(1)
            # Catch inside capture: the guard must reject before issuing a
            # host copy/synchronize that would invalidate the CUDA graph.
            with pytest.raises(RuntimeError, match="outside CUDA graph capture"):
                context.copy_from_host_int32(list(range(8)))
        graph.replay()
        assert context.copy_to_host_int32(8) == [3] * 8
