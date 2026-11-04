"""Numerical regressions for the CUDA routing ops.

Run after building the extension with ENABLE_GDRCOPY=0 KERNEL_USE_GDRCOPY=0:
    python -m pytest tests/ops/test_token_routing.py -q
The raw-pointer gather ABI is for eager scheduler merges, outside CUDA graphs.
"""

import ctypes
import importlib

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module", autouse=True)
def cuda_ops():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for token routing kernels")
    # On a CUDA host a missing/broken extension must fail, not silently skip.
    importlib.import_module("disagmoe_c")


def _tokens(rows, width):
    # Exactly representable, distinct rows/columns, with both signs.
    values = torch.arange(rows * width, device="cuda", dtype=torch.float32)
    return ((values.reshape(rows, width) % 97 - 48) / 16).to(torch.bfloat16)


def _mappings(count):
    generator = torch.Generator().manual_seed(4019)
    return torch.randperm(count, generator=generator).to(device="cuda", dtype=torch.int32)


def _pointer_array(rows):
    return (ctypes.c_uint64 * len(rows))(*(row.data_ptr() for row in rows))


@pytest.mark.parametrize("width", [17, 128, 132, 960, 2048, 2051])
@pytest.mark.parametrize("topk", [1, 8])
def test_permutation_and_top8_scatter(width, topk):
    tokens = _tokens(5, width)
    mappings = _mappings(tokens.size(0) * topk)
    actual = torch.ops.disag_ops.permute_tokens(tokens, mappings)
    # Looking up output destinations must recover the original token sequence.
    expected = tokens.repeat_interleave(topk, dim=0)
    torch.testing.assert_close(actual[mappings.long()], expected, rtol=0, atol=0)


@pytest.mark.parametrize("width", [17, 132, 2048, 2051])
def test_weighted_permutation_uses_each_source_weight_once(width):
    tokens = _tokens(5, width)
    mappings = torch.tensor([3, 0, 4, 1, 2, 99, 99], device="cuda", dtype=torch.int32)
    weights = torch.tensor([0.13, -0.37, 0.91, 1.7, 0.003, 99, 99], device="cuda", dtype=torch.float32)
    actual = torch.ops.disag_ops.apply_weights_and_permute_tokens(tokens, weights, mappings)
    expected = (tokens.float() * weights[:5, None]).to(torch.bfloat16)
    torch.testing.assert_close(actual[mappings[:5].long()], expected, rtol=0, atol=0)


def test_contiguous_unaligned_storage_view():
    storage = _tokens(1, 5 * 2048 + 1).flatten()
    tokens = storage[1:].view(5, 2048)
    assert tokens.is_contiguous() and tokens.data_ptr() % 8 != 0
    mappings = _mappings(5)
    actual = torch.ops.disag_ops.permute_tokens(tokens, mappings)
    torch.testing.assert_close(actual[mappings.long()], tokens, rtol=0, atol=0)


@pytest.mark.parametrize("width", [17, 132, 2048, 2051])
def test_gather_arbitrary_source_rows_and_temporary_host_table(width):
    source = _tokens(7, width)
    indices = [5, 1, 6, 0, 3]
    rows = [source[i] for i in indices]
    pointers = _pointer_array(rows)
    output = torch.empty((len(rows), width), device="cuda", dtype=torch.bfloat16)
    torch.ops.disag_ops.gather_tokens(output, ctypes.addressof(pointers), len(rows), width)
    # The op must take ownership of a copy of this temporary CPU table.
    for i in range(len(pointers)):
        pointers[i] = 0
    torch.testing.assert_close(output, source[indices], rtol=0, atol=0)


@pytest.mark.parametrize("width", [17, 132, 2048, 2051])
def test_top8_gather_sum_accumulates_small_contributions(width):
    n, topk = 3, 8
    # BF16 256 + 1 repeatedly rounds back to 256. FP32 accumulation must
    # retain all seven small contributions and round only the final result.
    source = torch.ones((n, topk, width), device="cuda", dtype=torch.bfloat16)
    source[:, 0] = 256
    source[1] *= -1
    source[2] *= 0.5
    expected = source.float().sum(dim=1).to(torch.bfloat16)
    pointers = _pointer_array([source[i, k] for i in range(n) for k in range(topk)])
    output = torch.empty_like(expected)
    torch.ops.disag_ops.gather_and_sum_tokens(output, ctypes.addressof(pointers), n, topk, width)
    for i in range(len(pointers)):
        pointers[i] = 0
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_scatter_weighted_return_and_top8_join():
    n, topk, width = 4, 8, 2048
    tokens = _tokens(n, width)
    dispatch = _mappings(n * topk)
    scattered = torch.ops.disag_ops.permute_tokens(tokens, dispatch)
    # Model an expert transform that depends on the expert slot, followed by
    # nonuniform routing weights and a nontrivial return ordering.
    expert_factors = torch.linspace(0.25, 2.0, n * topk, device="cuda", dtype=torch.float32)
    expert_outputs = (scattered.float() * expert_factors[:, None]).to(torch.bfloat16)
    weights = torch.linspace(0.01, 0.49, n * topk, device="cuda", dtype=torch.float32)
    return_order = dispatch.argsort().to(torch.int32)
    weighted = torch.ops.disag_ops.apply_weights_and_permute_tokens(expert_outputs, weights, return_order)
    pointers = _pointer_array(list(weighted.unbind(0)))
    result = torch.empty((n, width), device="cuda", dtype=torch.bfloat16)
    torch.ops.disag_ops.gather_and_sum_tokens(result, ctypes.addressof(pointers), n, topk, width)
    reference_branches = (expert_outputs.float() * weights[:, None]).to(torch.bfloat16)
    expected = reference_branches[dispatch.long()].view(n, topk, width).float().sum(dim=1).to(torch.bfloat16)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_empty_batches_do_not_launch_zero_sized_grids():
    tokens = torch.empty((0, 2048), device="cuda", dtype=torch.bfloat16)
    mappings = torch.empty(0, device="cuda", dtype=torch.int32)
    weights = torch.empty(0, device="cuda", dtype=torch.float32)
    assert torch.ops.disag_ops.permute_tokens(tokens, mappings).shape == tokens.shape
    assert torch.ops.disag_ops.apply_weights_and_permute_tokens(tokens, weights, mappings).shape == tokens.shape
    torch.ops.disag_ops.gather_tokens(tokens, 0, 0, 2048)
    torch.ops.disag_ops.gather_and_sum_tokens(tokens, 0, 0, 8, 2048)


@pytest.mark.parametrize("invalid", ["mapping_dtype", "weight_dtype", "short_weights", "short_mapping", "noncontiguous", "cpu_weights"])
def test_weighted_permutation_rejects_invalid_metadata(invalid):
    tokens = _tokens(5, 2048)
    mappings = _mappings(5)
    weights = torch.ones(5, device="cuda", dtype=torch.float32)
    if invalid == "mapping_dtype":
        mappings = mappings.long()
    elif invalid == "weight_dtype":
        weights = weights.to(torch.bfloat16)
    elif invalid == "short_weights":
        weights = weights[:4]
    elif invalid == "short_mapping":
        mappings = mappings[:4]
    elif invalid == "noncontiguous":
        tokens = tokens[:, ::2]
    elif invalid == "cpu_weights":
        weights = weights.cpu()
    with pytest.raises(RuntimeError):
        torch.ops.disag_ops.apply_weights_and_permute_tokens(tokens, weights, mappings)


@pytest.mark.parametrize("topk", [0, -1])
def test_gather_sum_rejects_invalid_topk(topk):
    output = _tokens(2, 2048)
    with pytest.raises(RuntimeError, match="topk must be positive"):
        torch.ops.disag_ops.gather_and_sum_tokens(output, 0, 2, topk, 2048)


def test_permutation_and_weighted_permutation_cuda_graph_replay():
    tokens = _tokens(5, 2048)
    mappings = _mappings(5)
    weights = torch.linspace(0.07, 0.83, 5, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.ops.disag_ops.permute_tokens(tokens, mappings)
            torch.ops.disag_ops.apply_weights_and_permute_tokens(tokens, weights, mappings)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        permuted = torch.ops.disag_ops.permute_tokens(tokens, mappings)
        weighted = torch.ops.disag_ops.apply_weights_and_permute_tokens(tokens, weights, mappings)
    tokens.mul_(2)
    weights.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(permuted[mappings.long()], tokens, rtol=0, atol=0)
    expected = (tokens.float() * weights[:, None]).to(torch.bfloat16)
    torch.testing.assert_close(weighted[mappings.long()], expected, rtol=0, atol=0)


def test_gather_ops_reject_raw_host_tables_during_cuda_graph_capture():
    source = _tokens(2, 2048)
    output = torch.empty_like(source)
    pointers = _pointer_array(list(source.unbind(0)))
    for name, args in [
        ("gather_tokens", (output, ctypes.addressof(pointers), 2, 2048)),
        ("gather_and_sum_tokens", (output, ctypes.addressof(pointers), 2, 1, 2048)),
    ]:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with pytest.raises(RuntimeError, match="outside CUDA graph capture"):
                getattr(torch.ops.disag_ops, name)(*args)


def test_ops_guard_the_input_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices required to test device guards")
    tokens = _tokens(5, 2048).to("cuda:0")
    mappings = _mappings(5).to("cuda:0")
    with torch.cuda.device(1):
        actual = torch.ops.disag_ops.permute_tokens(tokens, mappings)
    torch.testing.assert_close(actual[mappings.long()], tokens, rtol=0, atol=0)
