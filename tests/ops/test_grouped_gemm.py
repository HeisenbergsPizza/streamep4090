"""Standalone unit test for disagmoe_c CutlassGemmRunner (setup_meta + run)."""
import torch
import disagmoe_c

torch.set_default_dtype(torch.bfloat16)


def _ref_grouped_gemm(a, b, batch_sizes, *, allow_reduced_precision=False):
    """Per-expert torch.matmul without BF16 intermediate reductions by default.

    CUTLASS accumulates in FP32. BF16 intermediate reductions in the torch
    reference can lose accuracy after SiLU/gating, especially near cancellation.
    Scope this setting to reference GEMMs, including when called through pytest.
    The opt-in path is only used for failure diagnostics.
    """
    old_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = allow_reduced_precision
        num_experts = b.size(0)
        N = b.size(2)
        outputs = []
        offset = 0
        for i in range(num_experts):
            M = int(batch_sizes[i].item())
            if M > 0:
                ai = a[offset : offset + M]     # [M, K]
                bi = b[i]                      # [K, N]
                outputs.append(ai @ bi)        # [M, N]
            offset += M
        return torch.cat(outputs, dim=0) if outputs else torch.empty(0, N, device=a.device, dtype=a.dtype)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduction


def _comparison_summary(label, actual, expected):
    """Report the original allclose criterion without relaxing its tolerances."""
    a, b = actual.double(), expected.double()
    diff = (a - b).abs()
    limit = 0.1 + 0.1 * b.abs()
    finite = torch.isfinite(a) & torch.isfinite(b)
    bad = ~torch.isclose(actual, expected, atol=0.1, rtol=0.1)
    score = torch.where(finite, diff / limit, torch.full_like(diff, float('inf')))
    flat_index = int(score.reshape(-1).argmax().item())
    row, col = divmod(flat_index, actual.size(1))
    print(
        f"{label}: mismatches={bad.sum().item()}/{bad.numel()}, "
        f"nonfinite={(~finite).sum().item()}, max_abs={diff.max().item():.8g}, "
        f"rms={diff.square().mean().sqrt().item():.8g}; "
        f"worst tolerance ratio at [{row}, {col}]: "
        f"actual={a[row, col].item():.8g}, expected={b[row, col].item():.8g}, "
        f"abs_error={diff[row, col].item():.8g}, limit={limit[row, col].item():.8g}",
        flush=True,
    )


def _assert_gemm_close(label, actual, expected, a, b, batch_sizes):
    if torch.allclose(actual, expected, atol=0.1, rtol=0.1):
        return
    print(f"[FAIL] {label}; input dtype={a.dtype}, stride={a.stride()}, "
          f"contiguous={a.is_contiguous()}", flush=True)
    _comparison_summary("CUTLASS vs torch reference (BF16 reduced reduction disabled)", actual, expected)

    # Compare the SAME input/weights to an independent, higher precision GEMM.
    # Round once to the BF16 output format; this is diagnostic, not a new gate.
    ref64 = _ref_grouped_gemm(a.double(), b.double(), batch_sizes)
    rounded64 = ref64.to(actual.dtype)
    _comparison_summary("CUTLASS vs FP64 rounded to BF16", actual, rounded64)
    _comparison_summary("Torch reference vs FP64 rounded to BF16", expected, rounded64)

    reduced_ref = _ref_grouped_gemm(a, b, batch_sizes, allow_reduced_precision=True)
    _comparison_summary("CUTLASS vs torch with BF16 reduced reduction enabled",
                        actual, reduced_ref)
    _comparison_summary("Torch with BF16 reduced reduction enabled vs FP64 rounded to BF16",
                        reduced_ref, rounded64)
    raise AssertionError(f"{label}: atol=0.1, rtol=0.1 comparison failed; see diagnostics above")


def test_basic():
    """Basic correctness: runner.setup_meta + runner.run vs torch.matmul."""
    E, K, N = 8, 128, 256
    tokens_per_expert = 64
    total_tokens = E * tokens_per_expert

    a = torch.randn(total_tokens, K, device="cuda")
    b = torch.randn(E, K, N, device="cuda")
    c = torch.empty(total_tokens, N, device="cuda")
    batch_sizes = torch.full((E,), tokens_per_expert, dtype=torch.int64, device="cuda")

    runner = disagmoe_c.CutlassGemmRunner(b, total_tokens)
    runner.setup_meta(a, c, batch_sizes)
    runner.run()
    torch.cuda.synchronize()

    ref = _ref_grouped_gemm(a, b, batch_sizes)
    assert torch.allclose(c, ref, atol=1e-1, rtol=1e-1), \
        f"Max diff: {(c - ref).abs().max().item()}"
    print(f"[PASS] test_basic  (max diff = {(c - ref).abs().max().item():.6f})")


def test_uneven_batch_sizes():
    """Experts with different batch sizes (including zero)."""
    E, K, N = 4, 64, 128
    sizes = [10, 0, 30, 20]
    total_tokens = sum(sizes)

    a = torch.randn(total_tokens, K, device="cuda")
    b = torch.randn(E, K, N, device="cuda")
    c = torch.empty(total_tokens, N, device="cuda")
    batch_sizes = torch.tensor(sizes, dtype=torch.int64, device="cuda")

    runner = disagmoe_c.CutlassGemmRunner(b, total_tokens)
    runner.setup_meta(a, c, batch_sizes)
    runner.run()
    torch.cuda.synchronize()

    ref = _ref_grouped_gemm(a, b, batch_sizes)
    assert torch.allclose(c[:total_tokens], ref, atol=1e-1, rtol=1e-1), \
        f"Max diff: {(c[:total_tokens] - ref).abs().max().item()}"
    print(f"[PASS] test_uneven_batch_sizes  (max diff = {(c[:total_tokens] - ref).abs().max().item():.6f})")


def test_moe_pattern():
    """Mimics the MoEExpertsCUTLASS up/down projection pattern."""
    E = 8
    hidden = 256
    inter = 512
    tokens_per_expert = 32
    total_tokens = E * tokens_per_expert

    hiddens = torch.randn(total_tokens, hidden, device="cuda")
    w13 = torch.randn(E, hidden, inter * 2, device="cuda")
    w2 = torch.randn(E, inter, hidden, device="cuda")
    cache_up = torch.empty(total_tokens, inter * 2, device="cuda")
    cache_down = torch.empty(total_tokens, hidden, device="cuda")
    batch_sizes = torch.full((E,), tokens_per_expert, dtype=torch.int64, device="cuda")

    w13_runner = disagmoe_c.CutlassGemmRunner(w13, total_tokens)
    w2_runner = disagmoe_c.CutlassGemmRunner(w2, total_tokens)

    # Up projection
    w13_runner.setup_meta(hiddens, cache_up, batch_sizes)
    w13_runner.run()
    ref_up = _ref_grouped_gemm(hiddens, w13, batch_sizes)
    _assert_gemm_close("MoE up", cache_up, ref_up, hiddens, w13, batch_sizes)
    # SiLU below modifies cache_up in place; save this statistic first.
    up_diff = (cache_up.float() - ref_up.float()).abs().max().item()

    # SiLU + gate
    act = torch.nn.SiLU(inplace=True)
    up = act(cache_up[:, :inter]) * cache_up[:, inter:]

    # Down projection
    w2_runner.setup_meta(up, cache_down, batch_sizes)
    w2_runner.run()
    ref_down = _ref_grouped_gemm(up, w2, batch_sizes)
    _assert_gemm_close("MoE down", cache_down, ref_down, up, w2, batch_sizes)

    print(f"[PASS] test_moe_pattern  (up diff = {up_diff:.6f}, "
          f"down diff = {(cache_down - ref_down).abs().max().item():.6f})")


def test_cuda_graph_capture():
    """runner.setup_meta + runner.run inside CUDA graph capture."""
    E, K, N = 8, 128, 256
    tokens_per_expert = 32
    total_tokens = E * tokens_per_expert

    b = torch.randn(E, K, N, device="cuda")
    runner = disagmoe_c.CutlassGemmRunner(b, total_tokens)

    # Static tensors for graph capture
    static_a = torch.randn(total_tokens, K, device="cuda")
    static_c = torch.empty(total_tokens, N, device="cuda")
    static_bs = torch.full((E,), tokens_per_expert, dtype=torch.int64, device="cuda")

    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        runner.setup_meta(static_a, static_c, static_bs)
        runner.run()
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        runner.setup_meta(static_a, static_c, static_bs)
        runner.run()

    # Replay with same data
    graph.replay()
    torch.cuda.synchronize()

    ref = _ref_grouped_gemm(static_a, b, static_bs)
    assert torch.allclose(static_c, ref, atol=1e-1, rtol=1e-1), \
        f"Max diff: {(static_c - ref).abs().max().item()}"

    # Replay with NEW input data
    new_a = torch.randn(total_tokens, K, device="cuda")
    static_a.copy_(new_a)
    graph.replay()
    torch.cuda.synchronize()

    ref2 = _ref_grouped_gemm(new_a, b, static_bs)
    assert torch.allclose(static_c, ref2, atol=1e-1, rtol=1e-1), \
        f"Max diff after replay: {(static_c - ref2).abs().max().item()}"

    print(f"[PASS] test_cuda_graph_capture  "
          f"(replay diff = {(static_c - ref2).abs().max().item():.6f})")


def test_cuda_graph_varying_batch_sizes():
    """CUDA graph replayed with different batch_sizes contents each time."""
    E, K, N = 4, 64, 128
    max_tokens = 200

    b = torch.randn(E, K, N, device="cuda")
    runner = disagmoe_c.CutlassGemmRunner(b, max_tokens)

    static_a = torch.randn(max_tokens, K, device="cuda")
    static_c = torch.empty(max_tokens, N, device="cuda")
    static_bs = torch.zeros(E, dtype=torch.int64, device="cuda")

    # Warmup
    sizes1 = [30, 20, 40, 10]
    total1 = sum(sizes1)
    static_bs.copy_(torch.tensor(sizes1, dtype=torch.int64))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        runner.setup_meta(static_a, static_c, static_bs)
        runner.run()
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        runner.setup_meta(static_a, static_c, static_bs)
        runner.run()

    # Replay round 1
    a1 = torch.randn(max_tokens, K, device="cuda")
    static_a.copy_(a1)
    static_bs.copy_(torch.tensor(sizes1, dtype=torch.int64))
    graph.replay()
    torch.cuda.synchronize()

    ref1 = _ref_grouped_gemm(a1, b, torch.tensor(sizes1, dtype=torch.int64, device="cuda"))
    assert torch.allclose(static_c[:total1], ref1, atol=1e-1, rtol=1e-1), \
        f"Round 1 max diff: {(static_c[:total1] - ref1).abs().max().item()}"

    # Replay round 2 -- different batch sizes
    sizes2 = [50, 0, 10, 40]
    total2 = sum(sizes2)
    a2 = torch.randn(max_tokens, K, device="cuda")
    static_a.copy_(a2)
    static_bs.copy_(torch.tensor(sizes2, dtype=torch.int64))
    graph.replay()
    torch.cuda.synchronize()

    ref2 = _ref_grouped_gemm(a2, b, torch.tensor(sizes2, dtype=torch.int64, device="cuda"))
    assert torch.allclose(static_c[:total2], ref2, atol=1e-1, rtol=1e-1), \
        f"Round 2 max diff: {(static_c[:total2] - ref2).abs().max().item()}"

    print(f"[PASS] test_cuda_graph_varying_batch_sizes  "
          f"(r1 diff = {(static_c[:total1] - ref1).abs().max().item():.6f}, "
          f"r2 diff = {(static_c[:total2] - ref2).abs().max().item():.6f})")


def test_moe_pattern_cuda_graph():
    """Full MoE up+down with runner.setup_meta + runner.run inside CUDA graphs."""
    E = 8
    hidden = 256
    inter = 512
    tokens_per_expert = 32
    total_tokens = E * tokens_per_expert

    w13 = torch.randn(E, hidden, inter * 2, device="cuda")
    w2 = torch.randn(E, inter, hidden, device="cuda")

    w13_runner = disagmoe_c.CutlassGemmRunner(w13, total_tokens)
    w2_runner = disagmoe_c.CutlassGemmRunner(w2, total_tokens)

    static_hiddens = torch.randn(total_tokens, hidden, device="cuda")
    static_up_out = torch.empty(total_tokens, inter * 2, device="cuda")
    static_down_out = torch.empty(total_tokens, hidden, device="cuda")
    static_bs = torch.full((E,), tokens_per_expert, dtype=torch.int64, device="cuda")
    act = torch.nn.SiLU(inplace=True)

    # -- w13 graph --
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        w13_runner.setup_meta(static_hiddens, static_up_out, static_bs)
        w13_runner.run()
    torch.cuda.current_stream().wait_stream(s)

    graph_w13 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_w13, stream=s):
        w13_runner.setup_meta(static_hiddens, static_up_out, static_bs)
        w13_runner.run()

    graph_w13.replay()
    torch.cuda.synchronize()

    ref_up = _ref_grouped_gemm(static_hiddens, w13, static_bs)
    _assert_gemm_close("MoE graph up", static_up_out, ref_up, static_hiddens, w13, static_bs)
    up_diff = (static_up_out.float() - ref_up.float()).abs().max().item()

    # Activation (eager, between graphs)
    up = act(static_up_out[:, :inter]) * static_up_out[:, inter:]

    # -- w2 graph --
    static_up_input = up.clone()
    s2 = torch.cuda.Stream()
    s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2):
        w2_runner.setup_meta(static_up_input, static_down_out, static_bs)
        w2_runner.run()
    torch.cuda.current_stream().wait_stream(s2)

    graph_w2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_w2, stream=s2):
        w2_runner.setup_meta(static_up_input, static_down_out, static_bs)
        w2_runner.run()

    graph_w2.replay()
    torch.cuda.synchronize()

    ref_down = _ref_grouped_gemm(static_up_input, w2, static_bs)
    _assert_gemm_close("MoE graph down", static_down_out, ref_down, static_up_input, w2, static_bs)

    print(f"[PASS] test_moe_pattern_cuda_graph  "
          f"(w13 diff = {up_diff:.6f}, "
          f"w2 diff = {(static_down_out - ref_down).abs().max().item():.6f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-bf16-reduced-precision", action="store_true",
                        help="Also disable BF16 reduced reduction for the process; references already disable it locally")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.no_bf16_reduced_precision:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    print(f"Seed: {args.seed}; torch={torch.__version__}; "
          "reference BF16 reduced precision reduction=False (scoped); "
          f"process setting={torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}")
    dev = torch.cuda.current_device()
    cap = torch.cuda.get_device_capability(dev)
    print(f"Device: {torch.cuda.get_device_name(dev)}, SM {cap[0]}{cap[1]}")

    desc = disagmoe_c.init_grouped_gemm(dev)
    print(f"Config: {desc}")
    print()

    test_basic()
    test_uneven_batch_sizes()
    test_moe_pattern()
    print()

    print("--- CUDA graph tests ---")
    test_cuda_graph_capture()
    test_cuda_graph_varying_batch_sizes()
    test_moe_pattern_cuda_graph()

    print("\nAll tests passed!")
