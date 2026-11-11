"""Profile one 512-request burst against the matching DPA4/EP4 server."""
import argparse
import json
import os
import time
from pathlib import Path

BASE_URL = "http://127.0.0.1:30000"
BATCH_SIZE = 512
OUTPUT_TOKENS = 128
PROMPT_BODY = (
    "请用中文解释大语言模型生成文本时，"
    "prefill 和 decoding 分别做什么，并举一个简单例子。"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-tokens", type=int,
        help="Exact total prompt length, including chat template tokens; default: original text prompts",
    )
    args = parser.parse_args(argv)
    if args.input_tokens is not None and args.input_tokens < 1:
        parser.error("--input-tokens must be positive")
    return args


def make_prompt(tokenizer, i, phase):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{phase}任务，编号 {i}。" + PROMPT_BODY}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def fixed_prompt_ids(tokenizer, i, phase, input_tokens):
    """Keep the chat boundaries and request identity; resize only body tokens."""
    marker = "__SGLANG_PROFILE_USER_CONTENT__"
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if template.count(marker) != 1:
        raise ValueError("Chat template must preserve the user content marker exactly once")
    prefix, suffix = template.split(marker)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    identity_ids = tokenizer.encode(f"{phase}任务，编号 {i}。", add_special_tokens=False)
    room = input_tokens - len(prefix_ids) - len(identity_ids) - len(suffix_ids)
    if room < 1:
        minimum = input_tokens - room + 1
        raise ValueError(f"--input-tokens={input_tokens} is too short for the template and request ID; need at least {minimum}")
    body_ids = tokenizer.encode(PROMPT_BODY, add_special_tokens=False)
    # Ordinary text tokens count as real context. Do not use PAD tokens or cut
    # off the assistant generation suffix; the API receives no attention mask.
    filler_ids = tokenizer.encode(" 请详细说明。", add_special_tokens=False)
    if not filler_ids:
        raise ValueError("Tokenizer returned no tokens for the prompt filler")
    missing = max(0, room - len(body_ids))
    body_ids += filler_ids * ((missing + len(filler_ids) - 1) // len(filler_ids))
    return prefix_ids + identity_ids + body_ids[:room] + suffix_ids


def batch_payload(tokenizer, phase, input_tokens=None):
    if input_tokens is None:
        inputs = {"text": [make_prompt(tokenizer, i, phase) for i in range(BATCH_SIZE)]}
    else:
        inputs = {"input_ids": [
            fixed_prompt_ids(tokenizer, i, phase, input_tokens) for i in range(BATCH_SIZE)
        ]}
    return {
        **inputs,
        "sampling_params": {
            "temperature": 0, "max_new_tokens": OUTPUT_TOKENS, "ignore_eos": True,
        },
        "stream": False,
    }


def validate_batch(result, *, input_tokens=None, phase="测量"):
    if not isinstance(result, list) or len(result) != BATCH_SIZE:
        raise RuntimeError(f"{phase}：服务端没有返回全部 {BATCH_SIZE} 个请求的结果")
    prompts, completions, cached = [], [], []
    for index, item in enumerate(result):
        meta = item.get("meta_info") if isinstance(item, dict) else None
        if not isinstance(meta, dict):
            raise RuntimeError(f"{phase}：请求 {index} 缺少 meta_info")
        prompt, completion = meta.get("prompt_tokens"), meta.get("completion_tokens")
        if type(prompt) is not int or prompt < 1:
            raise RuntimeError(f"{phase}：请求 {index} 的 prompt_tokens 无效：{prompt!r}")
        if input_tokens is not None and prompt != input_tokens:
            raise RuntimeError(f"{phase}：请求 {index} 的 prompt_tokens={prompt}，预期 {input_tokens}")
        if type(completion) is not int or completion != OUTPUT_TOKENS:
            raise RuntimeError(f"{phase}：请求 {index} 的 completion_tokens={completion!r}，预期 {OUTPUT_TOKENS}")
        prompts.append(prompt)
        completions.append(completion)
        cache_count = meta.get("cached_tokens")
        if type(cache_count) is int and cache_count >= 0:
            cached.append(cache_count)
    return {
        "prompt_tokens": {"min": min(prompts), "max": max(prompts)},
        "completion_tokens": {"min": min(completions), "max": max(completions)},
        "cached_tokens": {
            "reported_requests": len(cached),
            "min": min(cached) if cached else None,
            "max": max(cached) if cached else None,
        },
    }


def graph_disabled_from_env():
    value = os.environ.get("SGLANG_DISABLE_CUDA_GRAPH", "0")
    if value not in ("0", "1"):
        raise ValueError("SGLANG_DISABLE_CUDA_GRAPH must be 0 or 1")
    return value == "1"


def validate_server(info, *, disable_cuda_graph=False):
    expected = {
        "tp_size": 4, "dp_size": 4, "ep_size": 4, "moe_dp_size": 1,
        "attn_cp_size": 1, "enable_dp_attention": True,
        "enable_dp_lm_head": False, "max_running_requests": BATCH_SIZE,
        "cuda_graph_max_bs": 128, "disable_cuda_graph": disable_cuda_graph,
        "moe_a2a_backend": "none", "moe_runner_backend": "triton",
        "attention_backend": "triton", "dtype": "bfloat16",
        "load_balance_method": "round_robin",
    }
    for key, value in expected.items():
        if info.get(key) != value:
            raise RuntimeError(
                f"服务配置不匹配：{key}={info.get(key)!r}，预期 {value!r}。"
                "请确认启动的是 profile_server_dpa4_ep4_b512.sh，"
                "且两端的 SGLANG_DISABLE_CUDA_GRAPH 设置一致。"
            )
    states = info.get("internal_states", [])
    if len(states) != 4:
        raise RuntimeError(f"预期 4 个 DP worker 状态，实际为 {len(states)}")
    # Response order is not DP-rank order. Check every worker independently.
    for state in states:
        cap = state.get("effective_max_running_requests_per_dp")
        if cap != BATCH_SIZE // 4:
            raise RuntimeError(f"某个 DP worker 实际请求上限为 {cap}，预期 128")


def main(argv=None):
    args = parse_args(argv)
    import requests

    disable_cuda_graph = graph_disabled_from_env()
    graph_mode = "graph-off" if disable_cuda_graph else "graph-on"
    input_tag = f"-input{args.input_tokens}" if args.input_tokens is not None else ""
    work = Path(os.environ["SGLANG_WORK"])
    model_dir = os.environ["MODEL_DIR"]
    output = work / "reports" / (
        f"generation-dpa4-ep4-b512-{graph_mode}{input_tag}-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 10**9:09d}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False  # All requests are to the local server.

    def get(path):
        response = session.get(BASE_URL + path, timeout=(10, 1800))
        response.raise_for_status()
        return response

    def post(path, payload):
        response = session.post(BASE_URL + path, json=payload, timeout=(10, 1800))
        response.raise_for_status()
        return response

    get("/health")
    info = get("/server_info").json()
    info_path = output.with_suffix(".server-info.json")
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    validate_server(info, disable_cuda_graph=disable_cuda_graph)
    print("服务配置检查通过：Attention DP4/TP1，Experts EP4/TP1，每 DP 请求上限 128。", flush=True)
    print(f"CUDA Graph: {graph_mode}", flush=True)
    print(f"服务端实际配置：{info_path}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    # Construct both batches before starting the profiler or sending warmup.
    warmup_payload = batch_payload(tokenizer, "预热", args.input_tokens)
    payload = batch_payload(tokenizer, "测量", args.input_tokens)
    if args.input_tokens is not None:
        print(f"固定输入 {args.input_tokens} token（包含聊天模板）；KV 长度随生成增长。", flush=True)

    print("=== 正常生成验证 ===", flush=True)
    check = post("/generate", {
        "text": make_prompt(tokenizer, 0, "验证"),
        "sampling_params": {"temperature": 0, "max_new_tokens": 128},
    }).json()
    print(check["text"], flush=True)

    print(f"=== {BATCH_SIZE} 请求预热 ===", flush=True)
    warmup_result = post("/generate", warmup_payload).json()
    warmup_summary = validate_batch(warmup_result, input_tokens=args.input_tokens, phase="预热")
    print(f"预热长度检查通过：{warmup_summary}", flush=True)
    del warmup_result

    print("=== 开始采集 ===", flush=True)
    start_message = post("/start_profile", {"activities": ["CUDA_PROFILER"]}).text
    print(start_message, flush=True)
    result = None
    elapsed = None
    stop_message = None
    request_error = None
    stop_error = None
    token_summary = None
    try:
        started = time.perf_counter()
        result = post("/generate", payload).json()
        elapsed = time.perf_counter() - started
        token_summary = validate_batch(result, input_tokens=args.input_tokens)
    except Exception as exc:
        request_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            stop_message = post("/stop_profile", {}).text
            print(stop_message, flush=True)
        except Exception as exc:
            stop_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            output.write_text(json.dumps({
                "batch_size": BATCH_SIZE,
                "input_tokens": args.input_tokens,
                "prompt_mode": "fixed_input_ids" if args.input_tokens is not None else "text",
                "output_tokens": OUTPUT_TOKENS,
                "cuda_graph_enabled": not disable_cuda_graph,
                "client_elapsed_seconds": elapsed,
                "server_info": info,
                "profile_start_response": start_message,
                "profile_stop_response": stop_message,
                "request_error": request_error,
                "profile_stop_error": stop_error,
                "warmup_token_summary": warmup_summary,
                "token_summary": token_summary,
                "request": payload,
                "response": result,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"结果已保存：{output}", flush=True)

    print(f"PASS：{BATCH_SIZE} 个请求均生成 {OUTPUT_TOKENS} token", flush=True)
    print(f"服务端 token 数量：{token_summary}", flush=True)
    print(f"客户端请求总耗时（含 prefill、decode 和 HTTP）：{elapsed:.3f} 秒", flush=True)
    print("等待 Nsight 报告生成后，在服务端终端按 Ctrl+C 停止服务。", flush=True)


if __name__ == "__main__":
    main()
