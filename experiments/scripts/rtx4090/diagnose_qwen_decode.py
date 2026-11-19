"""Rank StreamInfer's first differing token under the snapshot exporter's HF model."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from disagmoe.utils.decode_snapshot import DecodeSnapshot, validate_token_ids


def failed_rows(report, snapshot, vocab_size):
    requests = report["requests"]
    if (len(requests) != len(snapshot)
            or {r["snapshot_index"] for r in requests} != set(range(len(snapshot)))):
        raise ValueError("Report and snapshot have different request indices")
    failures = []
    for row in requests:
        i = row["snapshot_index"]
        actual = row["token_ids"]
        validate_token_ids(actual, vocab_size, f"requests[{i}].token_ids")
        reference = snapshot.expected_tokens(i)[:len(actual)]
        if (len(actual) != report["decode_steps"] or not reference
                or row["prefill_token"] != snapshot.first_token(i)
                or reference != row["reference_prefix"]):
            raise ValueError(f"Report and snapshot disagree for request {i}")
        step = next((s for s, token in enumerate(reference) if token != actual[s]), None)
        if step != row["first_different_step"]:
            raise ValueError(f"Inconsistent first_different_step for request {i}")
        if step is not None:
            failures.append(row)
    return failures


def replay_first_difference(model, snapshot, row):
    import torch

    index = row["snapshot_index"]
    step = row["first_different_step"]
    expected = row["reference_prefix"][step]
    actual = row["token_ids"][step]
    device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        prompt = torch.tensor([snapshot.prompt_ids(index)], device=device)
        prefill = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
        first_token = prefill.logits[0, -1].argmax().item()
        cache = prefill.past_key_values
        # Prefill constructs the version-specific HF cache. Install the exact
        # exported keys and values before stepping, even if prefill drifted.
        saved = snapshot.load_kv(index)
        kv_equal = True
        for layer in range(snapshot.layers):
            for axis, tensor in enumerate(cache[layer]):
                source = saved[layer, axis].transpose(0, 1).unsqueeze(0).to(tensor.device)
                if tensor.shape != source.shape or tensor.dtype != source.dtype:
                    raise ValueError(f"Incompatible reference KV layout at layer {layer}")
                if not torch.equal(tensor, source):
                    kv_equal = False
                    tensor.copy_(source)

        inputs = [row["prefill_token"], *row["reference_prefix"][:step]]
        greedy = []
        for j, token in enumerate(inputs):
            output = model(input_ids=torch.tensor([[token]], device=device), past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            logits = output.logits[0, -1].float()
            if not torch.isfinite(logits).all().item():
                raise ValueError(f"Nonfinite HF logits for request {index} at decode step {j}")
            greedy.append(logits.argmax().item())

        top_two = torch.topk(logits, 2).values
        return {
            "snapshot_index": index, "first_different_step": step,
            "prefill_token_reproduced": first_token == row["prefill_token"],
            "prefill_kv_bitwise_equal": kv_equal,
            "hf_reference_reproduced_through_step": greedy == row["reference_prefix"][:step + 1],
            "hf_greedy_token": greedy[-1],
            "reference_token": expected, "streaminfer_token": actual,
            "hf_reference_logit": logits[expected].item(), "hf_streaminfer_logit": logits[actual].item(),
            "hf_reference_rank": 1 + (logits > logits[expected]).sum().item(),
            "hf_streaminfer_rank": 1 + (logits > logits[actual]).sum().item(),
            "hf_reference_minus_streaminfer_logit": (logits[expected] - logits[actual]).item(),
            "hf_top1_minus_top2_logit": (top_two[0] - top_two[1]).item(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--decode-snapshot", required=True, type=Path)
    parser.add_argument("--decode-report", required=True, type=Path, help="Benchmark *.decode.json")
    parser.add_argument("--limit", type=int, help="Replay only the first N differing requests")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    snapshot = DecodeSnapshot(args.decode_snapshot, args.model_dir)
    report = json.loads(args.decode_report.read_text())
    config = json.loads((args.model_dir / "config.json").read_text())
    failures = failed_rows(report, snapshot, config["vocab_size"])
    selected = failures[:args.limit]
    print(f"Replaying {len(selected)}/{len(failures)} mismatches among {len(snapshot)} requests.",
          file=sys.stderr, flush=True)
    if not selected:
        return

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        parser.error("Run in the CUDA environment used to export the snapshot")
    manifest = json.loads((args.decode_snapshot / "manifest.json").read_text())
    print(f"Transformers {transformers.__version__} (export {manifest.get('transformers_version')}), "
          f"Torch {torch.__version__} (export {manifest.get('torch_version')})", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map="balanced", attn_implementation="eager",
    ).eval()
    reproduced = tied = top_two = 0
    for row in selected:
        result = replay_first_difference(model, snapshot, row)
        print(json.dumps(result), flush=True)
        if result["hf_reference_reproduced_through_step"]:
            reproduced += 1
            rank = result["hf_streaminfer_rank"]
            tied += rank == 1
            top_two += rank <= 2
    print(f"From saved KV: HF reproduced {reproduced}/{len(selected)} reference prefixes. "
          f"Among those, StreamInfer's token tied for first in {tied}, "
          f"ranked in the top two in {top_two} (ties share rank).", file=sys.stderr)


if __name__ == "__main__":
    main()
