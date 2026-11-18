"""Export BF16 prompt KV, the first generated token and a greedy reference continuation.

Run with a Qwen3-capable Transformers environment, for example transformers==4.51.3
with accelerate, torch and safetensors. All model and tokenizer files are read locally.
"""

import argparse
import json
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from disagmoe.utils.decode_snapshot import FORMAT, checkpoint_metadata, kv_name, validate_token_ids


@contextmanager
def prepare_directory(destination):
    """Stage expensive exports privately; publish the manifest only after every KV file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temp:
        staging = Path(temp)
        yield staging
        manifest = staging / "manifest.json"
        if not manifest.is_file():
            raise ValueError("Incomplete export: missing manifest")
        # mkdir is an exclusive claim, including when another exporter finishes first.
        destination.mkdir()
        with ExitStack() as rollback:
            rollback.callback(destination.rmdir)
            for source in [*sorted(staging.glob("kv-*.safetensors")), manifest]:
                target = destination / source.name
                source.rename(target)
                rollback.callback(target.unlink)
            rollback.pop_all()


def load_prompts(path, model_dir, vocab_size):
    prompts = []
    tokenizer = None
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or ("text" in row) == ("input_ids" in row):
                raise ValueError(f"{path}:{line_no}: supply either input_ids or text")
            if "input_ids" in row:
                ids = row["input_ids"]
            else:
                if not isinstance(row["text"], str):
                    raise ValueError(f"{path}:{line_no}: text must be a string")
                if tokenizer is None:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
                ids = tokenizer.encode(row["text"], add_special_tokens=False)
            validate_token_ids(ids, vocab_size, f"{path}:{line_no}")
            prompts.append(ids)
    if not prompts:
        raise ValueError("Prompt file is empty")
    return prompts


def export_prompt(model, ids, kv_path, reference_steps):
    import torch
    from safetensors.torch import save_file

    config = model.config
    device = model.get_input_embeddings().weight.device
    inputs = torch.tensor([ids], dtype=torch.int64, device=device)
    with torch.inference_mode():
        output = model(input_ids=inputs, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        token = output.logits[0, -1].argmax().item()
        first_token = token
        # Copy prompt KV before passing the mutable HF cache into a decode step.
        kv = torch.empty((config.num_hidden_layers, 2, len(ids), config.num_key_value_heads,
                          config.head_dim), device="cpu", dtype=torch.bfloat16)
        for layer in range(config.num_hidden_layers):
            key, value = cache[layer]
            for axis, tensor in enumerate((key, value)):
                expected = (1, config.num_key_value_heads, len(ids), config.head_dim)
                if tuple(tensor.shape) != expected or tensor.dtype != torch.bfloat16:
                    raise ValueError(f"Unexpected reference KV at layer {layer}: {tensor.shape} {tensor.dtype}")
                kv[layer, axis].copy_(tensor[0].transpose(0, 1))
        save_file({"kv": kv}, str(kv_path))
        del kv

        expected_tokens = []
        for _ in range(reference_steps):
            inputs = torch.tensor([[token]], dtype=torch.int64, device=device)
            output = model(input_ids=inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            token = output.logits[0, -1].argmax().item()
            expected_tokens.append(token)
    return {"input_ids": ids, "first_token": first_token, "expected_tokens": expected_tokens}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New snapshot directory")
    parser.add_argument("--prompts-jsonl", type=Path, required=True,
                        help="One object per line with input_ids or raw text (no chat templating)")
    parser.add_argument("--reference-steps", type=int, default=4,
                        help="Greedy steps after prefill, ignoring EOS just like fixed-length decode")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Requests per prompt; duplicates share only the offline file")
    args = parser.parse_args()
    if args.reference_steps < 1 or args.repeat < 1:
        parser.error("Require --reference-steps >= 1 and --repeat >= 1")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    config = json.loads((args.model_dir / "config.json").read_text())
    if config["model_type"] != "qwen3_moe":
        parser.error("Expected a Qwen3 MoE checkpoint")
    digests = checkpoint_metadata(args.model_dir)
    prompts = load_prompts(args.prompts_jsonl, args.model_dir, config["vocab_size"])
    if any(len(ids) + args.reference_steps > config["max_position_embeddings"] for ids in prompts):
        parser.error("Prompt plus reference steps exceeds model context size")
    if not torch.cuda.is_available():
        parser.error("The Qwen3-30B reference exporter requires CUDA")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map="balanced", attn_implementation="eager",
    ).eval()
    with prepare_directory(args.output) as staging:
        rows = []
        for i, ids in enumerate(prompts):
            rows.append(export_prompt(model, ids, staging / kv_name(i), args.reference_steps))
            print(f"Exported prompt {i + 1}/{len(prompts)} ({len(ids)} tokens)", flush=True)
        manifest = {"format": FORMAT, "checkpoint_metadata_sha256": digests, "repeat": args.repeat,
                    "transformers_version": transformers.__version__, "torch_version": str(torch.__version__),
                    "reference_backend": "eager-bfloat16-greedy", "prompts": rows}
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(rows) * args.repeat} requests in {args.output}")


if __name__ == "__main__":
    main()
