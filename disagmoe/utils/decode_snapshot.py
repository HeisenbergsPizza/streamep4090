"""Reader for offline Qwen prefill states in logical (unpadded) token order."""

import hashlib
import json
from pathlib import Path


FORMAT = "streaminfer.qwen3-decode.v1"


def checkpoint_metadata(model_dir):
    """Fingerprint the configuration and shard map; does not hash weight payloads."""
    root = Path(model_dir)
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("config.json", "model.safetensors.index.json")}


def kv_name(row):
    return f"kv-{row:06d}.safetensors"


def validate_token_ids(ids, vocab_size, name, allow_empty=False):
    if not isinstance(ids, list) or (not ids and not allow_empty) or any(
        type(token) is not int or token < 0 or token >= vocab_size for token in ids
    ):
        raise ValueError(f"{name} must contain token IDs in [0, {vocab_size})")


class DecodeSnapshot:
    """The first token is already selected by prefill. expected_tokens starts at y1."""

    def __init__(self, snapshot_dir, model_dir):
        self.root = Path(snapshot_dir)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest.get("format") != FORMAT:
            raise ValueError("Unsupported Qwen decode snapshot format")
        if manifest.get("checkpoint_metadata_sha256") != checkpoint_metadata(model_dir):
            raise ValueError("Checkpoint configuration or safetensors index differs from the snapshot")
        config = json.loads((Path(model_dir) / "config.json").read_text())
        if config["model_type"] != "qwen3_moe":
            raise ValueError("Snapshot requires a Qwen3 MoE checkpoint")
        self.layers = config["num_hidden_layers"]
        self.heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.repeat = manifest["repeat"]
        self.rows = manifest["prompts"]
        self._host_kv = None
        if type(self.repeat) is not int or self.repeat < 1:
            raise ValueError("Invalid snapshot repeat count")
        if not isinstance(self.rows, list) or not self.rows:
            raise ValueError("Snapshot must contain at least one prompt")
        for i, row in enumerate(self.rows):
            validate_token_ids(row["input_ids"], config["vocab_size"], "input_ids")
            validate_token_ids([row["first_token"]], config["vocab_size"], "first_token")
            validate_token_ids(row["expected_tokens"], config["vocab_size"], "expected_tokens", True)
            if not (self.root / kv_name(i)).is_file():
                raise ValueError(f"Snapshot KV file missing for prompt {i}")

    def __len__(self):
        return len(self.rows) * self.repeat

    def _row_index(self, i):
        if type(i) is not int or not 0 <= i < len(self):
            raise IndexError(f"Snapshot request index {i} out of range")
        return i // self.repeat

    def prompt_ids(self, i):
        return list(self.rows[self._row_index(i)]["input_ids"])

    def prompt_len(self, i):
        return len(self.rows[self._row_index(i)]["input_ids"])

    def first_token(self, i):
        return self.rows[self._row_index(i)]["first_token"]

    def expected_tokens(self, i):
        return list(self.rows[self._row_index(i)]["expected_tokens"])

    def stage_kv_on_host(self):
        """Read each unique prompt's KV into owned CPU storage before timed arrivals."""
        if self._host_kv is None:
            # A safetensors tensor can be backed by a file mapping. Cloning reads
            # the shared mount now and leaves ordinary host memory for admission.
            self._host_kv = [self.load_kv(i).clone() for i in range(0, len(self), self.repeat)]
        return sum(kv.numel() * kv.element_size() for kv in self._host_kv)

    def load_kv(self, i):
        """Load one request on CPU as [layer, K/V, position, KV head, head dim]."""
        row = self._row_index(i)
        if self._host_kv is not None:
            return self._host_kv[row]
        import torch
        from safetensors import safe_open

        path = self.root / kv_name(row)
        shape = (self.layers, 2, self.prompt_len(i), self.heads, self.head_dim)
        with safe_open(path, framework="pt", device="cpu") as f:
            if f.keys() != ["kv"] or tuple(f.get_slice("kv").get_shape()) != shape:
                raise ValueError(f"Invalid KV layout in {path}, expected a tensor of shape {shape}")
            kv = f.get_tensor("kv")
        if kv.dtype != torch.bfloat16:
            raise ValueError(f"Expected bfloat16 KV in {path}, found {kv.dtype}")
        return kv


def compare_decode_results(snapshot, index_by_request, results, decode_steps):
    """Compare only reference steps available, and retain the full measured continuation."""
    requests = []
    mismatches = 0
    expected_indices = set(range(len(snapshot)))
    if (len(index_by_request) != len(snapshot)
            or set(index_by_request.values()) != expected_indices):
        raise ValueError("Incomplete mapping of runtime requests to the snapshot")
    if len(results) != len(snapshot) or {r.req_id for r in results} != set(index_by_request):
        raise ValueError("Decode results do not match the prepared cohort")
    for result in results:
        i = index_by_request[result.req_id]
        tokens = result.token_ids
        if tokens is None or len(tokens) != decode_steps:
            raise ValueError(f"Request {result.req_id} did not return {decode_steps} decode tokens")
        expected = snapshot.expected_tokens(i)[:decode_steps]
        if not expected:
            raise ValueError("Export at least one greedy reference step for every prompt")
        first_difference = next((j for j, exp in enumerate(expected) if tokens[j] != exp), None)
        mismatches += first_difference is not None
        requests.append(dict(snapshot_index=i, req_id=result.req_id, prefill_token=snapshot.first_token(i),
                             token_ids=tokens, reference_prefix=expected,
                             first_different_step=first_difference))
    return dict(decode_steps=decode_steps, mismatched_requests=mismatches,
                matched_reference_prefix=mismatches == 0,
                requests=sorted(requests, key=lambda r: r["snapshot_index"]))
