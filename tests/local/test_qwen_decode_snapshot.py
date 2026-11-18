"""Compare exported incremental decode with a full forward on a tiny CPU Qwen MoE."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from disagmoe.utils.decode_snapshot import FORMAT, DecodeSnapshot, checkpoint_metadata, kv_name
from experiments.scripts.rtx4090.prepare_qwen_decode import export_prompt, load_prompts, prepare_directory


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.config = dict(model_type="qwen3_moe", num_hidden_layers=2, num_key_value_heads=1,
                           head_dim=8, vocab_size=32)
        (self.model / "config.json").write_text(json.dumps(self.config))
        (self.model / "model.safetensors.index.json").write_text('{"weight_map":{}}')

    def write_manifest(self, row, repeat=2):
        manifest = {"format": FORMAT, "checkpoint_metadata_sha256": checkpoint_metadata(self.model),
                    "repeat": repeat, "prompts": [row]}
        (self.root / "manifest.json").write_text(json.dumps(manifest))

    def test_snapshot_mismatch_and_input_validation(self):
        self.write_manifest(dict(input_ids=[1, 2], first_token=3, expected_tokens=[4]))
        with self.assertRaisesRegex(ValueError, "KV file missing"):
            DecodeSnapshot(self.root, self.model)
        (self.model / "model.safetensors.index.json").write_text('{"weight_map":{"w":"other"}}')
        with self.assertRaisesRegex(ValueError, "differs"):
            DecodeSnapshot(self.root, self.model)
        path = self.root / "ids.jsonl"
        path.write_text('{"input_ids":[true]}\n')
        with self.assertRaisesRegex(ValueError, "token IDs"):
            load_prompts(path, self.model, 32)

    def test_failed_export_can_retry_and_never_replace_completed_snapshot(self):
        dest = self.root / "run"
        with self.assertRaisesRegex(RuntimeError, "device error"):
            with prepare_directory(dest) as staging:
                (staging / kv_name(0)).write_bytes(b"partial")
                raise RuntimeError("device error")
        self.assertFalse(dest.exists())
        self.assertFalse(staging.exists())
        with prepare_directory(dest) as staging:
            (staging / kv_name(0)).write_bytes(b"complete")
            (staging / "manifest.json").write_text("{}")
        with self.assertRaises(FileExistsError):
            with prepare_directory(dest) as staging:
                (staging / "manifest.json").write_text("other")
        self.assertEqual((dest / "manifest.json").read_text(), "{}")
        self.assertEqual((dest / kv_name(0)).read_bytes(), b"complete")

    @unittest.skipUnless(all(importlib.util.find_spec(x) for x in ("torch", "safetensors", "transformers")),
                         "Reference test requires torch, safetensors and Qwen3-capable transformers")
    def test_tiny_qwen_prefill_decode_round_trip(self):
        import torch
        from safetensors.torch import save_file
        try:
            from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        except ImportError:
            self.skipTest("Reference test requires Qwen3-capable transformers")
        torch.manual_seed(41)
        config = Qwen3MoeConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                                num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
                                vocab_size=32, max_position_embeddings=64, attn_implementation="eager")
        model = Qwen3MoeForCausalLM(config).to(dtype=torch.bfloat16).eval()
        prompt = [1, 2, 5, 11]
        row = export_prompt(model, prompt, self.root / kv_name(0), reference_steps=2)
        self.write_manifest(row)
        snapshot = DecodeSnapshot(self.root, self.model)
        self.assertEqual(len(snapshot), 2)
        self.assertEqual(snapshot.prompt_ids(1), prompt)
        self.assertEqual(snapshot.prompt_len(0), len(prompt))
        self.assertRaises(IndexError, snapshot.first_token, -1)
        self.assertRaises(IndexError, snapshot.load_kv, 2)
        kv = snapshot.load_kv(1)
        from transformers import DynamicCache
        imported = DynamicCache()
        for layer in range(config.num_hidden_layers):
            k, v = (t.transpose(0, 1).unsqueeze(0) for t in kv[layer])
            imported.update(k, v, layer)
        with torch.inference_mode():
            ids = torch.tensor([prompt])
            token = snapshot.first_token(1)
            self.assertEqual(model(ids).logits[0, -1].argmax().item(), token)
            for expected in snapshot.expected_tokens(0):
                current = torch.tensor([[token]])
                ids = torch.cat([ids, current], dim=1)
                incremental = model(current, past_key_values=imported, use_cache=True)
                imported = incremental.past_key_values
                expected_logits = model(ids).logits[0, -1]
                torch.testing.assert_close(incremental.logits[0, -1], expected_logits,
                                           atol=0.004, rtol=0.02)
                token = incremental.logits[0, -1].argmax().item()
                self.assertEqual(token, expected)
        save_file({"kv": kv[..., :4].contiguous()}, str(self.root / kv_name(0)))
        with self.assertRaisesRegex(ValueError, "KV layout"):
            snapshot.load_kv(0)


if __name__ == "__main__":
    unittest.main()
