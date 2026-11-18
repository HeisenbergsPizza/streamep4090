"""Exercise real sharded checkpoint layout on CPU, including an EP-local expert."""
import json
from importlib.util import find_spec
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

if any(find_spec(name) is None for name in ("torch", "safetensors")):
    raise unittest.SkipTest("Qwen checkpoint layout test requires torch and safetensors")

import torch
from safetensors.torch import save_file

from disagmoe.models.qwen_checkpoint import QwenCheckpoint, attention_targets, expert_targets


class TestQwenCheckpoint(unittest.TestCase):
    def test_sharded_weights_match_hf_projection_math(self):
        dtype = torch.bfloat16
        h, i = 4, 3
        model = SimpleNamespace(hidden_size=h, head_dim=2, num_heads=2, num_kv_heads=1,
                                num_layers=1, num_experts=4, top_k=2, intermediate_size=i,
                                tp_size=1, dtype=dtype, num_shared_experts=0,
                                moe_linear_quant=None, attn_qkv_quant=None)
        config = dict(model_type="qwen3_moe", hidden_size=h, head_dim=2,
                      num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=1,
                      num_experts=4, num_experts_per_tok=2, moe_intermediate_size=i,
                      hidden_act="silu", norm_topk_prob=True, tie_word_embeddings=False,
                      rms_norm_eps=1e-6, rope_theta=10_000, max_position_embeddings=32, vocab_size=8)

        def param(*shape):
            return SimpleNamespace(weight=torch.empty(shape, dtype=dtype))

        op = SimpleNamespace(q_size=4, kv_size=2, qkv_proj=param(8, h), o_proj=param(h, 4),
                             q_norm=param(2), k_norm=param(2), pre_attention_layernorm=param(h),
                             post_attention_layernorm=param(h), gate=param(4, h))
        experts = SimpleNamespace(w13_weight=torch.empty(1, h, 2 * i, dtype=dtype),
                                  w2_weight=torch.empty(1, i, h, dtype=dtype))
        targets = attention_targets(0, op)
        targets.update(expert_targets(0, experts, [3]))
        gen = torch.Generator().manual_seed(718)
        tensors = {key: torch.randn(view.shape, dtype=dtype, generator=gen) for key, view in targets.items()}
        ptr = experts.w13_weight.data_ptr()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(config))
            keys = list(tensors)
            index = {}
            for shard_idx in (0, 1):
                filename = f"shard{shard_idx}.safetensors"
                shard = {key: tensors[key] for key in keys[shard_idx::2]}
                save_file(shard, root / filename)
                index.update({key: filename for key in shard})
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
            loader = QwenCheckpoint(root, model)
            loader.load(targets)
            with self.assertRaisesRegex(ValueError, "missing tensor"):
                loader.load({"model.missing.weight": torch.empty(1)})

        x = torch.randn((5, h), dtype=torch.float32, generator=gen)
        prefix = "model.layers.0.self_attn."
        qkv = [x @ tensors[prefix + name + "_proj.weight"].float().T for name in ("q", "k", "v")]
        torch.testing.assert_close(x @ op.qkv_proj.weight.float().T, torch.cat(qkv, dim=-1))

        pre = "model.layers.0.mlp.experts.3."
        gate = x @ tensors[pre + "gate_proj.weight"].float().T
        up = x @ tensors[pre + "up_proj.weight"].float().T
        reference = (torch.nn.functional.silu(gate) * up) @ tensors[pre + "down_proj.weight"].float().T
        gate, up = (x @ experts.w13_weight[0].float()).chunk(2, dim=-1)
        actual = (torch.nn.functional.silu(gate) * up) @ experts.w2_weight[0].float()
        torch.testing.assert_close(actual, reference)
        self.assertEqual(ptr, experts.w13_weight.data_ptr())


if __name__ == "__main__":
    unittest.main()
