"""Decode a tiny local Qwen checkpoint against Transformers; stand in for CUDA kernels on CPU."""
from pathlib import Path
from importlib.util import find_spec
import tempfile
from types import SimpleNamespace
import unittest

if any(find_spec(name) is None for name in ("torch", "transformers", "safetensors")):
    raise unittest.SkipTest("Tiny Qwen parity test requires torch, transformers and safetensors")
if find_spec("transformers.models.qwen3_moe") is None:
    raise unittest.SkipTest("Installed Transformers has no Qwen3-MoE implementation")

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm, apply_rotary_pos_emb

from disagmoe.models.qwen_checkpoint import QwenCheckpoint, attention_targets, expert_targets
from disagmoe.models.qwen_decode import qwen_attention


class Linear(nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class CPULayer:
    """vLLM-facing tensor layouts, Hugging Face RoPE, and dense one-token CPU attention."""
    def __init__(self, config, rope):
        self.rope = rope
        self.head_dim = config.head_dim
        self.top_k = config.num_experts_per_tok
        self.q_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.q_size = self.q_heads * self.head_dim
        self.kv_size = self.kv_heads * self.head_dim
        h, i, e = config.hidden_size, config.moe_intermediate_size, config.num_experts

        def linear(rows, cols):
            return Linear(cols, rows, bias=False, dtype=torch.bfloat16)

        def norm(d):
            return Qwen3MoeRMSNorm(d, config.rms_norm_eps).bfloat16()

        self.qkv_proj = linear(self.q_size + 2 * self.kv_size, h)
        self.o_proj = linear(h, self.q_size)
        self.q_norm, self.k_norm = norm(self.head_dim), norm(self.head_dim)
        self.pre_attention_layernorm, self.post_attention_layernorm = norm(h), norm(h)
        self.gate = linear(e, h)
        self.w13_weight = torch.empty(e, h, 2 * i, dtype=torch.bfloat16)
        self.w2_weight = torch.empty(e, i, h, dtype=torch.bfloat16)

    def rotary_emb(self, positions, q, k):
        cos, sin = self.rope(q, positions[None, :])
        q = q.reshape(1, self.q_heads, 1, self.head_dim)
        k = k.reshape(1, self.kv_heads, 1, self.head_dim)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q.reshape(1, -1), k.reshape(1, -1)

    def attn(self, q, k, v, kv_cache, attn_metadata):
        del attn_metadata
        for idx, value in enumerate((k, v)):
            kv_cache[idx] = torch.cat((kv_cache[idx], value.view(1, self.kv_heads, 1, -1)), dim=2)
        keys, values = (x.repeat_interleave(self.q_heads // self.kv_heads, dim=1) for x in kv_cache)
        q = q.reshape(1, self.q_heads, 1, self.head_dim)
        probs = torch.softmax((q @ keys.transpose(-1, -2)) * self.head_dim**-0.5,
                              dim=-1, dtype=torch.float32).to(q.dtype)
        return (probs @ values).reshape(1, -1)

    def experts(self, x, weights, ids):
        parts = []
        for weight, expert in zip(weights[0], ids[0]):
            gate, up = (x @ self.w13_weight[expert]).chunk(2, dim=-1)
            parts.append(((F.silu(gate) * up) @ self.w2_weight[expert]) * weight)
        return torch.stack(parts).float().sum(dim=0).to(x.dtype)


class TestTinyDecodeParity(unittest.TestCase):
    @torch.inference_mode()
    def test_two_decode_steps_from_prefill(self):
        torch.manual_seed(728)
        cfg = Qwen3MoeConfig(vocab_size=83, hidden_size=32, num_hidden_layers=2, head_dim=8,
                             num_attention_heads=4, num_key_value_heads=2, num_experts=4,
                             num_experts_per_tok=2, moe_intermediate_size=12, intermediate_size=16,
                             max_position_embeddings=64, rope_theta=1000000., norm_topk_prob=True)
        cfg._attn_implementation = "eager"
        hf = Qwen3MoeForCausalLM(cfg).bfloat16().eval()
        prompt = torch.tensor([[1, 30, 9, 23]])
        initial = hf(prompt, use_cache=True)
        past = initial.past_key_values
        cache = [[k.clone(), v.clone()] for k, v in (past[i] for i in range(cfg.num_hidden_layers))]
        token = initial.logits[:, -1].argmax(-1)
        layers = [CPULayer(cfg, hf.model.rotary_emb) for _ in range(cfg.num_hidden_layers)]
        embedding = torch.empty_like(hf.model.embed_tokens.weight)
        final_norm = Qwen3MoeRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).bfloat16()
        head = torch.empty_like(hf.lm_head.weight)
        targets = {"model.embed_tokens.weight": embedding, "model.norm.weight": final_norm.weight,
                   "lm_head.weight": head}
        for i, layer in enumerate(layers):
            targets.update(attention_targets(i, layer))
            targets.update(expert_targets(i, layer, range(cfg.num_experts)))
        with tempfile.TemporaryDirectory() as folder:
            hf.save_pretrained(folder, safe_serialization=True)
            runtime = SimpleNamespace(hidden_size=cfg.hidden_size, head_dim=cfg.head_dim,
                                      num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads,
                                      num_layers=len(layers), num_experts=cfg.num_experts, top_k=cfg.num_experts_per_tok,
                                      intermediate_size=cfg.moe_intermediate_size, tp_size=1, dtype=torch.bfloat16,
                                      num_shared_experts=0, moe_linear_quant=None, attn_qkv_quant=None)
            QwenCheckpoint(Path(folder), runtime).load(targets)

        for pos in (4, 5):
            out = hf(token[:, None], past_key_values=past, use_cache=True, output_router_logits=True)
            past = out.past_key_values
            x = F.embedding(token, embedding)
            for i, layer in enumerate(layers):
                z, weights, ids, residual = qwen_attention(layer, torch.tensor([pos]), x, cache[i], None)
                expected_ids = out.router_logits[i].float().softmax(-1).topk(cfg.num_experts_per_tok).indices
                torch.testing.assert_close(ids, expected_ids)
                x = residual + layer.experts(z, weights, ids)
                for actual, expected in zip(cache[i], past[i]):
                    torch.testing.assert_close(actual.float(), expected.float(), atol=0.02, rtol=0.02)
            logits = F.linear(final_norm(x), head)
            torch.testing.assert_close(logits.float(), out.logits[:, -1].float(), atol=0.005, rtol=0.02)
            token = logits.argmax(-1)
            self.assertEqual(token.item(), out.logits[:, -1].argmax(-1).item())


if __name__ == "__main__":
    unittest.main()
