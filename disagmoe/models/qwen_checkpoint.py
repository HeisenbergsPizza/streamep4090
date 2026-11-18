"""Load Hugging Face Qwen3-MoE BF16 safetensors into StreamInfer's TP1 layout."""
from collections import defaultdict
import json
from pathlib import Path

import torch


class QwenCheckpoint:
    def __init__(self, directory, model):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "config.json").read_text())
        expected = {
            "model_type": "qwen3_moe",
            "hidden_size": model.hidden_size,
            "head_dim": model.head_dim,
            "num_attention_heads": model.num_heads,
            "num_key_value_heads": model.num_kv_heads,
            "num_hidden_layers": model.num_layers,
            "num_experts": model.num_experts,
            "num_experts_per_tok": model.top_k,
            "moe_intermediate_size": model.intermediate_size,
            "hidden_act": "silu",
            "norm_topk_prob": True,
            "tie_word_embeddings": False,
        }
        for key, value in expected.items():
            if self.config.get(key) != value:
                raise ValueError(f"Qwen checkpoint {key}={self.config.get(key)!r}, expected {value!r}")
        if (self.config.get("rope_scaling") or self.config.get("use_sliding_window")
                or self.config.get("attention_bias") or self.config.get("quantization_config")
                or self.config.get("mlp_only_layers") or self.config.get("decoder_sparse_step", 1) != 1):
            raise ValueError("Unsupported Qwen checkpoint (requires full attention and a BF16 sparse MoE on every layer)")
        for key in ("rms_norm_eps", "rope_theta", "max_position_embeddings", "vocab_size"):
            if self.config.get(key, 0) <= 0:
                raise ValueError(f"Invalid Qwen checkpoint {key}")
        if (model.tp_size != 1 or model.dtype != torch.bfloat16 or model.num_shared_experts
                or model.moe_linear_quant not in (None, "none")
                or model.attn_qkv_quant not in (None, "none")):
            raise ValueError("Qwen decode requires TP1, BF16, and no shared experts or quantization")

        index = self.directory / "model.safetensors.index.json"
        self.weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else None

    @torch.no_grad()
    def load(self, targets):
        """Read only selected tensors. Preserve existing storage, including CUTLASS runner pointers."""
        from safetensors import safe_open
        shards = defaultdict(list)
        for key in targets:
            if self.weight_map is None:
                shard = "model.safetensors"
            else:
                if key not in self.weight_map:
                    raise ValueError(f"Qwen checkpoint missing tensor {key}")
                shard = self.weight_map[key]
            shards[shard].append(key)

        for shard, keys in shards.items():
            with torch.device("cpu"), safe_open(self.directory / shard, framework="pt", device="cpu") as reader:
                for key in keys:
                    source = reader.get_tensor(key)
                    target = targets[key]
                    if source.shape != target.shape or source.dtype != target.dtype:
                        raise ValueError(
                            f"{key}: checkpoint {tuple(source.shape)} {source.dtype}, "
                            f"runtime {tuple(target.shape)} {target.dtype}"
                        )
                    target.copy_(source)


def attention_targets(layer_id, operator):
    prefix = f"model.layers.{layer_id}."
    q, k, v = operator.qkv_proj.weight.split(
        [operator.q_size, operator.kv_size, operator.kv_size], dim=0
    )
    return {
        prefix + "self_attn.q_proj.weight": q,
        prefix + "self_attn.k_proj.weight": k,
        prefix + "self_attn.v_proj.weight": v,
        prefix + "self_attn.o_proj.weight": operator.o_proj.weight,
        prefix + "self_attn.q_norm.weight": operator.q_norm.weight,
        prefix + "self_attn.k_norm.weight": operator.k_norm.weight,
        prefix + "input_layernorm.weight": operator.pre_attention_layernorm.weight,
        prefix + "post_attention_layernorm.weight": operator.post_attention_layernorm.weight,
        prefix + "mlp.gate.weight": operator.gate.weight,
    }


def expert_targets(layer_id, operator, local_expert_ids):
    """The checkpoint stores [out,in]; CUTLASS packs gate/up in [in,2*out]."""
    targets = {}
    for local_idx, global_id in enumerate(local_expert_ids):
        prefix = f"model.layers.{layer_id}.mlp.experts.{global_id}."
        gate, up = operator.w13_weight[local_idx].chunk(2, dim=-1)
        targets[prefix + "gate_proj.weight"] = gate.T
        targets[prefix + "up_proj.weight"] = up.T
        targets[prefix + "down_proj.weight"] = operator.w2_weight[local_idx].T
    return targets
