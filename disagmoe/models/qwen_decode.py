"""Qwen3-MoE attention and routing for one decode token per request."""
import torch


def qwen_attention(layer, positions, hidden_states, kv_cache, metadata):
    normalized = layer.pre_attention_layernorm(hidden_states)
    packed, _ = layer.qkv_proj(normalized)
    q, k, v = packed.split((layer.q_size, layer.kv_size, layer.kv_size), dim=-1)
    q = layer.q_norm(q.reshape(-1, layer.head_dim)).reshape(-1, layer.q_size)
    k = layer.k_norm(k.reshape(-1, layer.head_dim)).reshape(-1, layer.kv_size)
    q, k = layer.rotary_emb(positions, q, k)
    attention = layer.attn(q, k, v, kv_cache=kv_cache, attn_metadata=metadata)
    projected, _ = layer.o_proj(attention)
    # This row stays on its attention owner until the expert branches return.
    residual = hidden_states + projected
    expert_input = layer.post_attention_layernorm(residual)
    logits, _ = layer.gate(expert_input)
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    weights, ids = torch.topk(probabilities, layer.top_k, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(expert_input.dtype)
    return expert_input, weights, ids, residual
