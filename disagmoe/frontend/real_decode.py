"""Request state on the attention owner during eager, one-token-at-a-time decode."""

import torch


class DecodeResiduals:
    """Keep one post-attention residual per request until all its experts return."""

    def __init__(self, req_ids, hidden_size, device, capacity=None):
        capacity = len(req_ids) if capacity is None else capacity
        if capacity < len(req_ids):
            raise ValueError("Residual capacity is smaller than the initial cohort")
        self.rows = {rid: row for row, rid in enumerate(req_ids)}
        if len(self.rows) != len(req_ids):
            raise ValueError("Duplicate request in decode cohort")
        self.pending_layer = {}
        self.free_rows = list(range(capacity - 1, len(req_ids) - 1, -1))
        self.buffer = torch.empty((capacity, hidden_size), device=device, dtype=torch.bfloat16)

    def admit(self, req_id):
        if req_id in self.rows:
            raise RuntimeError(f"Duplicate decode request {req_id}")
        if not self.free_rows:
            raise RuntimeError("No residual row available for an admitted decode request")
        self.rows[req_id] = self.free_rows.pop()

    def release(self, req_ids):
        for rid in req_ids:
            if rid in self.pending_layer:
                raise RuntimeError(f"Request {rid} finished with a pending residual")
            self.free_rows.append(self.rows.pop(rid))

    def indices(self, req_ids):
        if len(set(req_ids)) != len(req_ids):
            raise RuntimeError("Multiple decode tokens from one request in an attention batch")
        return torch.tensor([self.rows[rid] for rid in req_ids], device=self.buffer.device,
                            dtype=torch.long)

    def save(self, layer, req_ids, indices, residual):
        for rid in req_ids:
            if rid in self.pending_layer:
                raise RuntimeError(f"Request {rid} still has a pending residual")
        self.buffer.index_copy_(0, indices, residual)
        self.pending_layer.update(dict.fromkeys(req_ids, layer))

    def add_after_experts(self, layer, req_ids, indices, expert_outputs):
        for rid in req_ids:
            if self.pending_layer.get(rid) != layer:
                raise RuntimeError(f"Missing residual for request {rid} at layer {layer}")
        hidden = expert_outputs + self.buffer.index_select(0, indices)
        for rid in req_ids:
            del self.pending_layer[rid]
        return hidden


def restore_prompt_kv(snapshot_kv, layer_buffers, block_table):
    """Scatter one request's logical prompt into the block manager's physical pages.

    The last logical page can be partial. The next decode slot and all pages
    belonging to other requests retain their previous contents.
    """
    if not layer_buffers or snapshot_kv.ndim != 5:
        raise ValueError("KV import requires per-layer [2, tokens, heads, head_dim] tensors")
    first = layer_buffers[0]
    layers, kv, length, heads, width = snapshot_kv.shape
    if (layers != len(layer_buffers) or kv != 2 or length < 1 or
            first.ndim != 5 or first.shape[0] != 2 or first.shape[-2:] != (heads, width) or
            snapshot_kv.dtype != first.dtype):
        raise ValueError(f"Invalid KV shapes or dtype: snapshot {snapshot_kv.shape}, "
                         f"cache {first.shape}, layers {len(layer_buffers)}")
    page_size = first.shape[2]
    pages_needed = (length + page_size - 1) // page_size
    pages = block_table[:pages_needed]
    if (len(pages) != pages_needed or len(set(pages)) != pages_needed
            or min(pages) < 0 or max(pages) >= first.shape[1]):
        raise ValueError("Invalid physical block table for snapshot prompt")

    # One H2D for the request, followed by device-side writes to the paged caches.
    data = snapshot_kv.to(first.device)
    pages = torch.tensor(pages, device=first.device, dtype=torch.long)
    positions = torch.arange(length, device=first.device)
    slots = pages[positions // page_size] * page_size + positions % page_size
    for i, buffer in enumerate(layer_buffers):
        buffer.view(2, -1, heads, width).index_copy_(1, slots, data[i])
