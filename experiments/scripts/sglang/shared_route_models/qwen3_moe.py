"""Replay StreamInfer's request/step/layer indexed routes on SGLang's gathered MoE rows."""
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from sglang.srt.layers.dp_attention import dp_gather_partial
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM as NativeQwen3Moe
from sglang.srt.server_args import get_global_server_args


FIRST_MEASURED_ROOM = 11
PROMPT_LENGTH = 64


@triton.jit(do_not_specialize=["layer", "num_rows"])
def _lookup(table, keys, ids, weights, rows: tl.constexpr, steps: tl.constexpr,
            layers: tl.constexpr, layer, num_rows, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = i // 8
    key = tl.load(keys + token, mask=token < num_rows, other=-1)
    # Benchmark warmup uses its own request/step ordinals; its keys are signed to
    # exclude it from the measured audit. Distribute padding over the expert pool.
    key = tl.where(key == -1, (token * 8191) % (rows * steps),
                   tl.where(key < -1, -2 - key, key))
    flat = (key * layers + layer) * 8 + i % 8
    experts = tl.load(table + flat, mask=token < num_rows, other=0)
    tl.store(ids + i, experts.to(tl.int32), mask=token < num_rows)
    tl.store(weights + i, 0.125, mask=token < num_rows)


@triton.jit(do_not_specialize=["layer", "num_rows"])
def _audit_ids(keys, ids, audit, num_rows, steps: tl.constexpr, layer,
               audit_reqs: tl.constexpr, audit_steps: tl.constexpr, layers: tl.constexpr,
               BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = i // 8
    key = tl.load(keys + token, mask=token < num_rows, other=-1)
    request = key // steps
    step = key % steps
    valid = (token < num_rows) & (key >= 0) & (request < audit_reqs) & (step < audit_steps)
    actual = tl.load(ids + i, mask=valid, other=-1)
    dst = ((request * audit_steps + step) * layers + layer) * 8 + i % 8
    tl.store(audit + dst, actual, mask=valid)


class SharedRoutes:
    def __init__(self, config):
        path = Path(os.environ["STREAMEP_SHARED_TOP8"]).resolve(strict=True)
        source = np.load(path, mmap_mode="r", allow_pickle=False)
        if (source.ndim != 4 or source.shape[2:] != (config.num_hidden_layers, 8)
                or source.dtype != np.uint8 or not source.flags.c_contiguous
                or not all(source.shape) or int(source.max()) >= config.num_experts
                or config.num_experts != 128
                or config.num_experts_per_tok != 8):
            raise ValueError(f"Invalid Qwen shared Top-8 table: {source.shape}, {source.dtype}")
        self.rows, self.steps, self.layers, _ = source.shape
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.table = torch.tensor(source, device=self.device)
        self.audit = None
        self.audit_dir = os.environ.get("STREAMEP_ROUTE_AUDIT_DIR")
        if self.audit_dir:
            Path(self.audit_dir).mkdir(exist_ok=True, parents=True)
            self.audit = torch.full((min(4, self.rows), min(4, self.steps), self.layers, 8),
                                    -1, dtype=torch.int32, device=self.device)
        self.keys = None
        print(f"SHARED_TOP8 SGLang loaded {path} shape={source.shape} gpu={self.device}"
              f" audit={bool(self.audit_dir)}", flush=True)

    def ordinal(self, req):
        room = req.bootstrap_room
        if room is not None and 1 <= room < FIRST_MEASURED_ROOM and len(req.origin_input_ids) == 100:
            return -2 - ((room - 1) % self.rows)
        if room is None or room < FIRST_MEASURED_ROOM:
            return -1
        # SGLang's built-in PD warmup uses length-four prompts and int64 sentinel rooms.
        if room >= 2**60 and len(req.origin_input_ids) == 4:
            return -1
        ordinal = room - FIRST_MEASURED_ROOM
        if ordinal >= self.rows or len(req.origin_input_ids) != PROMPT_LENGTH:
            raise ValueError(f"Unexpected measured room {room} or prompt length "
                             f"{len(req.origin_input_ids)}")
        if req.sampling_params.max_new_tokens > self.steps + 1:
            raise ValueError("Measured output length exceeds shared Top-8 trace")
        return ordinal

    def bind_pool(self):
        # SGLang owns the slot allocator. Record the client-provided request identity
        # when a decode slot is allocated/reused; the captured graph reads a live tensor.
        from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
        alloc = DecodeReqToTokenPool.alloc
        free = DecodeReqToTokenPool.free
        state = self

        def allocate(pool, requests):
            slots = alloc(pool, requests)
            if slots is not None:
                with torch.inference_mode():
                    table = state.slot_table(pool)
                    for slot, request in zip(slots, requests):
                        table[slot].fill_(state.ordinal(request))
            return slots

        def release(pool, req):
            if state.audit is not None:
                ordinal = state.ordinal(req)
                if 0 <= ordinal < state.audit.size(0):
                    torch.cuda.synchronize(state.device)
                    values = state.audit[ordinal].cpu().tolist()
                    path = Path(state.audit_dir) / f"sglang-req-{ordinal}.jsonl"
                    with path.open("w") as sink:
                        for step, layers in enumerate(values):
                            for layer, experts in enumerate(layers):
                                if experts[0] >= 0:
                                    print(json.dumps(dict(request=ordinal, step=step, layer=layer,
                                                          experts=experts)), file=sink)
            return free(pool, req)

        DecodeReqToTokenPool.alloc = allocate
        if self.audit is not None:
            DecodeReqToTokenPool.free = release

    def slot_table(self, pool):
        if not hasattr(pool, "_shared_route_ordinals"):
            pool._shared_route_ordinals = torch.full(
                (pool.req_to_token.shape[0],), -1, dtype=torch.int32, device=self.device)
        return pool._shared_route_ordinals

    def prepare(self, batch):
        ordinals = self.slot_table(batch.req_to_token_pool)[batch.req_pool_indices.long()]
        # seq_lens padding is reset on every replay, unlike the tail of positions/slot indices.
        steps = batch.seq_lens.to(torch.int32) - (PROMPT_LENGTH + 1)
        warmup_steps = batch.seq_lens.to(torch.int32) - 101
        warmup_keys = torch.where((ordinals < -1) & (warmup_steps >= 0),
                                  -2 - ((-2 - ordinals) * self.steps + warmup_steps % self.steps),
                                  -1)
        keys = torch.where((ordinals >= 0) & (steps >= 0) & (steps < self.steps),
                           ordinals * self.steps + steps, warmup_keys).reshape(-1, 1)
        self.keys = torch.empty((batch.global_dp_buffer_len, 1),
                                device=self.device, dtype=keys.dtype)
        # The same MAX_LEN all-gather or SUM_LEN layout as LayerCommunicator.prepare_mlp.
        # One int32 key per row, once per forward, reused by every layer.
        dp_gather_partial(self.keys, keys, batch)


class ReplayTopK(torch.nn.Module):
    def __init__(self, state, layer):
        super().__init__()
        self.state = state
        self.layer = layer

    def forward(self, hidden_states, router_logits):
        state = self.state
        rows = hidden_states.shape[0]
        if rows != state.keys.shape[0]:
            raise ValueError(f"MoE rows {rows} disagree with DP metadata {state.keys.shape}")
        ids = torch.empty((rows, 8), dtype=torch.int32, device=hidden_states.device)
        weights = torch.empty_like(ids, dtype=torch.float32)
        if rows:
            grid = (triton.cdiv(rows * 8, 128),)
            _lookup[grid](state.table, state.keys, ids, weights, state.rows, state.steps,
                          state.layers, self.layer, rows, BLOCK=128)
            if state.audit is not None:
                _audit_ids[grid](state.keys, ids, state.audit, rows, state.steps, self.layer,
                                 state.audit.size(0), state.audit.size(1), state.layers, BLOCK=128)
        return StandardTopKOutput(weights, ids, router_logits)


class Qwen3MoeForCausalLM(NativeQwen3Moe):
    def __init__(self, config, quant_config=None, prefix=""):
        args = get_global_server_args()
        if (version("sglang") != "0.5.9" or args.disaggregation_mode != "decode"
                or args.disaggregation_transfer_backend != "fake" or args.load_format != "dummy"
                or args.tp_size != 4 or args.dp_size != 4 or args.ep_size != 4
                or not args.enable_dp_attention or args.moe_a2a_backend != "none"
                or args.moe_runner_backend != "triton" or args.enable_two_batch_overlap):
            raise ValueError("Shared Top-8 replay requires SGLang 0.5.9 DPA4/EP4 Triton dummy decode")
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.shared_routes = SharedRoutes(config)
        self.shared_routes.bind_pool()
        for layer in self.model.layers:
            layer.mlp.topk = ReplayTopK(self.shared_routes, layer.layer_id)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None):
        self.shared_routes.prepare(forward_batch)
        return super().forward(input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors)


EntryClass = Qwen3MoeForCausalLM
