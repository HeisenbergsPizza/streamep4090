"""Observe sampled token-layer fork/join paths without adding simulator events.

A source cohort is an attention batch selected inside a half-open time window.
Its tokens remain tracked through the next attention GPU-service start, even
outside that window. Batch IDs are private to this recorder and never modify
the simulator's buffer/reformation provenance.
"""
from __future__ import annotations

from collections import Counter
import gzip
import json
import math
from pathlib import Path


class TokenTraceRecorder:
    def __init__(self, path, *, num_layers, ticks_per_millisecond=10,
                 start_ms=0, end_ms=None, configuration=None):
        if not math.isfinite(ticks_per_millisecond) or ticks_per_millisecond <= 0:
            raise ValueError("ticks_per_millisecond must be finite and positive")
        if not math.isfinite(start_ms) or start_ms < 0:
            raise ValueError("token trace start_ms must be finite and nonnegative")
        if end_ms is not None and (not math.isfinite(end_ms) or end_ms <= start_ms):
            raise ValueError("token trace end_ms must be finite and greater than start_ms")
        self.ticks_per_ms = float(ticks_per_millisecond)
        self.start_ms, self.end_ms = float(start_ms), None if end_ms is None else float(end_ms)
        self.start_ticks = self.start_ms * self.ticks_per_ms
        self.end_ticks = math.inf if end_ms is None else self.end_ms * self.ticks_per_ms
        if not math.isfinite(self.start_ticks) or (end_ms is not None and not math.isfinite(self.end_ticks)):
            raise ValueError("token trace window exceeds the simulation clock range")
        self.num_layers = num_layers
        self.configuration = dict(configuration or {})
        prefix = Path(path).expanduser().resolve()
        self.paths = {"tokens": str(prefix) + ".tokens.jsonl.gz",
                      "batches": str(prefix) + ".batches.jsonl.gz",
                      "metadata": str(prefix) + ".metadata.json"}
        if any(Path(name).exists() for name in self.paths.values()):
            raise FileExistsError(f"token trace output already exists for prefix: {prefix}")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self.pending = {}
        self.batches = {}
        self.counts = Counter(source_batches=0, sampled_token_layers=0,
                              complete_token_layers=0, censored_token_layers=0,
                              tracked_branches=0, tracked_batches=0, completed_batches=0,
                              censored_batches=0)
        self.next_batch_id = 0
        self.last_ticks = 0.0
        self._metadata = None
        self._tokens = self._batch_file = None
        self._write_metadata(self._make_metadata("recording", False, 0, 0))
        try:
            self._tokens = gzip.open(self.paths["tokens"], "xt", encoding="utf-8")
            self._batch_file = gzip.open(self.paths["batches"], "xt", encoding="utf-8")
        except Exception:
            self.close(reason="artifact_open_failed")
            raise

    def _ms(self, ticks):
        return float(ticks) / self.ticks_per_ms

    def _observe(self, now_ticks):
        self.last_ticks = max(self.last_ticks, float(now_ticks))

    @staticmethod
    def _write(handle, record):
        handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")

    def batch_selected(self, trace, items):
        """Called after selection, before the existing service generator runs."""
        self._observe(trace["start"])
        batch_id = self.next_batch_id
        self.next_batch_id += 1
        stage, layer = trace["stage"], trace["layer"]
        source = (stage == "attention" and layer < self.num_layers
                  and self.start_ticks <= trace["start"] < self.end_ticks)
        if stage == "expert":
            relevant = any((item.token.tid, layer) in self.pending for item, _ in items)
        else:
            relevant = any((item.tid, layer - 1) in self.pending for item, _ in items)
        if not source and not relevant:
            return None
        loads = trace.get("expert_counts")
        batch = {
            "batch_id": batch_id, "gpu": trace["gpu"], "stage": stage, "layer": layer,
            "batch_size": trace["batch_size"],
            "active_experts": sum(count > 0 for count in loads) if loads is not None else 0,
            "selection_ms": self._ms(trace["start"]),
            "gpu_start_ms": self._ms(trace.get("gpu_start", trace["start"])),
            "compute_start_ms": self._ms(trace.get("compute_start", trace["start"])),
            "compute_end_ms": self._ms(trace.get("compute_end", trace["end"])),
            "actual_gpu_start_ms": None, "actual_completion_ms": None,
        }
        members = [item.token.tid if stage == "expert" else item.tid for item, _ in items]
        batch_record = dict(batch, token_ids=members,
                            expert_ids=([item.expert_id for item, _ in items] if stage == "expert" else None),
                            source_cohort=source)
        state = {"record": batch_record, "references": [], "next_keys": []}
        self.batches[batch_id] = state
        self.counts["tracked_batches"] += 1
        if source:
            self.counts["source_batches"] += 1
        for item, ready_at in items:
            if stage == "expert":
                record = self.pending.get((item.token.tid, layer))
                if record is None:
                    continue
                branch = self._branch(record, item.expert_id)
                if branch["expert_batch"] is not None:
                    raise RuntimeError("token trace branch selected twice")
                branch["expert_batch"] = dict(batch)
                state["references"].append(branch["expert_batch"])
                continue
            previous_key = (item.tid, layer - 1)
            previous = self.pending.get(previous_key)
            if previous is not None:
                if previous["next_attention"] is not None:
                    raise RuntimeError("token trace next attention selected twice")
                previous["next_attention"] = dict(batch, ready_at_ms=self._ms(ready_at))
                state["references"].append(previous["next_attention"])
                state["next_keys"].append(previous_key)
            if source:
                key = (item.tid, layer)
                if key in self.pending:
                    raise RuntimeError("token trace source attention selected twice")
                attention = dict(batch, ready_at_ms=self._ms(ready_at))
                self.pending[key] = {
                    "token_id": item.tid, "request_id": item.request_id,
                    "token_index": item.token_index, "home_gpu": item.home_gpu, "layer": layer,
                    "source_attention": attention, "route_expert_ids": None,
                    "branches": [], "join_ready_ms": None, "next_attention": None,
                }
                state["references"].append(attention)
                self.counts["sampled_token_layers"] += 1
        return batch_id

    def batch_gpu_started(self, batch_id, now_ticks):
        if batch_id is None:
            return
        self._observe(now_ticks)
        state = self.batches[batch_id]
        now = self._ms(now_ticks)
        state["record"]["actual_gpu_start_ms"] = now
        for reference in state["references"]:
            reference["actual_gpu_start_ms"] = now
        for key in state["next_keys"]:
            record = self.pending[key]
            if (record["route_expert_ids"] is None or record["join_ready_ms"] is None
                    or any(branch["return_arrival_ms"] is None
                           or branch["expert_batch"] is None
                           or branch["expert_batch"]["actual_completion_ms"] is None
                           for branch in record["branches"])):
                raise RuntimeError("token trace next attention started before complete fork/join")
            self._write(self._tokens, dict(record, status="complete", observed_until_ms=now))
            self.counts["complete_token_layers"] += 1
            del self.pending[key]

    def batch_completed(self, batch_id, now_ticks):
        if batch_id is None:
            return
        self._observe(now_ticks)
        state = self.batches.pop(batch_id)
        now = self._ms(now_ticks)
        state["record"]["actual_completion_ms"] = now
        for reference in state["references"]:
            reference["actual_completion_ms"] = now
        self._write(self._batch_file, dict(state["record"], status="complete"))
        self.counts["completed_batches"] += 1

    def routed(self, token, layer, expert_ids, experts_per_gpu, now_ticks):
        self._observe(now_ticks)
        record = self.pending.get((token.tid, layer))
        if record is None:
            return
        if record["route_expert_ids"] is not None:
            raise RuntimeError("token trace token-layer routed twice")
        record["route_expert_ids"] = list(expert_ids)
        record["branches"] = [
            {"expert_id": expert, "expert_gpu": expert // experts_per_gpu,
             "ready_at_ms": None, "expert_batch": None, "return_arrival_ms": None}
            for expert in expert_ids]
        self.counts["tracked_branches"] += len(expert_ids)

    @staticmethod
    def _branch(record, expert_id):
        for branch in record["branches"]:
            if branch["expert_id"] == expert_id:
                return branch
        raise RuntimeError("token trace encountered an unrouted expert")

    def expert_ready(self, branch, now_ticks):
        self._observe(now_ticks)
        record = self.pending.get((branch.token.tid, branch.layer))
        if record is not None:
            self._branch(record, branch.expert_id)["ready_at_ms"] = self._ms(now_ticks)

    def returned(self, branch, now_ticks):
        self._observe(now_ticks)
        record = self.pending.get((branch.token.tid, branch.layer))
        if record is not None:
            self._branch(record, branch.expert_id)["return_arrival_ms"] = self._ms(now_ticks)

    def joined(self, token, layer, now_ticks):
        self._observe(now_ticks)
        record = self.pending.get((token.tid, layer))
        if record is not None:
            record["join_ready_ms"] = self._ms(now_ticks)

    def _make_metadata(self, status, fully_drained, pending, pending_batches, reason=None):
        return {
            "format": "streaminfer.token_trace", "version": 1, "time_unit": "ms",
            "paths": self.paths, "status": status, "complete": status == "complete",
            "fully_drained": fully_drained, "simulation_end_ms": self._ms(self.last_ticks),
            "requested_window_ms": {"start": self.start_ms, "end": self.end_ms},
            "ticks_per_millisecond": self.ticks_per_ms, "num_layers": self.num_layers,
            "simulation_config": self.configuration, "hardware_calibrated": False,
            "counts": dict(self.counts), "pending_token_layers": pending,
            "pending_batches": pending_batches, "stop_reason": reason,
            "cohort_rule": "non-finalize attention batch selection in [start_ms,end_ms); all selected tokens followed through A_l -> E_l -> A_(l+1), including terminal finalize, without an end-window cutoff",
            "completion_rule": "token-layer complete at next attention's actual GPU-service start; source/expert actual_completion_ms records whole scheduled batch completion; grouped experts have no independently observed subexpert completion",
            "time_semantics": "selection_ms is reservation; gpu_start_ms, compute_start_ms, compute_end_ms are planned deterministic service boundaries; actual_gpu_start_ms is observed at an existing simulator event boundary; compute_start excludes GPU reformation and is a planned decomposition, not a separate event; next_attention.actual_completion_ms is normally null because the token row closes before next attention completes",
            "batch_membership": "batches file contains all members of each relevant source/expert/next-attention batch, including untracked tokens; expert token_ids may repeat for distinct expert_ids; IDs are recorder-local global selection ordinals, independent of output_batch_id",
            "censoring": "unfinished token-layers/batches are emitted with status=censored on early stop or exception; complete=true requires a fully drained simulation and no pending observations",
        }

    def _write_metadata(self, metadata):
        Path(self.paths["metadata"]).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n",
                                                encoding="utf-8")

    def _finalize(self, status, fully_drained, reason):
        pending, pending_batches = len(self.pending), len(self.batches)
        try:
            for record in self.pending.values():
                self._write(self._tokens, dict(record, status="censored", censor_reason=reason,
                                              observed_until_ms=self._ms(self.last_ticks)))
                self.counts["censored_token_layers"] += 1
            for state in self.batches.values():
                self._write(self._batch_file, dict(state["record"], status="censored", censor_reason=reason))
                self.counts["censored_batches"] += 1
        finally:
            if self._tokens is not None:
                self._tokens.close()
            if self._batch_file is not None:
                self._batch_file.close()
        self._metadata = self._make_metadata(status, fully_drained, pending, pending_batches, reason)
        self._write_metadata(self._metadata)
        self.pending.clear()
        self.batches.clear()
        return self._metadata

    def finish(self, now_ticks, *, fully_drained):
        if self._metadata is not None:
            return self._metadata
        self._observe(now_ticks)
        if fully_drained and (self.pending or self.batches):
            self._finalize("aborted", True, "fully_drained_with_pending_observations")
            raise RuntimeError("fully drained simulation has pending token trace observations")
        return self._finalize("complete" if fully_drained else "censored", fully_drained,
                              None if fully_drained else "simulation_stopped_early")

    def close(self, now_ticks=None, *, reason="exception_or_unfinished"):
        if self._metadata is not None:
            return self._metadata
        if now_ticks is not None:
            self._observe(now_ticks)
        return self._finalize("aborted", False, reason)


__all__ = ["TokenTraceRecorder"]
