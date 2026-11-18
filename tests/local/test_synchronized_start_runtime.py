"""Check real synchronized-start control flow with CPU stand-ins for GPU work.

Compile the production methods from their AST so Ray, CUDA and the extension
are not imported. The shared start gate and DP reservation methods stay real.
"""
import ast
import asyncio
import importlib.util
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import AsyncMock, patch
import typing


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from disagmoe.frontend.async_result import AsyncResult
from disagmoe.utils import synchronized_start as gate_api
from disagmoe.utils.runtime_checks import validate_request_lengths


def _source_class(path, name, methods, namespace):
    tree = ast.parse((REPO / path).read_text())
    original = next(node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == name)
    body = [node for node in original.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in methods]
    assert {node.name for node in body} == set(methods)
    probe = ast.ClassDef(name=name, bases=[], keywords=[], body=body, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[]))
    exec(compile(module, str(REPO / path), "exec"), namespace)
    return namespace[name]


class _Metadata(SimpleNamespace):
    def to_c(self):
        return self

    def num_tokens(self):
        return self.shape[0]

    def is_expert(self):
        return getattr(self, "expert", False)


class EngineSynchronizedStartTest(unittest.TestCase):
    def setUp(self):
        self.gate = gate_api.create_gate()
        self.requests = [dict(req_id=i + 1, input_len=64, output_len=32) for i in range(16)]
        self.calls = []
        self.markers = []
        self.registered = []
        self.incoming = []
        self.thread = None
        self.outcome = {}
        self.sync_entered = threading.Event()
        self.sync_can_finish = threading.Event()
        self.sync_can_finish.set()
        self.mutate_scheduled = lambda batch: batch
        self.leftover = False

        @contextmanager
        def nvtx_range(label):
            self.markers.append(label)
            yield

        def rand(shape, **kwargs):
            self.calls.append(("rand", shape))
            return SimpleNamespace(shape=shape, **kwargs)

        engine_class = _source_class("disagmoe/frontend/engine.py", "Engine", (
            "arm_synchronized_start", "get_synchronized_start_status",
            "cancel_synchronized_start", "_execute_synchronized_start",
        ), {
            "time": time, "BatchMetadata": _Metadata, "TokenBatch_C": SimpleNamespace,
            "TokenBatchCWrapper": SimpleNamespace(from_c=lambda batch: batch),
            "torch": SimpleNamespace(rand=rand, bfloat16="bfloat16"),
            "nvtx_range": nvtx_range,
        })
        self.engine = engine_class()
        self.engine.device = "cuda:0"
        self.engine.device_id = 2
        self.engine.attn_dp_rank = 2
        self.engine.loop_error = None
        self.engine.end_flag = False
        self.engine.loop_thread = SimpleNamespace(is_alive=lambda: True)
        self.engine.has_attn = self.engine.has_expert = True
        self.engine._tp_enabled = False
        self.engine._synchronized_start_status = None
        self.engine._synchronized_start_pending = None
        self.engine._synchronized_start_cancelled = threading.Event()
        self.engine._synchronized_start_labels = False
        self.engine._nsys_markers_enabled = True
        self.engine.engine_config = SimpleNamespace(max_batch_size_attn=16)
        self.engine.model_config = SimpleNamespace(hidden_size=2048)
        self.engine.record_max_output_lens = lambda req, length: self.registered.append((req, length))
        self.engine.pool = SimpleNamespace(put_batch=self.incoming.append)
        self.engine._advanced_logger = SimpleNamespace(log_pool_put=lambda count: None)
        self.engine.step_profile = lambda count: self.calls.append(("profile", count))

        def schedule():
            if not self.incoming:
                return SimpleNamespace(data=object() if self.leftover else None), None
            return self.mutate_scheduled(self.incoming[-1]), None

        def preprocess(wrapper):
            self.calls.append(("preprocess", wrapper.metadata.num_tokens()))
            return SimpleNamespace(proc_func=execute, wrapper=wrapper)

        def execute(batch):
            self.calls.append(("execute", batch.wrapper.metadata.num_tokens()))
            return "gpu-result"

        def synchronize():
            self.sync_entered.set()
            if not self.sync_can_finish.wait(2):
                raise TimeoutError("test did not unblock fake stream")
            self.calls.append("synchronized")

        self.engine._schedule_batch_with_optional_trace = schedule
        self.engine.preprocess_batch = preprocess
        self.engine.stream = SimpleNamespace(synchronize=synchronize)

    def tearDown(self):
        self.engine.cancel_synchronized_start()
        self.sync_can_finish.set()
        if self.thread is not None:
            self.thread.join(2)
            self.assertFalse(self.thread.is_alive(), "engine probe did not stop")
        gate_api.cleanup_gate(self.gate["directory"])

    def start_thread(self):
        def run():
            try:
                self.outcome["result"] = self.engine._execute_synchronized_start()
            except BaseException as exc:
                self.outcome["error"] = exc
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def wait_for_state(self, expected):
        deadline = time.perf_counter() + 2
        while time.perf_counter() < deadline:
            if self.engine.get_synchronized_start_status()["state"] == expected:
                return
            if "error" in self.outcome:
                raise self.outcome["error"]
            time.sleep(0.001)
        self.fail(f"engine did not reach {expected}")

    def assert_no_execution(self):
        self.assertFalse(any(isinstance(call, tuple) and call[0] == "execute"
                             for call in self.calls))

    def test_complete_batch_prepared_before_ready_and_held_until_shared_release(self):
        self.sync_can_finish.clear()
        self.engine.arm_synchronized_start(self.gate, self.requests)
        self.assertEqual(self.calls, [], "arming must not perform GPU work")
        self.start_thread()
        self.assertTrue(self.sync_entered.wait(2))
        self.assertEqual(self.engine.get_synchronized_start_status()["state"], "preparing")
        self.assert_no_execution()
        self.sync_can_finish.set()
        self.wait_for_state("ready")
        self.assert_no_execution()
        self.assertEqual(len(self.incoming), 1)
        meta = self.incoming[0].metadata
        self.assertEqual(meta.shape, [16, 2048])
        self.assertEqual(meta.req_ids, list(range(1, 17)))
        self.assertEqual(meta.attn_dp_ranks, [2] * 16)
        self.assertEqual(meta.init_prefill_lens, [64] * 16)
        self.assertEqual(self.registered, [(i, 32) for i in range(1, 17)])

        release = gate_api.publish_gate(**self.gate, delay_s=0.1)
        self.thread.join(2)
        self.assertFalse(self.thread.is_alive())
        self.assertNotIn("error", self.outcome)
        self.assertEqual(self.outcome["result"][1], "gpu-result")
        status = self.engine.get_synchronized_start_status()
        self.assertEqual(status["state"], "executed")
        self.assertLess(status["prepared_ns"], release["release_ns"])
        self.assertGreaterEqual(status["execute_begin_ns"], release["release_ns"])
        self.assertEqual(self.calls.count(("execute", 16)), 1)
        self.assertIn("Engine.execute.attn.layer=0.batch=16.initial", self.markers)

    def test_wrong_initial_cohort_never_executes(self):
        mutations = {
            "partial": lambda meta: setattr(meta, "shape", [15, 2048]),
            "wrong-layer": lambda meta: setattr(meta, "layer_id", 1),
            "expert": lambda meta: setattr(meta, "expert", True),
            "wrong-request": lambda meta: setattr(meta, "req_ids", [999] + meta.req_ids[1:]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                self.engine._synchronized_start_status = None
                self.incoming.clear()
                self.calls.clear()
                def changed(batch):
                    mutate(batch.metadata)
                    return batch
                self.mutate_scheduled = changed
                self.engine.arm_synchronized_start(self.gate, self.requests)
                with self.assertRaisesRegex(RuntimeError, "complete layer-0"):
                    self.engine._execute_synchronized_start()
                self.assert_no_execution()
                self.assertFalse(self.sync_entered.is_set())

    def test_leftover_work_is_rejected_before_injecting_requests(self):
        self.leftover = True
        self.engine.arm_synchronized_start(self.gate, self.requests)
        with self.assertRaisesRegex(RuntimeError, "empty scheduler"):
            self.engine._execute_synchronized_start()
        self.assertEqual(self.incoming, [])
        self.assertEqual(self.registered, [])
        self.assert_no_execution()

    def test_cancelled_gate_does_not_execute(self):
        self.engine.arm_synchronized_start(self.gate, self.requests)
        self.start_thread()
        self.wait_for_state("ready")
        self.engine.cancel_synchronized_start()
        self.thread.join(2)
        self.assertIsInstance(self.outcome.get("error"), RuntimeError)
        self.assertIn("cancelled", str(self.outcome["error"]))
        self.assert_no_execution()

    def test_gate_timeout_does_not_execute(self):
        self.engine.arm_synchronized_start(self.gate, self.requests)
        real_wait = gate_api.wait_for_release
        def short_wait(*args, **kwargs):
            return real_wait(*args, timeout_s=0.02, **kwargs)
        with patch.object(gate_api, "wait_for_release", short_wait):
            with self.assertRaises(TimeoutError):
                self.engine._execute_synchronized_start()
        self.assert_no_execution()

    def test_one_shot_arm_and_status_failure_reporting(self):
        self.engine.arm_synchronized_start(self.gate, self.requests)
        with self.assertRaisesRegex(RuntimeError, "one-shot"):
            self.engine.arm_synchronized_start(self.gate, self.requests)
        snapshot = self.engine.get_synchronized_start_status()
        snapshot["state"] = "external edit"
        self.assertEqual(self.engine.get_synchronized_start_status()["state"], "armed")
        self.engine.loop_error = "GPU worker failed"
        self.assertEqual(self.engine.get_synchronized_start_status()["state"], "failed")


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU tensor test requires torch")
class PacedWorkerTest(unittest.TestCase):
    def test_mixed_first_layer_batch_restores_new_kv_in_scheduled_order(self):
        import torch
        from disagmoe.frontend.real_decode import DecodeResiduals

        class Snapshot:
            def __len__(self):
                return 4

            def prompt_len(self, i):
                return 2

            def first_token(self, i):
                return 10 + i

            def load_kv(self, i):
                return torch.full((1, 2, 2, 1, 1), i + 1, dtype=torch.bfloat16)

        ns = {"torch": torch, "TokenBatchCWrapper": object, "AttentionForwardBatch": SimpleNamespace,
              "Optional": Optional, "use_gdrcopy_optimization": False,
              "AttentionScheduleBatch": SimpleNamespace(build=lambda meta, data: meta)}
        worker = _source_class("disagmoe/frontend/engine.py", "Engine",
                               ("_paced_request_input", "_restore_snapshot_rows"), ns)()
        mixin = _source_class("disagmoe/frontend/engine.py", "AttentionEngineMixin",
                              ("preprocess_batch_attn",), ns)
        worker.model_config = SimpleNamespace(checkpoint_path="/model", num_layers=1, hidden_size=1)
        worker.model_total_num_layers = 1
        worker.cache_config = SimpleNamespace(num_gpu_blocks=6)
        worker._paced_snapshot = Snapshot()
        worker._pending_snapshot_indices = {}
        worker.device = "cpu"
        worker.decode_residuals = DecodeResiduals([7], 1, "cpu", capacity=3)
        pages = torch.full((2, 6, 2, 1, 1), -1., dtype=torch.bfloat16)
        worker.attn_executor = SimpleNamespace(embed_tokens=lambda ids: ids[:, None],
                                               kv_cache=SimpleNamespace(get_kv_buffer=lambda layer: pages))
        worker.execute_batch_attn = worker.postprocess_batch_attn = lambda batch: None
        for rid, index in [(20, 0), (30, 3)]:
            req = SimpleNamespace(req_id=rid, snapshot_index=index, init_prefill_len=2)
            self.assertEqual(worker._paced_request_input(req).item(), 10 + index)

        # The continuing request has live KV on page 0. New admissions are
        # scheduled in reverse arrival order and own pages 2 and 3.
        meta = SimpleNamespace(req_ids=[30, 20, 7], num_prefill_seqs=2, layer_id=0,
                               meta_c=None, data=torch.tensor([[13], [10], [90]]),
                               num_tokens=lambda: 3, seq_lens_tensor=torch.tensor([3, 3, 4]))
        worker._attn_driver_preprocess = lambda *_: SimpleNamespace(
            block_tables=torch.tensor([[2, 4], [3, 5], [0, 1]]))
        forward = mixin.preprocess_batch_attn(worker, SimpleNamespace(metadata=meta, data=meta.data))
        self.assertEqual(forward.positions.tolist(), [2, 2, 3])
        self.assertTrue((pages[:, 2] == 4).all())
        self.assertTrue((pages[:, 3] == 1).all())
        self.assertTrue((pages[:, :2] == -1).all())
        self.assertEqual(worker._pending_snapshot_indices, {})
        freed = worker.decode_residuals.rows[30]
        worker.decode_residuals.release([30])
        worker.decode_residuals.admit(40)
        self.assertEqual(worker.decode_residuals.rows[40], freed)


class _RemoteMethod:
    def __init__(self, function):
        self.function = function

    async def remote(self, *args):
        return self.function(*args)


class RealTokenFeedbackTest(unittest.TestCase):
    def test_reset_rejects_inflight_residual_and_rearms_drained_worker(self):
        from concurrent.futures import Future
        ns = {"Future": Future, "FutureTimeout": TimeoutError}
        worker = _source_class("disagmoe/frontend/engine.py", "Engine",
                               ("reset_real_cohort", "_finish_real_cohort_reset"), ns)()
        worker._synchronized_start_status = {"state": "executed"}
        worker._real_cohort_reset = None
        worker.model_config = SimpleNamespace(checkpoint_path="/model")
        worker.loop_error = None
        worker.end_flag = False
        worker.decode_residuals = SimpleNamespace(pending_layer={7: 3})
        worker.dummy_sampler = SimpleNamespace(output_len={})
        worker.block_mgr = SimpleNamespace(req_manager=SimpleNamespace(get_active_req_ids=lambda: []))
        drained = threading.Event()
        worker.stream = SimpleNamespace(synchronize=drained.set)

        def acknowledge():
            deadline = time.monotonic() + 2
            while worker._real_cohort_reset is None and time.monotonic() < deadline:
                time.sleep(0.001)
            if worker._real_cohort_reset is not None:
                worker._finish_real_cohort_reset()

        thread = threading.Thread(target=acknowledge)
        thread.start()
        with self.assertRaisesRegex(RuntimeError, "requests are active"):
            worker.reset_real_cohort()
        thread.join()
        self.assertFalse(drained.is_set())
        worker.decode_residuals.pending_layer.clear()
        thread = threading.Thread(target=acknowledge)
        thread.start()
        worker.reset_real_cohort()
        thread.join()
        self.assertTrue(drained.is_set())
        self.assertIsNone(worker.decode_residuals)
        self.assertIsNone(worker._synchronized_start_status)

    def test_reordered_requests_emit_ids_and_only_embed_continuing_rows(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU PyTorch needed to exercise real token feedback")
        ns = {"AttentionScheduleBatch": object, "TokenBatchCWrapper": SimpleNamespace,
              "BatchDecodeResult": SimpleNamespace, "List": list, "Tuple": tuple,
              "Dict": dict, "use_gdrcopy_optimization": False}
        engine = _source_class("disagmoe/frontend/engine.py", "AttentionEngineMixin",
                               ("sample_results",), ns)()
        engine.dummy_sampler = _source_class("disagmoe/frontend/engine.py", "DummySampler",
                                              ("__init__", "sample_once", "create_request",
                                               "clean_request", "check_end"), ns)()
        engine.dummy_sampler.create_request(71, 1)
        engine.dummy_sampler.create_request(19, 2)
        engine.attn_dp_rank = 0
        engine.model_config = SimpleNamespace(checkpoint_path="/model")
        from disagmoe.frontend.real_decode import DecodeResiduals
        engine.decode_residuals = DecodeResiduals([71, 19], 3, "cpu")
        reported, freed = [], []
        engine.detokenizer_socket = SimpleNamespace(send_pyobj=reported.append)
        engine.release_seqs = freed.extend
        engine.attn_executor = SimpleNamespace(greedy_token_ids=lambda x: x.argmax(-1),
                                               embed_tokens=lambda ids: (ids + 10).unsqueeze(1))
        def sample(req_ids, rows):
            meta = SimpleNamespace(index_select=lambda indices: SimpleNamespace(
                req_ids=[req_ids[j] for j in indices]))
            return engine.sample_results(SimpleNamespace(req_ids=req_ids, meta_c=meta,
                                                         data=torch.tensor(rows)))

        nxt = sample([19, 71], [[0., 3., 1.], [3., 2., 1.]])
        self.assertEqual((reported[0].token_ids, reported[0].is_eos), ([1, 0], [False, True]))
        self.assertEqual(nxt.metadata.req_ids, [19])
        self.assertEqual(nxt.data.tolist(), [[11]])
        self.assertEqual(freed, [71])
        last = sample([19], [[0., 1., 2.]])
        self.assertEqual((reported[1].token_ids, reported[1].is_eos), ([2], [True]))
        self.assertEqual(last.data.shape[0], 0)
        self.assertEqual(freed, [71, 19])
        self.assertEqual(engine.decode_residuals.rows, {})


class _Worker:
    def __init__(self, rank):
        self.rank = rank
        self.node = "same-ray-node"
        self.requests = []
        self.gate = None
        self.state = "unarmed"
        self.arm_error = None
        self.cancel_count = 0
        self.get_worker_identity = _RemoteMethod(lambda interface: {"ray_node_id": self.node})
        self.arm_synchronized_start = _RemoteMethod(self.arm)
        self.get_synchronized_start_status = _RemoteMethod(self.status)
        self.cancel_synchronized_start = _RemoteMethod(self.cancel)

    def arm(self, gate, requests):
        self.gate = gate
        self.requests = requests
        if self.arm_error:
            raise self.arm_error
        self.state = "ready"

    def status(self):
        return {"state": self.state, "attn_dp_rank": self.rank,
                "initial_batch_size": len(self.requests), "layer_id": 0,
                "req_ids": [request["req_id"] for request in self.requests]}

    def cancel(self):
        self.cancel_count += 1
        self.state = "cancelled"


class ControllerSynchronizedStartTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        controller_class = _source_class("disagmoe/frontend/controller.py", "Controller", (
            "prepare_synchronized_start", "abort_synchronized_start",
            "_get_synchronized_start_statuses", "release_synchronized_start",
            "finish_synchronized_start",
        ), {
            "asyncio": asyncio, "time": time, "AsyncResult": AsyncResult,
            "validate_request_lengths": validate_request_lengths,
            "get_logger": lambda: SimpleNamespace(exception=lambda message: None),
            "ray": SimpleNamespace(get_runtime_context=lambda: SimpleNamespace(
                get_node_id=lambda: "same-ray-node")),
        })
        scheduler_class = _source_class("disagmoe/scheduler/dpscheduler.py", "DPScheduler", (
            "required_blocks", "add_seq", "del_seq",
        ), {})
        self.controller = controller_class()
        self.controller._poll_error = None
        self.controller._synchronized_start_gate = None
        self.controller._synchronized_start_responses = []
        self.controller.synchronized_start_report = None
        self.controller.model_config = SimpleNamespace(dp_size=4, max_seq_len=4096)
        self.controller.cache_config = SimpleNamespace(block_size=16)
        self.controller._max_cache_blocks = 128
        self.controller.host_ifname = ""
        self.controller.in_flight_reqs = set()
        self.controller.request_results = {}
        self.controller.device_ids = [20, 0, 30, 10]
        self.controller.model_place = SimpleNamespace(attn_dp_rank_at=lambda device: device // 10)
        self.controller.workers = [_Worker(rank) for rank in (2, 0, 3, 1)]
        scheduler = scheduler_class()
        scheduler.waiting_queue = asyncio.Queue()
        scheduler.block_size = 16
        scheduler.kv_cache_stats = [128] * 4
        scheduler.free_request_slots = None
        scheduler.seq_ranks = {}
        scheduler.seq_max_len = {}
        scheduler.sch_event = asyncio.Event()
        self.controller.dp_scheduler = scheduler
        self.next_id = 0
        self.polling_stopped = False

        def new_id():
            self.next_id += 1
            self.controller.in_flight_reqs.add(self.next_id)
            return self.next_id

        async def stop_polling():
            self.polling_stopped = True

        self.controller.get_new_req_id = new_id
        self.controller.stop_polling_results = stop_polling

    async def asyncTearDown(self):
        if self.controller._synchronized_start_gate:
            await self.controller.abort_synchronized_start()

    async def test_real_cohort_leaves_append_headroom_and_maps_snapshot_by_dp_rank(self):
        self.controller.model_config.checkpoint_path = "/model"
        self.controller.model_config.decode_snapshot = "/snapshot"
        self.controller.dp_scheduler.kv_cache_stats[0] = 96
        with self.assertRaisesRegex(ValueError, "spare KV block on DP rank 0"):
            await self.controller.prepare_synchronized_start(64, 64, 32)
        self.assertFalse(self.controller.in_flight_reqs)
        self.controller.dp_scheduler.kv_cache_stats[0] = 97
        await self.controller.prepare_synchronized_start(64, 64, 32)
        for w in self.controller.workers:
            self.assertEqual([r["snapshot_index"] for r in w.requests],
                             list(range(w.rank * 16, (w.rank + 1) * 16)))

    async def test_capacity_checks_every_rank_before_allocating_any_request(self):
        # One rank is short by one block although the other three have room.
        self.controller.dp_scheduler.kv_cache_stats[3] = 95
        before = self.controller.dp_scheduler.kv_cache_stats.copy()
        with self.assertRaisesRegex(ValueError, "DP rank 3"):
            await self.controller.prepare_synchronized_start(64, 64, 32)
        self.assertEqual(self.next_id, 0)
        self.assertEqual(self.controller.in_flight_reqs, set())
        self.assertEqual(self.controller.dp_scheduler.seq_ranks, {})
        self.assertEqual(self.controller.dp_scheduler.kv_cache_stats, before)
        self.assertIsNone(self.controller._synchronized_start_gate)
        self.assertTrue(all(not worker.requests for worker in self.controller.workers))

    async def test_balanced_cohorts_follow_dp_rank_and_register_every_response(self):
        responses = await self.controller.prepare_synchronized_start(64, 64, 32)
        self.assertEqual(len(responses), 64)
        self.assertEqual(set(self.controller.request_results), set(range(1, 65)))
        self.assertEqual(self.controller.in_flight_reqs, set(range(1, 65)))
        self.assertEqual(self.controller.dp_scheduler.kv_cache_stats, [32] * 4)
        for worker in self.controller.workers:
            self.assertEqual(len(worker.requests), 16)
            self.assertEqual({self.controller.dp_scheduler.seq_ranks[r["req_id"]]
                              for r in worker.requests}, {worker.rank})
            self.assertTrue(all(r["input_len"] == 64 and r["output_len"] == 32
                                for r in worker.requests))
        directory = Path(self.controller._synchronized_start_gate["directory"])
        self.assertFalse((directory / "release.json").exists(), "prepare must not release")
        self.assertEqual(self.controller.synchronized_start_report["initial_batch_per_rank"], 16)

        await self.controller.abort_synchronized_start()
        self.assertFalse(directory.exists())
        self.assertTrue(self.polling_stopped)
        self.assertEqual(self.controller.dp_scheduler.kv_cache_stats, [128] * 4)
        self.assertEqual(self.controller.in_flight_reqs, set())
        for response in responses:
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                await response.get()

    async def test_partial_arm_failure_cancels_all_workers_and_releases_reservations(self):
        failure = RuntimeError("worker arm rejected")
        self.controller.workers[1].arm_error = failure
        with self.assertRaises(RuntimeError) as caught:
            await self.controller.prepare_synchronized_start(64, 64, 32)
        self.assertIs(caught.exception, failure)
        self.assertTrue(self.polling_stopped)
        self.assertTrue(all(worker.cancel_count == 1 for worker in self.controller.workers))
        self.assertEqual(self.controller.dp_scheduler.kv_cache_stats, [128] * 4)
        self.assertEqual(self.controller.dp_scheduler.seq_ranks, {})
        self.assertEqual(self.controller.request_results, {})
        self.assertEqual(self.controller.in_flight_reqs, set())
        self.assertIsNone(self.controller._synchronized_start_gate)
        for worker in self.controller.workers:
            self.assertFalse(Path(worker.gate["directory"]).exists())

    async def test_remote_node_is_rejected_before_reservation(self):
        self.controller.workers[0].node = "other-ray-node"
        with self.assertRaisesRegex(ValueError, "single Ray node"):
            await self.controller.prepare_synchronized_start(64, 64, 32)
        self.assertEqual(self.next_id, 0)
        self.assertIsNone(self.controller._synchronized_start_gate)

    async def test_release_rechecks_all_workers_ready(self):
        await self.controller.prepare_synchronized_start(64, 64, 32)
        self.controller.workers[0].state = "preparing"
        with self.assertRaisesRegex(RuntimeError, "still be ready"):
            await self.controller.release_synchronized_start()
        directory = Path(self.controller._synchronized_start_gate["directory"])
        self.assertFalse((directory / "release.json").exists())


@unittest.skipIf(sys.version_info < (3, 10), "scheduler annotations require Python >=3.10")
class ControllerPacedAdmissionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        with patch.object(typing, "override", getattr(typing, "override", lambda f: f), create=True):
            from disagmoe.scheduler.dpscheduler import DPSchedulerMax
        cls = _source_class("disagmoe/frontend/controller.py", "Controller", (
            "start_scheduler", "stop_scheduler", "get_new_req_id", "put_snapshot_request",
            "_dispatch_snapshot", "_enqueue_request", "process_finished_results",
            "_fail_pending_requests",
        ), dict(asyncio=asyncio, time=time, AsyncResult=AsyncResult, partial=partial,
                validate_request_lengths=validate_request_lengths, List=typing.List,
                SloStat=SimpleNamespace, ray=SimpleNamespace(get=lambda result: result),
                get_logger=lambda: SimpleNamespace(error=lambda _: None)))
        master = self.master = cls()
        master.model_config = SimpleNamespace(max_seq_len=4096)
        master.cache_config = SimpleNamespace(block_size=16)
        master.dp_scheduler = DPSchedulerMax(1, 16)
        master.device_ids = [0]
        master.model_place = SimpleNamespace(has_attn=lambda _: True, attn_dp_rank_at=lambda _: 0)
        master.workers = [SimpleNamespace(
            get_configured_kv_cache_blocks=SimpleNamespace(remote=AsyncMock(return_value=6)),
            prepare_paced_snapshot=SimpleNamespace(remote=AsyncMock(
                return_value={"requests": 3, "request_slots": 1})),
        )]
        master.req_id_generator = iter(range(1, 100))
        master._poll_error = None
        master._snapshot_sends = set()
        master.in_flight_reqs = set()
        master.request_results = {}
        self.sent = asyncio.Queue()

        async def send(req_id, input_len, output_len, rank, *, snapshot_index):
            self.sent.put_nowait((req_id, snapshot_index, rank))

        master.tokenizer = SimpleNamespace(put_single_request=SimpleNamespace(remote=send))
        await master.start_scheduler(paced_real_decode=True)
        self.addAsyncCleanup(master.stop_scheduler)

    async def test_slot_backpressure_releases_next_snapshot_with_new_id(self):
        master = self.master
        first = master.put_snapshot_request(2, 16, 16)
        second = master.put_snapshot_request(0, 16, 16)
        self.assertEqual(await asyncio.wait_for(self.sent.get(), 1), (first.req_id, 2, 0))
        self.assertIsNotNone(first.admitted_at)
        self.assertIsNone(second.admitted_at)
        self.assertEqual(master.dp_scheduler.kv_cache_stats, [3])  # 6 - 1 spare - 2 reserved
        await master.process_finished_results([SimpleNamespace(req_id=first.req_id)])
        self.assertEqual(await asyncio.wait_for(self.sent.get(), 1), (second.req_id, 0, 0))
        self.assertGreaterEqual(second.admitted_at, first.admitted_at)
        await master.process_finished_results([SimpleNamespace(req_id=second.req_id)])
        self.assertFalse(master.in_flight_reqs)
        self.assertEqual(master.dp_scheduler.kv_cache_stats, [5])
        self.assertEqual(master.dp_scheduler.free_request_slots, {0: 1})
        with self.assertRaisesRegex(ValueError, "KV blocks"):
            master.put_snapshot_request(0, 64, 32)
        with self.assertRaisesRegex(ValueError, "outside the prepared workload"):
            master.put_snapshot_request(3, 16, 16)
        self.assertFalse(master.request_results)

    async def test_delivery_failure_reaches_queued_and_dispatched_requests(self):
        master = self.master
        master.tokenizer.put_single_request.remote = AsyncMock(side_effect=RuntimeError("send failed"))
        first = master.put_snapshot_request(0, 16, 16)
        waiting = master.put_snapshot_request(1, 16, 16)
        for response in (first, waiting):
            with self.assertRaisesRegex(RuntimeError, "send failed"):
                await asyncio.wait_for(response.get(), 1)
        self.assertIsNone(waiting.admitted_at)
        with self.assertRaisesRegex(RuntimeError, "Inference service has failed"):
            master.put_snapshot_request(2, 16, 16)


if __name__ == "__main__":
    unittest.main()
