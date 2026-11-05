"""CPU-only checks for the one-shot gate and fixed-cohort admission rules."""

import json
import multiprocessing
from pathlib import Path
from queue import Empty
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from disagmoe.utils.synchronized_start import (
    cleanup_gate,
    create_gate,
    publish_gate,
    validate_capacity,
    validate_configuration,
    validate_gate,
    wait_for_release,
)


def _waiting_process(gate, ready, results):
    validate_gate(**gate)
    ready.put(True)
    try:
        payload = wait_for_release(**gate, timeout_s=10)
        results.put((payload, time.perf_counter_ns(), None))
    except Exception as error:
        results.put((None, None, repr(error)))


def _configuration(**overrides):
    values = dict(
        synchronized_start=True, num_nodes=1, num_gpus=4, placement="colocate",
        tp_size=1, dp_size=4, ep_size=4, generator_type="offline",
        num_requests=64, max_batch_size_attn=16, min_input_len=64,
        max_input_len=65, min_output_len=32, max_output_len=33,
        analyze_throughput=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class GateTest(unittest.TestCase):
    def setUp(self):
        self.gate = create_gate()
        self.addCleanup(cleanup_gate, self.gate["directory"])

    def test_processes_wait_for_publication_and_share_one_future_target(self):
        # spawn also verifies that the gate has no process-local coordination.
        context = multiprocessing.get_context("spawn")
        ready, results = context.Queue(), context.Queue()
        workers = [context.Process(target=_waiting_process,
                                   args=(self.gate, ready, results)) for _ in range(2)]
        try:
            for worker in workers:
                worker.start()
            for _ in workers:
                self.assertTrue(ready.get(timeout=10))
            with self.assertRaises(Empty):
                results.get(timeout=0.05)
            target = publish_gate(**self.gate, delay_s=0.5)
            for _ in workers:
                payload, observed_ns, error = results.get(timeout=10)
                self.assertIsNone(error)
                self.assertEqual(payload, target)
                self.assertGreaterEqual(observed_ns, target["release_ns"])
            for worker in workers:
                worker.join(timeout=5)
                self.assertEqual(worker.exitcode, 0)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)
            ready.close()
            results.close()

    def test_publication_is_one_shot_even_with_competing_publishers(self):
        barrier = threading.Barrier(2)
        outcomes = []

        def publish():
            barrier.wait(timeout=2)
            try:
                outcomes.append(publish_gate(**self.gate))
            except FileExistsError as error:
                outcomes.append(error)

        workers = [threading.Thread(target=publish) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
        self.assertEqual(sum(isinstance(value, dict) for value in outcomes), 1)
        self.assertEqual(sum(isinstance(value, FileExistsError) for value in outcomes), 1)
        with self.assertRaises(FileExistsError):
            publish_gate(**self.gate)

    def test_cancellation_before_publication(self):
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            wait_for_release(**self.gate, cancelled=lambda: True)

    def test_cancellation_while_waiting_for_future_target(self):
        publish_gate(**self.gate, delay_s=1)
        cancel_at = time.perf_counter_ns() + 10_000_000
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            wait_for_release(**self.gate,
                             cancelled=lambda: time.perf_counter_ns() >= cancel_at)

    def test_timeout_before_publication(self):
        with self.assertRaises(TimeoutError):
            wait_for_release(**self.gate, timeout_s=0.01)

    def test_timeout_includes_future_target_wait(self):
        publish_gate(**self.gate, delay_s=1)
        with self.assertRaises(TimeoutError):
            wait_for_release(**self.gate, timeout_s=0.01)

    def test_late_first_observation_fails_instead_of_starting(self):
        publish_gate(**self.gate, delay_s=0.001)
        time.sleep(0.01)
        with self.assertRaisesRegex(RuntimeError, "already past"):
            wait_for_release(**self.gate)

    def test_marker_run_id_mismatch_is_rejected(self):
        wrong = dict(self.gate, run_id="other-run")
        for operation in (validate_gate, publish_gate, wait_for_release):
            with self.subTest(operation=operation.__name__), self.assertRaisesRegex(ValueError, "run_id"):
                operation(**wrong)

    def test_release_run_id_mismatch_is_rejected(self):
        release = {"run_id": "other-run", "release_ns": time.perf_counter_ns() + 10**9}
        (Path(self.gate["directory"]) / "release.json").write_text(json.dumps(release))
        with self.assertRaisesRegex(ValueError, "run_id"):
            wait_for_release(**self.gate)

    def test_invalid_target_is_rejected(self):
        release_path = Path(self.gate["directory"]) / "release.json"
        for target in (None, -1, True, "123"):
            release_path.write_text(json.dumps({"run_id": self.gate["run_id"], "release_ns": target}))
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "release_ns"):
                wait_for_release(**self.gate)

    def test_invalid_delays_and_timeouts_do_not_publish(self):
        for value in (0, -1, float("nan"), float("inf"), True, "1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    publish_gate(**self.gate, delay_s=value)
                with self.assertRaises(ValueError):
                    wait_for_release(**self.gate, timeout_s=value)
        self.assertFalse((Path(self.gate["directory"]) / "release.json").exists())
        publish_gate(**self.gate)

    def test_cleanup_is_idempotent_and_rejects_unrelated_directory(self):
        cleanup_gate(self.gate["directory"])
        cleanup_gate(self.gate["directory"])
        self.assertFalse(Path(self.gate["directory"]).exists())
        with tempfile.TemporaryDirectory() as unrelated:
            with self.assertRaises(ValueError):
                cleanup_gate(unrelated)
            self.assertTrue(Path(unrelated).exists())


class ConfigurationTest(unittest.TestCase):
    def test_default_mode_does_not_require_new_configuration(self):
        validate_configuration(SimpleNamespace())
        validate_configuration(SimpleNamespace(synchronized_start=False))

    def test_balanced_full_batch_is_accepted(self):
        validate_configuration(_configuration())
        validate_configuration(_configuration(num_requests=128, max_batch_size_attn=32))

    def test_unsupported_layout_or_workload_fails_early(self):
        cases = (
            {"num_nodes": 2}, {"placement": "disaggregate"}, {"tp_size": 2},
            {"dp_size": 2}, {"ep_size": 2}, {"num_gpus": 0},
            {"generator_type": "uniform"}, {"num_requests": 63},
            {"num_requests": 0}, {"num_requests": 68}, {"max_batch_size_attn": 0},
            {"max_input_len": 128}, {"max_output_len": 64},
            {"min_input_len": 0}, {"min_output_len": 0},
            {"mode": "server"}, {"is_api_server": True}, {"analyze_throughput": True},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_configuration(_configuration(**overrides))

    def test_capacity_checks_all_ranks_without_mutation(self):
        stats = [96, 100, 95, 200]
        with self.assertRaisesRegex(ValueError, "DP rank 2"):
            validate_capacity(stats, blocks_per_request=6, batch_per_rank=16)
        self.assertEqual(stats, [96, 100, 95, 200])
        validate_capacity([96] * 4, 6, 16)
        validate_capacity({0: 96, 1: 96, 2: 96, 3: 96}, 6, 16)

    def test_invalid_capacity_arguments(self):
        for stats, blocks, batch in (([], 6, 16), ({1: 96}, 6, 16),
                                     ([96, -1], 6, 16), ([96, True], 6, 16),
                                     ([96], 0, 16), ([96], 6, 0)):
            with self.subTest(stats=stats, blocks=blocks, batch=batch), self.assertRaises(ValueError):
                validate_capacity(stats, blocks, batch)


if __name__ == "__main__":
    unittest.main()
