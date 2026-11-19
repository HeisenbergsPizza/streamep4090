"""Exercise the worker's actual dequeue and export methods without CUDA or Ray."""

import ast
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from disagmoe.utils.scheduler_choices import SchedulerChoices


def worker(advanced=False, gdr=False):
    path = Path(__file__).resolve().parents[2] / "disagmoe/frontend/engine.py"
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "Engine")
    wanted = {"_schedule_batch_with_optional_trace", "dump_scheduler_choices"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    cls.bases = []
    clock = Mock()
    namespace = {"time": clock, "use_gdrcopy_optimization": gdr}
    exec(compile(ast.Module([cls], type_ignores=[]), str(path), "exec"), namespace)
    engine = namespace["Engine"]()
    engine.scheduler = Mock()
    engine._advanced_logger = SimpleNamespace(enabled=advanced)
    engine.device_id = 2
    engine._scheduler_choices = SchedulerChoices(48)
    return engine, clock.perf_counter_ns


def batch(layer, expert, size):
    meta = SimpleNamespace(layer_id=layer, is_expert=lambda: expert, num_tokens=lambda: size)
    return SimpleNamespace(data=object(), metadata=meta)


class SchedulerChoiceWorkerTest(unittest.TestCase):
    def test_successful_dequeues_and_half_open_window(self):
        engine, now = worker(advanced=True, gdr=False)
        empty = SimpleNamespace(data=None)
        batches = [batch(5, False, 1), empty, batch(0, False, 2),
                   batch(47, True, 4), batch(48, False, 3), batch(6, True, 7)]
        engine.scheduler.schedule.side_effect = batches
        now.side_effect = [19, 20, 21, 22, 23]
        for item in batches:
            self.assertEqual(engine._schedule_batch_with_optional_trace(), (item, None))
        self.assertEqual(now.call_count, 5)
        engine.scheduler.schedule_trace.assert_not_called()
        with TemporaryDirectory() as out:
            self.assertEqual(engine.dump_scheduler_choices(out, 20, 23), 2)
            path = Path(out) / "choices-gpu-2.csv"
            with path.open() as handle:
                self.assertEqual(list(csv.reader(handle)), [
                    ["time_ns", "stage", "layer", "batch_size"],
                    ["20", "attention", "0", "2"], ["21", "expert", "47", "4"],
                    ["22", "finalize", "48", "3"],
                ])
            engine.dump_scheduler_choices(out, 24, 25)
            self.assertEqual(path.read_text().splitlines(), ["time_ns,stage,layer,batch_size"])

    def test_advanced_scheduler_and_disabled_recording(self):
        engine, now = worker(advanced=True, gdr=True)
        item = batch(1, True, 9)
        trace = SimpleNamespace(batch=item, pool_snapshot=[9])
        engine.scheduler.schedule_trace.return_value = trace
        now.return_value = 100
        self.assertEqual(engine._schedule_batch_with_optional_trace(), (item, [9]))
        now.assert_called_once()
        engine._scheduler_choices = None
        self.assertEqual(engine._schedule_batch_with_optional_trace(), (item, [9]))
        now.assert_called_once()
        self.assertEqual(engine.scheduler.schedule_trace.call_count, 2)
        engine.scheduler.schedule.assert_not_called()
