"""Exercise formal request timing and per-worker scheduler choice export on CPU."""
from __future__ import annotations

import ast
import asyncio
import csv
import json
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from disagmoe.utils.scheduler_choices import SchedulerChoices
from experiments.scripts.rtx4090.run_local import parse_configuration


REPO = Path(__file__).resolve().parents[2]


def load_functions(path, names, namespace, class_name=None):
    """Load real function bodies without importing the CUDA/Ray runtime."""
    tree = ast.parse((REPO / path).read_text())
    if class_name:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    functions = [n for n in tree.body if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
                 and n.name in names]
    exec(compile(ast.Module(functions, type_ignores=[]), path, "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


class BenchmarkExportTest(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_then_formal_driver_and_worker_export(self):
        with TemporaryDirectory() as directory:
            _, args, _ = parse_configuration(["benchmark", "--scheduler-choices-dir", directory,
                                              "--rate", "8", "--num-requests", "2"])
            device_ids = [3, 0, 2, 1]
            master = SimpleNamespace(device_ids=device_ids, model_config=args, workers=[])
            engine = load_functions("disagmoe/frontend/engine.py", ["setup_engine", "dump_scheduler_choices"], {
                "initialize_logger": Mock(), "torch": Mock(), "get_logger": Mock(),
                "set_tensor_model_parallel_config": Mock(), "set_global_engine_config": Mock(),
                "SchedulerChoices": SchedulerChoices,
            }, "Engine")
            for rank in device_ids:
                worker = SimpleNamespace(device_id=rank, has_attn=False, has_expert=False)
                engine.setup_engine(worker, "unified", args, args)
                worker.dump_scheduler_choices = SimpleNamespace(remote=partial(engine.dump_scheduler_choices, worker))
                master.workers.append(worker)
            controller = load_functions("disagmoe/frontend/controller.py", ["dump_scheduler_choices"],
                                        {"Path": Path, "json": json, "ray": SimpleNamespace(get=list)}, "Controller")
            master.dump_scheduler_choices = partial(controller.dump_scheduler_choices, master)

            # Keep both clocks on the same deterministic monotonic timeline.
            clock_ns = 8_000_000_000
            async def sleep(seconds):
                nonlocal clock_ns
                clock_ns += round(seconds * 1e9)

            async def response():
                await sleep(1 / 16)
                return "done"

            def submit(input_len, output_len):
                meta = SimpleNamespace(layer_id=1, is_expert=lambda: False, num_tokens=lambda: input_len)
                master.workers[0]._scheduler_choices.record(clock_ns, meta)
                return SimpleNamespace(get=response)

            def generator(rate, cv, in_min, in_max, out_min, out_max):
                return SimpleNamespace(generate_num=lambda n: [(i / rate, in_min, out_min) for i in range(n)])

            master.put_single_request = submit
            bench = load_functions("benchmark/benchmark_serving.py", ["run_benchmark", "process_response"], {
                "time": SimpleNamespace(perf_counter_ns=lambda: clock_ns, perf_counter=lambda: clock_ns / 1e9),
                "asyncio": SimpleNamespace(sleep=sleep, gather=asyncio.gather, create_task=asyncio.create_task),
                "get_generator": lambda kind: generator, "DatasetGenerator": object,
                "tqdm": Mock(), "logger": Mock(),
            })
            out = Path(directory)
            await bench.run_benchmark(master, args, "uniform", 1, 100, 101, 50, 51, 5, warmup=True)
            self.assertEqual(list(out.iterdir()), [])
            start = clock_ns
            result = await bench.run_benchmark(master, args, "uniform", 2, 64, 65, 128, 129, 8)
            self.assertEqual(result, (["done", "done"], [0, 0.125], [0.1875, 0.25], 0.25))
            manifest = json.loads((out / "run.json").read_text())
            self.assertEqual((manifest["start_ns"], manifest["end_ns"]), (start, start + 250_000_000))
            self.assertEqual((manifest["submitted_s"], manifest["finished_s"]), result[1:3])
            self.assertEqual(manifest["device_ids"], sorted(device_ids))
            self.assertEqual(manifest["num_layers"], 48)
            self.assertEqual(manifest["rate"], 8)
            self.assertEqual(manifest["configuration"]["num_requests"], 2)
            for rank in device_ids:
                with (out / f"choices-gpu-{rank}.csv").open() as handle:
                    rows = list(csv.reader(handle))
                expected = [["time_ns", "stage", "layer", "batch_size"]]
                if rank == device_ids[0]:
                    expected += [[str(start), "attention", "1", "64"],
                                 [str(start + 125_000_000), "attention", "1", "64"]]
                self.assertEqual(rows, expected)
