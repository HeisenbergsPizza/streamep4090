"""CPU-only checks for shared expert workloads and standalone NCU commands.

These checks do not validate GPU numerics, CUDA compatibility, or NCU counters.
"""

import importlib.util
from collections import Counter
import contextlib
import io
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


REPO = Path(__file__).resolve().parents[2]
OPS = REPO / "experiments/scripts/ops"
LAUNCHER = OPS / "profile_experts.sh"


def import_script(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, OPS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


adapter = import_script("sglang_expert_adapter.py", "expert_bench_sglang_adapter_test")


class SharedExpertRoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = import_script("expert_bench.py", "expert_bench_helpers_test")

    def config(self, **changes):
        return {
            "tokens": 256, "hidden": 2048, "intermediate": 768,
            "global_experts": 128, "ep_size": 4, "local_rank": 0,
            "top_k": 8, "seed": 17, "routing": "balanced", **changes,
        }

    def test_balanced_case_has_512_local_branches_and_16_rows_per_expert(self):
        config = self.config()
        self.bench.validate_config(config)
        routes = self.bench.make_routes(config)
        self.assertEqual(len(routes), 256)
        for selected in routes:
            self.assertEqual(len(selected), 8)
            self.assertEqual(len(set(selected)), 8)
        self.assertEqual(Counter(expert for row in routes for expert in row),
                         Counter({expert: 16 for expert in range(128)}))
        for rank in range(4):
            with self.subTest(rank=rank):
                local, counts = self.bench.localize_routes(routes, 128, 4, rank)
                self.assertEqual(counts, [16] * 32)
                self.assertEqual(sum(counts), 512)
                self.assertEqual(sum(value >= 0 for row in local for value in row), 512)
                for global_row, local_row in zip(routes, local):
                    for original, mapped in zip(global_row, local_row):
                        expected = original - rank * 32 if rank * 32 <= original < (rank + 1) * 32 else -1
                        self.assertEqual(mapped, expected)

    def test_random_routes_are_seeded_unique_and_partition_without_lost_branches(self):
        config = self.config(tokens=73, routing="random")
        routes = self.bench.make_routes(config)
        self.assertEqual(routes, self.bench.make_routes(config))
        self.assertNotEqual(routes, self.bench.make_routes({**config, "seed": 18}))
        for row in routes:
            self.assertEqual(len(row), 8)
            self.assertEqual(len(set(row)), 8)
            self.assertTrue(all(0 <= expert < 128 for expert in row))
        global_counts = Counter(expert for row in routes for expert in row)
        mapped_total = 0
        for rank in range(4):
            local, counts = self.bench.localize_routes(routes, 128, 4, rank)
            self.assertEqual(counts, [global_counts[rank * 32 + expert] for expert in range(32)])
            mapped_total += sum(value >= 0 for row in local for value in row)
        self.assertEqual(mapped_total, 73 * 8)

    def test_global_to_local_mapping_keeps_expert_boundaries(self):
        routes = [[0, 31, 32, 63, 64, 95, 96, 127]]
        local, counts = self.bench.localize_routes(routes, 128, 4, 2)
        self.assertEqual(local, [[-1, -1, -1, -1, 0, 31, -1, -1]])
        self.assertEqual(counts, [1] + [0] * 30 + [1])

    def test_invalid_shapes_and_parallel_configuration_are_rejected(self):
        invalid = [
            {"tokens": 0}, {"tokens": -1}, {"hidden": 2049},
            {"intermediate": 0}, {"intermediate": 769},
            {"global_experts": 127}, {"ep_size": 0},
            {"local_rank": -1}, {"local_rank": 4},
            {"top_k": 0}, {"top_k": 129}, {"routing": "unknown"},
            {"tokens": True}, {"tokens": 2.5}, {"seed": "17"},
        ]
        for change in invalid:
            with self.subTest(change=change):
                with self.assertRaises((ValueError, TypeError)):
                    self.bench.validate_config(self.config(**change))

    def test_corrupted_global_routes_are_rejected(self):
        for row in ([0, 0], [-1, 32], [128, 32], [True, 32], [1.5, 32]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.bench.localize_routes([row], 128, 4, 0)


@unittest.skipIf(torch is None, "Optional fixture checks require CPU torch")
class ExpertTensorFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = import_script("expert_bench.py", "expert_bench_tensor_test")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="expert-fixture-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "shared.pt"
        self.args = types.SimpleNamespace(
            output=str(self.path), tokens=8, hidden=8, intermediate=16,
            global_experts=8, ep_size=2, local_rank=1, top_k=2,
            seed=17, routing="balanced",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.bench.prepare(self.args)

    def test_saved_case_roundtrip_preserves_exact_shared_tensors(self):
        first = self.bench.load_case(self.path)
        second = self.bench.load_case(self.path)
        for name in ("x", "w13", "w2", "global_topk_ids", "topk_ids",
                     "topk_weights", "batch_sizes"):
            self.assertTrue(torch.equal(first[name], second[name]), name)
            self.assertEqual(first[name].device.type, "cpu")
        self.assertEqual(first["batch_sizes"].tolist(), [2, 2, 2, 2])
        self.assertEqual(first["w13"].dtype, torch.bfloat16)
        self.assertEqual(tuple(first["w13"].shape), (4, 8, 32))
        self.assertEqual(tuple(first["w2"].shape), (4, 16, 8))
        digest = self.bench.digest(self.path)
        with self.assertRaises(FileExistsError):
            self.bench.prepare(self.args)
        self.assertEqual(self.bench.digest(self.path), digest)

    def test_load_rejects_corrupted_routes_weights_shapes_and_schema(self):
        def duplicate_global(case):
            case["global_topk_ids"][0, 1] = case["global_topk_ids"][0, 0]

        def wrong_local(case):
            row, slot = torch.nonzero(case["topk_ids"] >= 0)[0].tolist()
            case["topk_ids"][row, slot] = (case["topk_ids"][row, slot] + 1) % 4

        def wrong_counts(case):
            case["batch_sizes"][0] += 1

        def wrong_weights(case):
            case["topk_weights"][0] = 0

        def nonfinite(case):
            case["x"][0, 0] = float("nan")

        mutations = {
            "duplicate global expert": duplicate_global,
            "incorrect local expert": wrong_local,
            "incorrect counts": wrong_counts,
            "unnormalized weights": wrong_weights,
            "nonfinite input": nonfinite,
            "wrong shape": lambda case: case.update(x=case["x"][:, :-1]),
            "wrong dtype": lambda case: case.update(w13=case["w13"].float()),
            "wrong schema": lambda case: case.update(schema_version=999),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                case = torch.load(self.path, map_location="cpu", weights_only=True)
                mutate(case)
                corrupted = self.directory / "corrupted.pt"
                torch.save(case, corrupted)
                with self.assertRaises(ValueError):
                    self.bench.load_case(corrupted)

    def test_numeric_gate_accepts_small_rounding_and_rejects_wrong_output(self):
        expected = torch.tensor([[1.0, -2.0], [0.25, 0.75]])
        stats = self.bench.compare_output(expected + 1e-3, expected)
        self.assertGreater(stats["relative_l2"], 0)
        self.assertLess(stats["relative_l2"], 0.02)
        invalid = [
            (expected * 2, expected),
            (expected[:, :1], expected),
            (expected * float("nan"), expected),
            (expected, expected * float("nan")),
        ]
        for actual, reference in invalid:
            with self.subTest(actual=actual, reference=reference), self.assertRaises(AssertionError):
                self.bench.compare_output(actual, reference)


class StandaloneExpertLauncherTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="expert-bench-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.case = self.root / "shared case [512].pt"
        self.output = self.root / "reports with spaces" / "run 1"
        # Nonexistent binaries prove dry-run does not import a GPU framework or
        # try to run an installed profiler. Dollar signs stay literal argv data.
        self.python = str(self.root / "Python env" / "bin" / "python")
        self.ncu = str(self.root / "NCU install $literal" / "ncu")

    def launch(self, *args, env=None):
        environment = dict(os.environ, PYTHON_BIN=self.python, NCU_BIN=self.ncu,
                           CUDA_VISIBLE_DEVICES="2")
        if env:
            environment.update(env)
        return subprocess.run(
            ["bash", str(LAUNCHER), *args], env=environment, cwd=REPO,
            text=True, capture_output=True, timeout=10,
        )

    def dry_run(self, backend, stage, *extra):
        result = self.launch(backend, stage, "--case", str(self.case),
                             "--output-dir", str(self.output), "--dry-run", *extra)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines()
                    if line.startswith("  ")]
        self.assertEqual(len(commands), 2, result.stdout)
        return result, commands

    def test_dry_run_preserves_shared_case_and_paths_for_both_backends(self):
        for backend in ("streaminfer", "sglang"):
            for stage in ("up", "down", "activation", "full"):
                with self.subTest(backend=backend, stage=stage):
                    result, (timing, capture) = self.dry_run(backend, stage)
                    self.assertEqual(timing[0], self.python)
                    self.assertEqual(capture[0], self.ncu)
                    self.assertEqual(timing[1], str(OPS / "expert_bench.py"))
                    python_index = capture.index(self.python)
                    self.assertEqual(capture[python_index + 1], timing[1])
                    for command in (timing, capture):
                        self.assertEqual(command[command.index("--case") + 1], str(self.case))
                        self.assertEqual(command[command.index("--backend") + 1], backend)
                        self.assertEqual(command[command.index("--stage") + 1], stage)
                        self.assertNotIn("nsys", command)
                        self.assertNotIn("torchrun", command)
                    self.assertNotIn("--profile-only", timing)
                    self.assertIn("--profile-only", capture)
                    self.assertEqual(timing[timing.index("--output") + 1],
                                     str(self.output / "timing.json"))
                    self.assertEqual(capture[capture.index("--output") + 1],
                                     str(self.output / "capture.json"))
                    self.assertIn("CUDA_VISIBLE_DEVICES=2", result.stdout)
                    self.assertFalse(self.output.exists())
                    self.assertFalse(self.case.exists())
                    self.assertEqual(list(self.root.iterdir()), [])

    def test_capture_selects_post_warmup_stage_and_same_cache_policy(self):
        for backend in ("streaminfer", "sglang"):
            with self.subTest(backend=backend):
                _, (_, capture) = self.dry_run(backend, "down", "--detailed")
                expected = {
                    "--nvtx-include": "ExpertBench.down/",
                    "--profile-from-start": "off",
                    "--replay-mode": "kernel",
                    "--cache-control": "all",
                    "--clock-control": "none",
                }
                for flag, value in expected.items():
                    self.assertEqual(capture[capture.index(flag) + 1], value)
                sections = [capture[i + 1] for i, part in enumerate(capture)
                            if part == "--section"]
                self.assertCountEqual(sections, ["SpeedOfLight", "LaunchStats", "Occupancy",
                                                 "ComputeWorkloadAnalysis", "MemoryWorkloadAnalysis"])
                self.assertNotIn("--launch-count", capture)
                self.assertNotIn("--launch-skip", capture)

    def test_invalid_invocations_fail_before_creating_reports(self):
        cases = [
            (("unknown", "full", "--case", str(self.case)), None),
            (("streaminfer", "unknown", "--case", str(self.case)), None),
            (("streaminfer", "up"), None),
            (("streaminfer", "up", "--case"), None),
            (("sglang", "full", "--case", str(self.case)), {"CUDA_VISIBLE_DEVICES": "0,1"}),
            (("sglang", "full", "--case", str(self.case)), {"CUDA_VISIBLE_DEVICES": ""}),
        ]
        for args, env in cases:
            with self.subTest(args=args, env=env):
                result = self.launch(*args, "--output-dir", str(self.output), "--dry-run", env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ERROR:", result.stderr)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_report_is_preserved_without_launching_commands(self):
        self.case.touch()
        self.output.mkdir(parents=True)
        sentinel = self.output / "keep.txt"
        sentinel.write_text("previous report", encoding="utf-8")
        for binary in (self.python, self.ncu):
            path = Path(binary)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/bin/sh\necho SHOULD_NOT_RUN\nexit 99\n", encoding="utf-8")
            path.chmod(0o755)
        result = self.launch("streaminfer", "up", "--case", str(self.case),
                             "--output-dir", str(self.output))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output already exists", result.stderr)
        self.assertNotIn("SHOULD_NOT_RUN", result.stdout + result.stderr)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "previous report")
        self.assertEqual(list(self.output.iterdir()), [sentinel])


class SGLangAdapterGuardTest(unittest.TestCase):
    def test_invalid_stage_fails_without_loading_sglang(self):
        with mock.patch.object(adapter, "version") as version:
            with self.assertRaisesRegex(ValueError, "stage"):
                adapter.build_sglang_runner({}, "router")
        version.assert_not_called()

    def test_wrong_sglang_release_fails_before_gpu_imports(self):
        for installed_version in ("0.5.8", "0.5.10", "0.5.9.post1", "0.5.9rc1"):
            with self.subTest(installed_version=installed_version), \
                    mock.patch.object(adapter, "version", return_value=installed_version):
                with self.assertRaisesRegex(RuntimeError, "supports sglang==0.5.9"):
                    adapter.build_sglang_runner({}, "full")


if __name__ == "__main__":
    unittest.main()
