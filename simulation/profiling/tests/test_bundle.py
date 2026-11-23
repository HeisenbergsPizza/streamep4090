from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from simulation.profiling import bundle, manifest
from simulation.profiling.common import FORMAT, VERSION, StageWriter, load_manifest, write_json
from simulation.profiling.__main__ import build_parser
from simulation.profiling.__main__ import main as profile_main
from simulation import run, steady_experiment


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.root = self.directory / "profile"
        self.root.mkdir()
        self.gpus = [dict(name="test GPU", sm="8.9", logical_index=i) for i in range(2)]
        self.config = dict(format=FORMAT, version=VERSION,
                           model={**manifest.QWEN, "num_layers": 1, "num_experts": 4, "top_k": 2},
                           hardware={"gpus": self.gpus, "interconnect": "pcie"},
                           parallel=dict(dp=2, ep=2, tp=1),
                           runtime=dict(cuda_graph=True, dtype="bfloat16"))
        write_json(self.root / "manifest.json", self.config)

    def publish(self):
        with StageWriter(self.root, "expert") as stage:
            (stage / "expert.csv").write_text("batch_size,time_ms\n1,0.09\n2,0.1\n")
        with StageWriter(self.root, "attention") as stage:
            with (stage / "attention.csv").open("w") as f:
                f.write("batch_size,context_length,time_ms\n")
                for b, c in itertools.product((1, 3), (64, 68)):
                    f.write(f"{b},{c},0.01\n")
        with StageWriter(self.root, "network") as stage:
            write_json(stage / "parameters.json", {"--network-latency-us": 8.0,
                                                  "--network-bandwidth-gbps": 93.0})
        with StageWriter(self.root, "grouped") as stage:
            vectors = ([0, 1], [1, 0], [2, 0], [1, 1], [0, 2])
            with (stage / "grouped.csv").open("w") as f:
                f.write("expert_counts,time_ms\n")
                for vector in vectors:
                    f.write(f'\"{json.dumps(vector)}\",0.11\n')

    def test_load_hf_moe_width(self):
        path = self.directory / "config.json"
        write_json(path, dict(moe_intermediate_size=768, intermediate_size=4096, num_hidden_layers=48,
                              num_experts_per_tok=8, num_experts=128, hidden_size=2048,
                              num_attention_heads=32, head_dim=128, num_key_value_heads=4))
        self.assertEqual(manifest.get_model(path)["intermediate_size"], 768)
        self.assertEqual(manifest.get_model(path)["head_dim"], 128)

    def test_init_records_visible_node_and_collect_uses_separate_stages(self):
        target = self.directory / "new"
        with patch.object(manifest, "visible_gpus", return_value=self.gpus), \
                patch.object(manifest, "command", return_value=None), \
                patch("torch.cuda.can_device_access_peer", return_value=False), redirect_stdout(StringIO()):
            profile_main(["init", "--bundle", str(target), "--interconnect", "pcie", "--cuda-graph"])
        info = load_manifest(target)
        self.assertEqual(info["parallel"], {"dp": 2, "ep": 2, "tp": 1})
        self.assertEqual(info["hardware"]["cuda_peer_access"], [[None, False], [False, None]])
        with patch("simulation.profiling.__main__.subprocess.run") as launch, redirect_stdout(StringIO()):
            profile_main(["collect", "--bundle", str(target), "--stages", "expert", "attention",
                          "--expert-cap", "256", "--contexts", "64,191"])
        commands = [call.args[0] for call in launch.call_args_list]
        self.assertEqual([cmd[3] for cmd in commands], ["expert", "attention"])
        self.assertIn("256", commands[0])
        self.assertIn("64,191", commands[1])

    def test_abort_retry_and_signature(self):
        with self.assertRaises(RuntimeError):
            with StageWriter(self.root, "host") as temp:
                write_json(temp / "parameters.json", {})
                raise RuntimeError("interrupted")
        self.assertFalse((self.root / "host").exists())
        with StageWriter(self.root, "host") as temp:
            write_json(temp / "parameters.json", {})
        self.assertIn("cpu_launch_us", bundle.inspect(self.root)[1]["parameters_without_estimate"]["host"])
        self.config["runtime"]["cuda_graph"] = False
        write_json(self.root / "manifest.json", self.config)
        with self.assertRaisesRegex(ValueError, "manifest"):
            bundle.inspect(self.root)

    def test_selection_coverage_and_tamper(self):
        self.publish()
        args = ["--profile-bundle", str(self.root), "--output-tokens", "2"]
        with redirect_stderr(StringIO()):
            for extra in (["--initial-context", "68"], ["--gpus", "4"], ["--expert-batch-cap", "3"]):
                with self.assertRaises(SystemExit):
                    bundle.parse_simulator_args(run.build_parser(), args + extra)
        parsed = bundle.parse_simulator_args(run.build_parser(), args)
        self.assertEqual(parsed.gpus, 2)
        self.assertIsNone(parsed.grouped_profile)
        self.assertEqual(parsed.overhead_model, "pipeline")
        self.assertEqual(parsed.network_bandwidth_gbps, 93)
        (self.root / "expert" / "expert.csv").write_text("batch_size,time_ms\n1,0.01\n")
        with self.assertRaisesRegex(ValueError, "Changed profile"):
            bundle.inspect(self.root)

    def test_plugins_help_without_gpu(self):
        with redirect_stdout(StringIO()):
            for name in ("expert", "attention", "grouped", "network", "reformation", "collect", "host", "init"):
                with self.assertRaises(SystemExit) as exit_code:
                    build_parser().parse_args([name, "--help"])
                self.assertEqual(exit_code.exception.code, 0)

    def test_both_cli_entry_points(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.publish()
        parquet = self.directory / "routes.parquet"
        pq.write_table(pa.table(dict(rid=[0, 0], token_index=[0, 1], layer=[0, 0],
                                     expert_logical_k0=[0, 1], expert_logical_k1=[2, 3])), parquet)
        args = ["--profile-bundle", str(self.root), "--routing-profile", str(parquet),
                "--use-bundle-grouped", "--output-tokens", "2", "--active-request-cap", "2"]
        with redirect_stdout(StringIO()):
            run.main(args + ["--requests", "2", "--output-dir", str(self.directory / "finite")])
            steady_experiment.main(args + ["--policy", "defrag", "--rate", "200",
                                           "--warmup-s", "0", "--duration-s", "0.04", "--block-s", "0.02",
                                           "--trace-duration-ms", "0", "--output-dir", str(self.directory / "paced")])
        finite = json.loads((self.directory / "finite" / "inputs.json").read_text())
        self.assertEqual(finite["configuration"]["expert_batch_cap"], 2)
        info = finite["provenance"]["profile_bundle"]
        self.assertEqual(info["applied_parameters"]["network_latency_us"], 8)
        result = json.loads((self.directory / "paced" / "result.json").read_text())
        self.assertTrue(result["validation"]["fully_drained"])
        self.assertGreater(result["simulation"]["completed_tokens"], 0)
        self.assertIn("sha256", result["configuration"]["grouped_profile"])

    def test_topology(self):
        self.config["parallel"]["tp"] = 2
        write_json(self.root / "manifest.json", self.config)
        with self.assertRaisesRegex(ValueError, "TP=1"):
            load_manifest(self.root)
