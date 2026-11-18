"""Clock and counting checks for the real decode arrival-rate experiment."""

import asyncio
import ast
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import List
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT))
import paced_decode
from disagmoe.frontend.request_stats import SloStat
from disagmoe.utils.decode_snapshot import FORMAT, checkpoint_metadata, kv_name


class PacedWindowTest(unittest.TestCase):
    def test_half_open_window_and_waiting_vs_outstanding(self):
        rows = [
            dict(scheduled_s=0, submitted_s=0, admitted_s=.5, token_times_s=[1, 2], response_s=2.5),
            dict(scheduled_s=1, submitted_s=1, admitted_s=3.5, token_times_s=[4, 5], response_s=5.5),
            dict(scheduled_s=2, submitted_s=2, admitted_s=2.1, token_times_s=[2.3, 3], response_s=3),
        ]
        window = paced_decode.summarize_window(rows, 1, 3)
        self.assertEqual(window, dict(
            start_s=1, end_s=3, scheduled_requests=2, completed_requests=1, output_tokens=3,
            req_throughput=.5, token_throughput=1.5, outstanding_start=1, outstanding_end=2,
            waiting_admission_start=0, waiting_admission_end=1))
        empty = paced_decode.summarize_window(rows, 0, .25)
        self.assertEqual((empty["completed_requests"], empty["output_tokens"]), (0, 0))


class PacedDriverTest(unittest.IsolatedAsyncioTestCase):
    async def test_submissions_follow_deadlines_while_all_requests_are_outstanding(self):
        now = 10.0
        dispatched = []
        all_submitted = asyncio.Event()
        yield_task = asyncio.sleep

        async def sleep(delay):
            nonlocal now
            now += delay
            await yield_task(0)

        def submit(index, input_len, output_len):
            nonlocal now
            dispatched.append(now)
            # Charge client time to each submission. A relative sleep between
            # submissions would drift; absolute deadlines stay at 10, 10.5, 11.
            now += .25
            admitted = now
            if len(dispatched) == 3:
                all_submitted.set()

            async def get():
                await all_submitted.wait()
                return SimpleNamespace(req_id=7+index, t_token_timestamps=[now])
            return SimpleNamespace(req_id=7+index, admitted_at=admitted, get=get)

        args = SimpleNamespace(num_requests=3, rate=2, min_input_len=64, min_output_len=1,
                               arrival_window=None)
        with patch.object(paced_decode, "time", SimpleNamespace(perf_counter=lambda: now)), \
                patch.object(paced_decode.asyncio, "sleep", sleep):
            results, report = await asyncio.wait_for(paced_decode.run_paced_requests(
                SimpleNamespace(put_snapshot_request=submit), args), 1)
        self.assertEqual(dispatched, [10, 10.5, 11])
        self.assertEqual([r.req_id for r in results], [7, 8, 9])
        self.assertEqual(report["window"]["end_s"], 1.5)
        self.assertEqual(report["duration_s"], 1.25)
        self.assertEqual(report["window"]["output_tokens"], 3)
        self.assertEqual([r["scheduled_s"] for r in report["requests"]], [0, .5, 1])


class PacedReportTest(unittest.TestCase):
    def test_validation_and_timing_artifacts_through_post_benchmark(self):
        try:
            import numpy as np
            import pandas as pd
        except ImportError:
            self.skipTest("CSV reporting needs the benchmark's numpy and pandas dependencies")
        # Isolate the real reporting functions from the GPU/Ray imports.
        filename = ROOT / "benchmark/benchmark_serving.py"
        tree = ast.parse(filename.read_text())
        tree.body = [n for n in tree.body if getattr(n, "name", None) in
                     ("BenchmarkMetrics", "analyze_results", "post_benchmark")]
        scope = dict(np=np, os=os, json=json, List=List, SloStat=SloStat, dataclass=dataclass,
                     logger=logging.getLogger("paced-report-test"))
        exec(compile(tree, str(filename), "exec"), scope)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(dict(model_type="qwen3_moe",
                num_hidden_layers=48, num_key_value_heads=4, head_dim=128, vocab_size=32)))
            (root / "model.safetensors.index.json").write_text('{"weight_map":{}}')
            (root / kv_name(0)).touch()
            (root / "manifest.json").write_text(json.dumps(dict(format=FORMAT, repeat=2,
                checkpoint_metadata_sha256=checkpoint_metadata(root),
                prompts=[dict(input_ids=[1, 2], first_token=3, expected_tokens=[4])])))
            args = SimpleNamespace(file=str(root / "metrics.csv"), checkpoint_path=tmp,
                decode_snapshot=tmp, paced_decode=True, min_output_len=1, trace=False,
                analyze_throughput=False, num_requests=2, rate=1, dp_size=4, ep_size=4, num_experts=128)
            Path(args.file).write_text("num_requests,rate\n1,1\n")
            rows = [dict(req_id=i+10, snapshot_index=i, scheduled_s=i, submitted_s=i,
                         admitted_s=i+.1, token_times_s=[i+.2], response_s=i+.3) for i in range(2)]
            report = dict(window=paced_decode.summarize_window(rows, 0, 1), requests=rows)
            results = [SloStat(r["req_id"], 0, 0, 0, [], r["token_times_s"], [4]) for r in rows]

            def write_report():
                return scope["post_benchmark"](None, args, results, [0, 1], [.3, 1.3], 1.3, report)

            write_report()
            csv = pd.read_csv(args.file)
            self.assertEqual(len(csv), 2)
            self.assertEqual(csv.iloc[-1].window_token_throughput, 1)
            self.assertEqual(csv.iloc[-1].window_req_throughput, 1)
            self.assertAlmostEqual(csv.iloc[-1].req_latency_mean_ms, 300)
            self.assertAlmostEqual(csv.iloc[-1].window_req_latency_p99_ms, 300)
            self.assertEqual(json.loads(Path(args.file + ".paced.json").read_text()), report)
            self.assertTrue(json.loads(Path(args.file + ".decode.json").read_text())["matched_reference_prefix"])

            args.file = str(root / "mismatch.csv")
            results[0].token_ids = [5]
            with self.assertRaisesRegex(ValueError, "differs from reference"):
                write_report()
            self.assertFalse(Path(args.file).exists())
            self.assertEqual(json.loads(Path(args.file + ".decode.json").read_text())["mismatched_requests"], 1)


if __name__ == "__main__":
    unittest.main()
