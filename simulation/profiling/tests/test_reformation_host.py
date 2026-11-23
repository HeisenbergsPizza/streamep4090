"""Exercise kernel cost fitting and engine-thread associations without a GPU."""

import argparse
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from simulation.profiling import host, reformation
from simulation.profiling.common import FORMAT, VERSION


class ReformationTests(unittest.TestCase):
    def test_distinct_spans_and_bypasses(self):
        self.assertEqual(reformation.source_indices(8, 3), [0, 1, 2, 4, 5, 6, 8, 9])
        reordered = reformation.source_indices(16, 4, 3, "shuffled")
        spaced = reformation.source_indices(16, 4, 3, "forward")
        self.assertCountEqual(reordered, spaced)
        self.assertNotEqual(reordered, spaced)
        cases = list(reformation.points([2], [1, 2, 4, 32], 8))
        self.assertEqual([(x["kind"], x["input_rows"], x["fragments"]) for x in cases],
                         [("bypass", 2, 1), ("gather", 2, 2),
                          ("gather_and_sum", 16, 1), ("gather_and_sum", 16, 2),
                          ("gather_and_sum", 16, 4)])
        self.assertEqual({p["kind"] for p in reformation.points([2], [1, 2], 1)}, {"gather", "bypass"})

    def test_fit_preserves_simulator_units_and_separate_reduction(self):
        cases = []
        for p in reformation.points([4, 20, 200], [1, 2, 4], 8):
            b = (p["input_rows"] + p["output_rows"]) * 2048 * 2
            fixed, bw = (2, 200) if p["kind"] == "gather" else (3, 80)
            cases.append(dict(**p, traffic_bytes=b, median_kernel_us=fixed+b/(1000*bw)+.25*p["fragments"]))
        parameters, diagnostic = reformation.fit(cases)
        self.assertLess(diagnostic["rmse_us"], 1e-8)
        self.assertAlmostEqual(parameters["--reformation-bandwidth-gb-s"], 200)
        self.assertAlmostEqual(parameters["--reduction-bandwidth-gb-s"], 80)
        self.assertAlmostEqual(parameters["--reformation-fixed-us"], 2)
        self.assertAlmostEqual(parameters["--reduction-fixed-us"], 3)
        self.assertAlmostEqual(parameters["--reformation-per-fragment-us"], .25)
        self.assertFalse(reformation.fit(cases[1:3])[0])


class HostTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.executescript("""
          CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER);
          CREATE TABLE StringIds(id INTEGER, value TEXT);
          CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER, nameId INTEGER, correlationId INTEGER);
          CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, deviceId INTEGER, globalPid INTEGER, demangledName INTEGER, graphId INTEGER, correlationId INTEGER);
          CREATE TABLE OSRT_API(start INTEGER, end INTEGER, globalTid INTEGER, nameId INTEGER);
          INSERT INTO StringIds VALUES(1,'cudaLaunchKernel_v7000'),(2,'void gather_and_sum_tokens_kernel<...>'),
              (3,'cudaStreamSynchronize_v3020'),(4,'pthread_cond_timedwait'),(5,'Engine.preprocess');
          INSERT INTO NVTX_EVENTS VALUES
           (50,410,7,'Engine.schedule.sample_1_in_64',NULL),(425,430,7,'Engine.schedule.sample_1_in_64',NULL),
           (100,400,7,'Scheduler::dequeue_merge_unified',NULL),(450,850,7,'WG.work=7',NULL),
           (470,580,7,NULL,5),(600,800,7,'Engine.execute.expert.layer=4.batch=9',NULL),
           (1050,1150,7,'Engine.previous_event_wait',NULL),(1160,NULL,7,'WG.retire=7',NULL),
           (1200,1300,7,'Engine.postprocess',NULL);
          INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(200,230,7,1,9),(500,550,7,3,18),(100,500,8,3,19);
          INSERT INTO OSRT_API VALUES(520,570,7,4);
          INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(250,410,0,0,2,NULL,9),(200,250,0,16777216,2,NULL,9);
        """)

    def test_thread_known_wait_union_and_kernel_join(self):
        self.assertEqual(host.thread_id(self.db, None), 7)
        data = {r["phase"]: r for r in host.samples(self.db, 7, 0, 2000)}
        self.assertEqual(data["sampled_selection_to_dequeue"]["wall_us"], .05)
        self.assertEqual(data["sampled_schedule_no_observed_dequeue"]["wall_us"], .005)
        self.assertEqual(data["gather_launch_api"]["sampled_schedule_to_submit_residual_us"], .18)
        self.assertEqual(data["gather_launch_api"]["gpu_end_ns"], 410)
        self.assertEqual(data["dequeue_after_launch"]["wall_us"], .17)
        self.assertEqual(data["pre"]["known_wait_us"], .07)
        self.assertEqual(data["pre"]["residual_wall_us"], .04)
        self.assertEqual(data["post"]["stage"], "expert")
        self.assertEqual(data["post"]["batch_rows"], 9)
        self.assertEqual(data["previous_event_wait"]["residual_wall_us"], 0)
        # Window starts mid-work: named phases can still be timed but a post
        # with no retired work mapping must not be attached to the next batch.
        clipped = {r["phase"]: r for r in host.samples(self.db, 7, 700, 1300)}
        self.assertEqual(clipped["post"]["stage"], "unknown")
        self.assertNotIn("execute", clipped)

    def test_graph_and_other_thread_launch_do_not_label_cpu_tail(self):
        self.db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphId=2")
        data = {r["phase"] for r in host.samples(self.db, 7, 0, 2000)}
        self.assertIn("dequeue_unresolved", data)
        self.assertNotIn("dequeue_after_launch", data)

    def test_opt_in_wall_parameters_do_not_double_charge_shape_terms(self):
        # Five fully resolved, sampled submissions trigger the opt-in fixed proxy.
        for copy in range(1, 5):
            offset = copy * 2000
            self.db.execute("""INSERT INTO NVTX_EVENTS SELECT start+?,end+?,globalTid,text,textId
                               FROM NVTX_EVENTS WHERE start<1500""", (offset, offset))
            self.db.execute("""INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME SELECT start+?,end+?,globalTid,
                               nameId,correlationId+? FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start<1500""",
                            (offset, offset, offset))
            self.db.execute("""INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL SELECT start+?,end+?,deviceId,
                               globalPid,demangledName,graphId,correlationId+?
                               FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start<1500""",
                            (offset, offset, offset))
        self.db.commit()
        with TemporaryDirectory() as temp:
            path = Path(temp)
            report = path / "input.sqlite"
            with sqlite3.connect(report) as target:
                self.db.backup(target)
            model = dict(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_layers=3)
            for opt_in in (False, True):
                bundle = path / str(opt_in)
                bundle.mkdir()
                (bundle / "manifest.json").write_text(json.dumps(dict(format=FORMAT, version=VERSION,
                    model=model, parallel=dict(dp=2, ep=2, tp=1), runtime=dict(dtype="bfloat16", cuda_graph=False),
                    hardware=dict(interconnect="pcie", gpus=[dict(logical_index=i) for i in range(2)]))))
                host.run(argparse.Namespace(bundle=bundle, sqlite=report, start_ns=0, end_ns=20000,
                                            tid=None, use_wall_proxy=opt_in))
                params = json.loads((bundle / "host/parameters.json").read_text())
                if opt_in:
                    self.assertEqual(params["--cpu-fixed-us"], .18)
                    self.assertEqual(params["--cpu-reduce-tail-us"], .17)
                    for key in ("row", "fragment", "queue"):
                        self.assertEqual(params[f"--cpu-per-{key}-us"], 0)
                else:
                    self.assertEqual(params, {})

    def test_multiple_workers_require_selection(self):
        self.db.execute("INSERT INTO NVTX_EVENTS VALUES(101,190,11,'Scheduler::dequeue_merge_unified',NULL)")
        with self.assertRaisesRegex(ValueError, "Select one engine"):
            host.thread_id(self.db, None)
        self.assertEqual(host.thread_id(self.db, 11), 11)
