"""Causality checks for the StreamInfer host/GPU pipeline, using synthetic costs."""

from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from simulation.overheads import BatchOverheadModel, BatchWorkload
from simulation.sim_async import run_simulation


class HostPipelineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.profile = self.root / "expert.csv"
        self.profile.write_text("batch_size,time_ms\n" + "".join(
            f"{size},0.011\n" for size in range(1, 17)))

    def run_case(self, *, requests=1, ranks=1, experts=(0,), costs=None, **options):
        routing = self.root / "routing.parquet"
        rows = [
            dict(rid=request, token_index=0, layer=layer, **{
                f"expert_logical_k{slot}": expert for slot, expert in enumerate(experts)})
            for request in range(requests)
            for layer in range(options.get("num_layers", 1))
        ]
        pq.write_table(pa.Table.from_pylist(rows), routing)
        # One tick per microsecond. Holding all workload coefficients at zero
        # lets each expected timeline follow directly from its stage constants.
        pipeline = dict(
            preset="pipeline", cpu_fixed_us=2, cpu_per_queue_us=0, cpu_per_row_us=0,
            cpu_per_fragment_us=0, cpu_pre_attention_us=4, cpu_pre_expert_us=3,
            cpu_launch_us=1, cpu_post_attention_us=5, cpu_post_expert_us=2,
            reformation_scale=0,
        )
        pipeline.update(costs or {})
        arguments = dict(
            overhead_config=pipeline, ticks_per_millisecond=1000,
            ep_group_size=ranks, attn_dp_group_size=ranks,
            total_expert_count=max(max(experts) + 1, ranks), routing_top_k=len(experts),
            num_layers=1, total_requests=requests, tokens_per_request=1, arrival_rate=0,
            global_request_max_batch_size=requests, max_batch_size=1,
            max_attention_batch_size=1, finalize_per_token_t=0,
            attn_service_t=20, expert_profile_path=str(self.profile),
            profile_routing_path=str(routing), schedule_trace_limit=100, quiet=True,
        )
        arguments.update(options)
        return run_simulation(**arguments)

    def test_isolated_batch_flushes_and_postprocess_releases_output(self):
        # No other batch can wake this worker until it publishes its own output.
        result = self.run_case()
        attention, expert, finalize = result["schedule_trace"]
        self.assertEqual([row["stage"] for row in (attention, expert, finalize)],
                         ["attention", "expert", "finalize"])
        self.assertEqual((attention["start"], attention["gather_submit"],
                          attention["compute_submit"], attention["compute_end"]), (0, 2, 7, 27))
        self.assertEqual((attention["post_start"], attention["output_ready"], expert["start"]),
                         (27, 32, 32))
        self.assertEqual((expert["compute_submit"], expert["compute_end"],
                          expert["output_ready"], finalize["start"]), (38, 49, 51, 51))
        self.assertIsNone(expert["gather_start"])
        self.assertEqual(result["gpu_compute_busy_ticks"], [31])
        self.assertEqual(result["completed_requests"], 1)
        self.assertEqual(finalize["output_ready"], result["makespan"])

    def test_one_batch_can_prepare_ahead_with_one_serial_gpu_stream(self):
        result = self.run_case(requests=4)
        rows = result["schedule_trace"]
        first, second, third = rows[:3]
        self.assertEqual(first["stage"], second["stage"])
        self.assertLess(second["compute_submit"], first["compute_end"])
        self.assertEqual(second["compute_start"], first["compute_end"])
        # A queue of arbitrarily many prepared batches would let the engine
        # select batch three before the first one's normal postprocess.
        self.assertGreaterEqual(third["start"], first["output_ready"])
        gpu_rows = [r for r in rows if r["compute_end"] > r["compute_start"]]
        for previous, current in zip(gpu_rows, gpu_rows[1:]):
            self.assertGreaterEqual(current["compute_start"], previous["compute_end"])
        for retired, later in zip(rows, rows[2:]):
            self.assertGreaterEqual(later["start"], retired["output_ready"])
        self.assertEqual(result["completed_requests"], 4)
        self.assertEqual(result["gpu_compute_busy_ticks"], [4 * (20 + 11)])

    def test_host_only_finalize_publishes_while_another_expert_is_running(self):
        rows = self.run_case(requests=3)["schedule_trace"]
        finalize = next(r for r in rows if r["stage"] == "finalize")
        previous = rows[rows.index(finalize) - 1]
        self.assertTrue(finalize["host_only_finalize"])
        self.assertIsNone(finalize["gather_start"])
        self.assertEqual(finalize["output_ready"], finalize["compute_submit"])
        self.assertEqual(previous["stage"], "expert")
        self.assertLess(previous["compute_start"], finalize["output_ready"])
        self.assertLess(finalize["output_ready"], previous["compute_end"])

    def test_gather_and_host_preparation_overlap_without_double_counting(self):
        # Cover both orders of predecessors: CPU ready first and GPU ready first.
        for prepare_us in (2, 50):
            with self.subTest(prepare_us=prepare_us):
                result = self.run_case(
                    experts=(0, 1), num_layers=2, max_batch_size=2,
                    costs=dict(reformation_scale=1, reformation_fixed_us=10,
                               cpu_pre_attention_us=prepare_us))
                rows = result["schedule_trace"]
                first = next(r for r in rows if r["stage"] == "attention" and r["layer"] == 0)
                self.assertIsNone(first["gather_start"])  # one contiguous admission buffer
                attn = next(r for r in rows if r["stage"] == "attention" and r["layer"] == 1)
                self.assertEqual(attn["gather_start"], attn["gather_submit"])
                self.assertGreater(attn["gather_end"], attn["gather_start"])
                self.assertEqual(attn["compute_submit"], attn["gather_submit"] + prepare_us + 1)
                self.assertEqual(attn["compute_start"], max(attn["gather_end"], attn["compute_submit"]))
                self.assertEqual(attn["compute_end"] - attn["compute_start"], 20)
                expert = next(row for row in rows if row["stage"] == "expert")
                self.assertIsNone(expert["gather_start"])  # one contiguous dispatch buffer
                # Top-2 outputs must also be reduced before Finalize publishes.
                finalize = rows[-1]
                self.assertGreater(finalize["gather_end"], finalize["gather_start"])
                expected_busy = sum(
                    (r["gather_end"] - r["gather_start"] if r["gather_start"] is not None else 0)
                    + r["compute_end"] - r["compute_start"] for r in rows)
                self.assertAlmostEqual(expected_busy, result["gpu_compute_busy_ticks"][0])

    def test_top_eight_join_depends_on_remote_receive_after_expert_post(self):
        result = self.run_case(
            ranks=2, experts=tuple(range(8)), max_batch_size=8,
            net_delay_fn=lambda src, dst, count: 13 if (src, dst) == (0, 1) else 11)
        rows = result["schedule_trace"]
        attention = next(r for r in rows if r["stage"] == "attention")
        experts = {r["gpu"]: r for r in rows if r["stage"] == "expert"}
        final = next(r for r in rows if r["stage"] == "finalize")
        self.assertEqual({rank: r["batch_size"] for rank, r in experts.items()}, {0: 4, 1: 4})
        self.assertEqual(experts[0]["start"], attention["output_ready"])
        self.assertEqual(experts[1]["start"], attention["output_ready"] + 13)
        self.assertEqual(final["start"], max(experts[0]["output_ready"], experts[1]["output_ready"] + 11))
        self.assertEqual(result["incomplete_joins_at_end"], 0)
        self.assertEqual(result["inflight_network_branches_at_end"], 0)

    def test_host_costs_and_selected_rows_do_not_depend_on_trace_retention(self):
        recorded = self.run_case(requests=4)
        hidden = self.run_case(requests=4, schedule_trace_limit=0)
        self.assertEqual(hidden["schedule_trace"], [])
        for field in ("completion_times", "gpu_compute_busy_ticks", "host_busy_ticks_by_phase",
                      "attention_batch_histogram", "expert_group_batch_histogram"):
            self.assertEqual(recorded[field], hidden[field], field)

    def test_early_stop_clips_resource_demand_to_elapsed_time(self):
        # The remote expert owner has work in flight when the first home rank
        # emits its response. Planned future GPU completions must not be billed.
        result = self.run_case(requests=8, ranks=2, experts=(1,),
                               tokens_after_reaching_max_bs=1,
                               net_delay_fn=lambda *_: 20)
        self.assertTrue(result["stopped_early"])
        horizon = result["makespan"]
        rows = result["schedule_trace"]
        self.assertTrue(any(r["compute_start"] < horizon < r["compute_end"] for r in rows))
        for gpu in (0, 1):
            expected = sum(max(0, min(horizon, r["compute_end"]) - r["compute_start"])
                           for r in rows if r["gpu"] == gpu)
            self.assertAlmostEqual(result["gpu_compute_busy_ticks"][gpu], expected)
            phases = result["gpu_busy_ticks_by_phase"][gpu]
            self.assertAlmostEqual(sum(phases.values()), expected)

    def test_legacy_serial_and_explicit_none_still_agree(self):
        default = self.run_case(requests=4, overhead_config=None)
        explicit = self.run_case(requests=4, overhead_config={"preset": "none"})
        self.assertEqual(default["schedule_trace"], explicit["schedule_trace"])
        self.assertEqual(default["completion_times"], explicit["completion_times"])
        rows = default["schedule_trace"]
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(rows, rows[1:])))


class PipelineCostTests(unittest.TestCase):
    def model(self, **overrides):
        settings = dict(
            preset="pipeline", ticks_per_millisecond=250,
            cpu_fixed_us=2, cpu_per_row_us=0.25, cpu_per_fragment_us=1,
            cpu_per_queue_us=0.5, cpu_gather_tail_us=7, cpu_reduce_tail_us=13,
            cpu_pre_attention_us=11, cpu_pre_expert_us=3, cpu_pre_per_row_us=1.5,
            cpu_launch_us=1, cpu_post_attention_us=8, cpu_post_expert_us=6,
            cpu_post_per_row_us=2, cpu_finalize_us=9, reformation_scale=2,
            reformation_fixed_us=4, reformation_bandwidth_gb_s=10,
            reformation_per_fragment_us=0.5, reduction_fixed_us=12, reduction_bandwidth_gb_s=20,
        )
        settings.update(overrides)
        return BatchOverheadModel(**settings)

    def test_gather_and_fused_reduction_have_separate_costs_in_resource_ticks(self):
        model = self.model()
        # B=4 outputs for both stages. N=32 incoming rows for Top-8 Attention.
        # A microsecond is 0.25 ticks; decimal GB/s is 1000 bytes/microsecond.
        gather_input = BatchWorkload(4, 2, 8, 1000, 2, reduce_topk=False)
        reduction_input = BatchWorkload(32, 4, 8, 1000, 2, reduce_topk=True)
        expert = model.evaluate("expert", 10, 4, gather_input)
        attention = model.evaluate("attention", 10, 4, reduction_input)
        self.assertAlmostEqual(expert.reformation_raw_ticks, 0.25 * 2 * (4 + 16_000 / 10_000 + 1))
        self.assertAlmostEqual(attention.reformation_raw_ticks, 0.25 * 2 * (12 + 72_000 / 20_000 + 2))
        self.assertEqual(attention.gpu_ticks, 10 + attention.reformation_raw_ticks)
        self.assertEqual(expert.cpu_dequeue_ticks, 0.25 * (2 + 0.25 * 4 + 2 + 0.5 * 8))
        self.assertEqual(attention.cpu_dequeue_ticks, 0.25 * (2 + 0.25 * 32 + 4 + 0.5 * 8))
        self.assertEqual(expert.cpu_prelaunch_ticks, 0.25 * (7 + 3 + 1.5 * 4 + 1))
        self.assertEqual(attention.cpu_prelaunch_ticks, 0.25 * (13 + 11 + 1.5 * 4 + 1))
        self.assertEqual(expert.cpu_post_ticks, 0.25 * (6 + 2 * 4))
        self.assertEqual(attention.cpu_post_ticks, 0.25 * (8 + 2 * 4))
        self.assertEqual(expert.details["components"]["reformation_kind"], "gather")
        self.assertEqual(attention.details["components"]["reformation_kind"], "gather_and_sum")

    def test_bypass_charges_no_gather_or_gather_tail(self):
        inputs = BatchWorkload(4, 1, 8, 1000, 2, contiguous=True, reduce_topk=False)
        for stage, expected_pre_us in (("expert", 3 + 1.5 * 4 + 1),
                                       ("attention", 11 + 1.5 * 4 + 1), ("finalize", 9)):
            with self.subTest(stage=stage):
                service = self.model().evaluate(stage, 10, 4, inputs)
                self.assertEqual(service.details["components"]["reformation_kind"], "bypass")
                self.assertEqual(service.reformation_raw_ticks, 0)
                self.assertEqual(service.cpu_prelaunch_ticks, expected_pre_us * 0.25)
                self.assertGreater(service.cpu_dequeue_ticks, 0)
        self.assertEqual(self.model().evaluate("finalize", 10, 4, inputs).cpu_post_ticks, 0)

    def test_config_roundtrip_and_validation(self):
        original = self.model()
        restored = BatchOverheadModel(**original.config(), ticks_per_millisecond=250)
        self.assertEqual(restored.config(), original.config())
        for args in ({"cpu_schedule_overlap": 0.5}, {"reformation_overlap": 0.25},
                     {"reduction_bandwidth_gb_s": 0}, {"cpu_reduce_tail_us": -1},
                     {"cpu_pre_per_row_us": float("inf")}, {"cpu_post_per_row_us": True}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.model(**args)
        for preset in ("none", "paper", "structural"):
            with self.subTest(preset=preset), self.assertRaises(ValueError):
                BatchOverheadModel(preset=preset, reduction_fixed_us=12)


if __name__ == "__main__":
    unittest.main()
