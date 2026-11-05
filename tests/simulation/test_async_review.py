"""Independent checks of replay compatibility and async event boundaries."""
import csv
import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from simulation import sim_async, sim_sync, sim_tbo


class AsyncReviewTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.trace = self.root / 'trace.parquet'
        self.expert = self.root / 'expert.csv'
        self.expert.write_text('batch_size,time_ms\n1,0.3\n2,0.4\n3,0.5\n4,0.6\n')
        self.write_routes([0, 1])

    def write_routes(self, selected):
        rows = [dict(rid=17, token_index=token, layer=0,
                     **{f'expert_logical_k{k}': eid for k, eid in enumerate(selected)})
                for token in (10, 20)]
        pq.write_table(pa.Table.from_pylist(rows), self.trace)

    def options(self, **overrides):
        options = dict(ep_group_size=1, attn_dp_group_size=1, total_expert_count=2,
                       routing_top_k=2, num_layers=1, total_requests=1,
                       tokens_per_request=2, arrival_rate=0,
                       max_batch_size=4, max_attention_batch_size=4,
                       global_request_max_batch_size=4, initial_context_length=10,
                       expert_profile_path=str(self.expert),
                       profile_routing_path=str(self.trace), attn_service_t=2.0)
        options.update(overrides)
        return options

    def run_async(self, **overrides):
        return sim_async.run_simulation(**self.options(quiet=True, **overrides))

    def test_single_request_costs_match_both_baselines_with_profiles(self):
        grouped = self.root / 'grouped.csv'
        with grouped.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['expert_counts', 'time_ms'])
            writer.writerow(['[1, 1]', 0.2])
        attention = self.root / 'attention.csv'
        attention.write_text('batch_size,context_length,time_ms\n1,10,0.05\n1,11,0.1\n')
        options = self.options(grouped_profile_path=str(grouped), expert_cost_scale=2.0,
                               attention_profile_path=str(attention))
        results = []
        for module in (sim_async, sim_sync, sim_tbo):
            with contextlib.redirect_stdout(io.StringIO()):
                results.append(module.run_simulation(**options))
        for result in results:
            self.assertEqual(result['makespan'], 9.5)
            self.assertEqual(result['completion_times'], {0: 4.5, 1: 9.5})
            self.assertEqual(result['avg_per_expert_batch_size'], 1.0)
            self.assertEqual(result['avg_expert_group_batch_size'], 2.0)

    def component_attention_options(self):
        grouped = self.root / 'component_grouped.csv'
        grouped.write_text('expert_counts,time_ms\n"[1, 1]",0.2\n')
        attention = self.root / 'component_attention.csv'
        with attention.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['batch_size', 'context_length', 'time_ms', 'component'])
            for context, core in ((8, 0.1), (12, 0.5)):
                writer.writerows([(1, context, 0.05, 'qkv_proj'),
                                  (1, context, core, 'attention'),
                                  (1, context, 0.05, 'o_proj')])
        return dict(grouped_profile_path=str(grouped), expert_cost_scale=2,
                    attention_profile_path=str(attention))

    def test_component_context_interpolation_matches_all_three_modes(self):
        options = self.options(**self.component_attention_options())
        results = {}
        for module in (sim_async, sim_sync, sim_tbo):
            trace_options = {'schedule_trace_limit': 1000} if module is sim_async else {}
            with self.subTest(mode=module.__name__), contextlib.redirect_stdout(io.StringIO()):
                result = module.run_simulation(**options, **trace_options)
            results[module] = result
            # At history 10/11: core 3/4 ticks plus .5+.5 projection ticks.
            # Each expert group costs .2 ms * 10 ticks/ms * scale 2 = 4 ticks.
            self.assertEqual(result['completion_times'], {0: 8, 1: 17})
            self.assertEqual(result['makespan'], 17)
            metadata = result['attention_cost_model']
            self.assertEqual(metadata['kind'], 'decode_batch_context_profile')
            self.assertEqual(metadata['context_aggregation'], 'arithmetic_mean')
            self.assertEqual(set(metadata['components']), {'qkv_proj', 'attention', 'o_proj'})
        attention = [event for event in results[sim_async]['schedule_trace']
                     if event['stage'] == 'attention']
        self.assertEqual(len(attention), 2)
        self.assertEqual([event['attention_context_mean'] for event in attention], [10, 11])
        for event, core in zip(attention, (3, 4)):
            self.assertEqual(event['attention_components_ticks'],
                             {'qkv_proj': 0.5, 'attention': core, 'o_proj': 0.5})
            self.assertEqual(sum(event['attention_components_ticks'].values()),
                             event['end'] - event['start'])

    def test_component_trace_collection_does_not_change_async_timing(self):
        options = self.component_attention_options()
        without = self.run_async(**options, schedule_trace_limit=0)
        traced = self.run_async(**options, schedule_trace_limit=1000)
        self.assertEqual(without['schedule_trace'], [])
        self.assertTrue(traced['schedule_trace'])
        for key in ('completion_times', 'makespan', 'schedule_count', 'completed_tokens',
                    'gpu_compute_busy_ticks', 'gpu_stage_busy_ticks', 'metrics_ms',
                    'attention_batch_histogram', 'attention_cost_model'):
            with self.subTest(field=key):
                self.assertEqual(without[key], traced[key])

    def test_dispatch_and_return_count_every_expert_copy(self):
        self.write_routes([2, 3])
        transfers = []

        def delay(src, dst, copies):
            transfers.append((src, dst, copies))
            return float(copies)

        result = self.run_async(ep_group_size=2, attn_dp_group_size=2,
                                total_expert_count=4, tokens_per_request=1,
                                net_delay_fn=delay)
        self.assertEqual(transfers, [(0, 1, 2), (1, 0, 2)])
        self.assertEqual(result['remote_expert_copy_count'], 4)
        self.assertEqual(result['makespan'], 12.0)

    def test_join_releases_once_and_rejects_duplicate_branch(self):
        with patch.object(sim_async.AsyncMoESimulator, 'run', lambda self: self):
            sim = self.run_async()
        token = sim_async.Token(100, 0, 100, 0)
        sim.joins[(100, 0)] = sim_async.Join(token, {0, 1})
        sim.join(sim_async.Branch(token, 0, 0))
        self.assertEqual(sim.ready_attention, 0)
        sim.join(sim_async.Branch(token, 1, 0))
        self.assertEqual(sim.ready_attention, 1)
        self.assertEqual(len(sim.workers[0].attention[1]), 1)
        self.assertEqual(sim.stats['join_wait'].count, 1)
        with self.assertRaisesRegex(RuntimeError, 'duplicate or unexpected'):
            sim.join(sim_async.Branch(token, 1, 0))
        self.assertEqual(sim.ready_attention, 1)

    def test_zero_stop_and_empty_workload_keep_legacy_float_outputs(self):
        for options in (dict(global_request_max_batch_size=1, tokens_after_reaching_max_bs=0),
                        dict(total_requests=0)):
            result = self.run_async(**options)
            self.assertEqual(result['completed_tokens'], 0)
            self.assertEqual(result['makespan'], 0.0)
            for key in ('tail_throughput_req_per_sec', 'avg_request_latency_ms',
                        'p90_request_latency_ms', 'p99_request_latency_ms'):
                self.assertEqual(format(result[key], '.6f'), 'nan')
            self.assertTrue(all(math.isfinite(value) for value in result['gpu_compute_busy_fraction']))

    def test_unreached_saturation_does_not_truncate_workload(self):
        result = self.run_async(global_request_max_batch_size=8,
                                tokens_after_reaching_max_bs=1)
        self.assertEqual(result['completed_tokens'], 2)
        self.assertFalse(result['stopped_early'])
        self.assertIsNone(result['saturation_time'])


if __name__ == '__main__':
    unittest.main()
