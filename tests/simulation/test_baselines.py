"""CPU-only invariants for the synchronous and two-microbatch baselines."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from simulation import sim_sync, sim_tbo
from simulation.costs import ExpertCostModel


class SilentProgress:
    def update(self, *args, **kwargs):
        pass

    def finalize(self):
        pass


class FixedRouter:
    top_k = 2
    num_layers = 2

    def route(self, request_ids, **kwargs):
        ids = torch.tensor([[0, 1]] * len(request_ids), dtype=torch.int32)
        return torch.ones_like(ids, dtype=torch.float32) / 2, ids


def make_simulator(module, **overrides):
    kwargs = dict(
        request_arrivals=[0.0], tokens_per_request=1, num_layers=1,
        total_expert_count=4, ep_group_size=2, max_batch_size=4,
        attn_service_t=2.0, attn_dp_group_size=2,
        net_t_attn_to_expert=0.1, net_t_expert_to_attn=0.1,
        compute_time_lookup={1: 3.0, 2: 4.0, 3: 5.0, 4: 6.0},
        profile_router=FixedRouter(), progress_tracker=SilentProgress(),
        global_request_max_batch_size=8,
    )
    kwargs.update(overrides)
    cls = module.SyncMoESimulator if module is sim_sync else module.TBOMoESimulator
    return cls(**kwargs)


class BaselineCostTests(unittest.TestCase):
    def test_expert_total_cap_and_distribution_are_shared(self):
        for module in (sim_sync, sim_tbo):
            with self.subTest(mode=module.__name__):
                batches = []
                cost = ExpertCostModel({1: 3.0, 2: 4.0, 3: 5.0, 4: 6.0})

                def record(counts):
                    batches.append(list(counts))
                    return cost(counts)

                sim = make_simulator(module, expert_cost_model=record)
                # Two launches: f(3)+f(1), then f(2). No implicit 0.7 multiplier.
                self.assertEqual(sim._simulate_worker_compute_time([3, 3]), 12.0)
                self.assertEqual(batches, [[3, 1], [0, 2]])
                self.assertEqual(sim.total_expert_batch_size, 6)
                self.assertEqual(sim.total_expert_batch_count, 3)
                self.assertEqual(sim.total_expert_group_batch_count, 2)

    def test_grouped_measurement_requires_exact_distribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'grouped.csv'
            with path.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['expert_counts', 'time_ms'])
                writer.writerow(['[3, 1]', 0.7])
            for module in (sim_sync, sim_tbo):
                sim = make_simulator(module, expert_cost_model=ExpertCostModel(
                    {1: 3.0, 2: 4.0, 3: 5.0, 4: 6.0}, grouped_profile_path=path))
                self.assertEqual(sim._simulate_worker_compute_time([3, 1]), 7.0)
                with self.assertRaisesRegex(ValueError, 'no exact measurement'):
                    sim._simulate_worker_compute_time([2, 2])

    def test_attention_uses_home_gpu_batches_and_contexts(self):
        for module in (sim_sync, sim_tbo):
            with self.subTest(mode=module.__name__):
                seen = []

                def cost(contexts):
                    seen.append(list(contexts))
                    return 10.0 + len(contexts)

                sim = make_simulator(module, max_attention_batch_size=2,
                                     initial_context_length=10, attention_cost_model=cost)
                tokens = [module.Token(i, 0, i, i, home_gpu=0 if i < 3 else 1)
                          for i in range(4)]
                self.assertEqual(sim._attention_time(tokens), 23.0)
                self.assertEqual(seen, [[10, 11], [12], [13]])

    def test_local_transfers_cost_zero_and_remote_counts_include_copies(self):
        for module in (sim_sync, sim_tbo):
            observed = []

            def delay(src, dst, count):
                observed.append((src, dst, count))
                return float(count)

            sim = make_simulator(module, net_delay_fn=delay)
            token = module.Token(0, 0, 0, 0, home_gpu=1)
            if module is sim_sync:
                _, _, counts = sim._route_layer(0, [token])
            else:
                sim._compute_layer_times(0, [token], {0: 0})
                self.assertIn((1, 0, 2), observed)
                counts = {0: {1: 2}, 1: {1: 7}}
            observed.clear()
            self.assertEqual(sim._communication_transfers(counts, reverse=False), [(1, 0, 2.0)])
            self.assertEqual(observed, [(1, 0, 2)])
            self.assertEqual(sim._communication_transfers({1: {1: 8}}, reverse=True), [])

    def test_fixed_delay_mode_has_no_receive_contention(self):
        for module in (sim_sync, sim_tbo):
            sim = make_simulator(module)
            transfers = sim._communication_transfers({0: {1: 2, 2: 4}}, reverse=False)
            self.assertEqual(sim._communication_duration(transfers), 0.1)
            self.assertEqual(sim._communication_duration([]), 0.0)

    def test_tokens_carry_actual_previous_completion_for_itl(self):
        for module in (sim_sync, sim_tbo):
            observed = []
            sim = make_simulator(
                module, request_arrivals=[0.0, 0.0], tokens_per_request=2,
                net_delay_fn=lambda src, dst, count: 8.0,
                per_token_begin_cb=lambda token: setattr(token, 'sampled_for_stats', True),
                per_token_stats_cb=lambda token, when: observed.append((token, when)),
            )
            sim.run()
            last = {}
            gaps = []
            for token, when in observed:
                self.assertEqual(token.previous_completion_time, last.get(token.request_id))
                if token.previous_completion_time is not None:
                    gaps.append(token.birth_time - token.previous_completion_time)
                    self.assertGreater(when - token.previous_completion_time, 0)
                last[token.request_id] = when
            self.assertEqual(len(observed), 4)
            for rid, when in last.items():
                self.assertEqual(sim.request_lookup[rid].last_completion_time, when)
            if module is sim_tbo:
                self.assertTrue(any(gap > 0 for gap in gaps))

    def test_previous_completion_is_updated_without_stats_callback(self):
        for module in (sim_sync, sim_tbo):
            sim = make_simulator(module, tokens_per_request=2)
            result = sim.run()
            self.assertEqual(sim.request_lookup[0].last_completion_time,
                             max(result['completion_times'].values()))

    def test_finalize_cost_parallel_across_home_gpus(self):
        for module in (sim_sync, sim_tbo):
            sim = make_simulator(module, finalize_per_token_t=0.5)
            tokens = [module.Token(i, 0, i, 0, home_gpu=(i % 2)) for i in range(5)]
            self.assertEqual(sim._finalize_time(tokens), 1.5)


class TBOReceiveQueueTests(unittest.TestCase):
    def schedule(self, tasks0, tasks1, **kwargs):
        return sim_tbo.TBOMoESimulator._schedule_resources(
            tasks0, tasks1, step_idx=0, iteration_base_time=0,
            timeline_capture=None, **kwargs)

    def test_final_combine_must_complete_before_token_completes(self):
        finish = self.schedule([('compute', 2), ('combine', 7)], [])
        self.assertEqual(finish, (9.0, 0.0))

    def test_fixed_delay_microbatches_overlap_without_receive_resources(self):
        task = [('dispatch', 5, 'dispatch', 0, None)]
        self.assertEqual(self.schedule(task, task), (5.0, 5.0))

    def test_two_microbatches_share_destination_receive_queue(self):
        a = [('dispatch', 5, 'dispatch', 0, [(1, 0, 5)])]
        b = [('dispatch', 5, 'dispatch', 0, [(2, 0, 5)])]
        self.assertEqual(self.schedule(a, b), (5.0, 10.0))

    def test_dispatch_and_combine_share_same_receive_queue(self):
        a = [('dispatch', 5, 'dispatch', 0, [(1, 0, 5)]),
             ('compute', 1), ('combine', 2, 'combine', 0, [(1, 0, 2)])]
        b = [('dispatch', 5, 'dispatch', 0, [(2, 0, 5)]),
             ('compute', 1), ('combine', 2, 'combine', 0, [(2, 0, 2)])]
        self.assertEqual(self.schedule(a, b), (12.0, 14.0))

    def test_independent_receive_channels_can_overlap(self):
        a = [('dispatch', 5, 'dispatch', 0, [(1, 0, 5)])]
        b = [('dispatch', 5, 'dispatch', 0, [(2, 0, 5)])]
        self.assertEqual(self.schedule(a, b, n_gpu_per_host=2), (5.0, 5.0))
        c = [('dispatch', 5, 'dispatch', 0, [(0, 1, 5)])]
        self.assertEqual(self.schedule(a, c), (5.0, 5.0))


class BaselineSmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.trace = root / 'trace.parquet'
        self.expert = root / 'expert.csv'
        rows = [dict(rid=10, token_index=tok, layer=layer,
                     expert_logical_k0=0, expert_logical_k1=1)
                for layer in range(2) for tok in (5, 9)]
        pq.write_table(pa.Table.from_pylist(rows), self.trace)
        self.expert.write_text('batch_size,time_ms\n1,0.3\n2,0.4\n3,0.5\n4,0.6\n')

    def run_mode(self, module, **overrides):
        kwargs = dict(ep_group_size=2, total_expert_count=4, max_batch_size=4,
                      num_layers=2, total_requests=1, tokens_per_request=2,
                      attn_dp_group_size=2, routing_top_k=2,
                      expert_profile_path=str(self.expert),
                      profile_routing_path=str(self.trace),
                      global_request_max_batch_size=4)
        kwargs.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return module.run_simulation(**kwargs)

    def test_single_microbatch_matches_sync_and_all_tokens_complete(self):
        sync = self.run_mode(sim_sync)
        tbo = self.run_mode(sim_tbo)
        self.assertEqual(len(sync['completion_times']), 2)
        self.assertEqual(sync['completion_times'], tbo['completion_times'])
        self.assertEqual(sync['makespan'], 32.0)
        self.assertFalse(sync['stopped_early'])
        self.assertFalse(tbo['stopped_early'])

    def test_per_expert_and_group_batch_metrics_have_distinct_denominators(self):
        for module in (sim_sync, sim_tbo):
            result = self.run_mode(module)
            self.assertEqual(result['avg_per_expert_batch_size'], 1.0)
            self.assertEqual(result['avg_expert_group_batch_size'], 2.0)
            self.assertEqual(result['completed_tokens'], 2)
            self.assertEqual(result['completed_requests'], 1)
            seconds = result['makespan'] / 10 / 1000
            self.assertEqual(result['output_tokens_per_sec'], 2 / seconds)
            self.assertEqual(result['completed_requests_per_sec'], 1 / seconds)

    def test_early_stop_reports_actual_completed_requests(self):
        for module in (sim_sync, sim_tbo):
            result = self.run_mode(module, total_requests=4, tokens_per_request=3,
                                   arrival_rate=0, tokens_after_reaching_max_bs=1)
            self.assertTrue(result['stopped_early'])
            self.assertGreater(result['completed_tokens'], 0)
            self.assertEqual(result['completed_requests'], 0)
            self.assertEqual(result['completed_requests_per_sec'], 0)
            self.assertGreater(result['avg_throughput_req_per_sec'], 0)

    def test_nonzero_finalize_cost_is_applied_in_both_modes(self):
        for module in (sim_sync, sim_tbo):
            result = self.run_mode(module, finalize_per_token_t=0.5)
            self.assertEqual(result['makespan'], 33.0)

    def test_explicit_grouped_scale_is_wired_into_both_entrypoints(self):
        grouped = Path(self.tmp.name) / 'grouped.csv'
        with grouped.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['expert_counts', 'time_ms'])
            writer.writerow(['[1, 1]', 0.2])
        for module in (sim_sync, sim_tbo):
            result = self.run_mode(module, grouped_profile_path=str(grouped), expert_cost_scale=2.0)
            self.assertEqual(result['makespan'], 24.0)

    def test_attention_profile_uses_growing_context_in_both_entrypoints(self):
        attention = Path(self.tmp.name) / 'attention.csv'
        attention.write_text('batch_size,context_length,time_ms\n1,10,0.05\n1,11,0.1\n')
        for module in (sim_sync, sim_tbo):
            result = self.run_mode(module, attention_profile_path=str(attention),
                                   initial_context_length=10, max_attention_batch_size=1)
            self.assertEqual(result['makespan'], 27.0)

    def test_rejects_layer_count_outside_trace(self):
        for module in (sim_sync, sim_tbo):
            with self.assertRaisesRegex(ValueError, 'routing trace range'):
                self.run_mode(module, num_layers=3)

    def test_two_microbatches_finish_remote_return_traffic(self):
        for module in (sim_sync, sim_tbo):
            result = self.run_mode(module, total_requests=4, tokens_per_request=2,
                                   net_delay_fn=lambda src, dst, count: count * 0.5)
            self.assertEqual(len(result['completion_times']), 8)
            self.assertTrue(all(value > 0 for value in result['completion_times'].values()))
            self.assertFalse(result['stopped_early'])


if __name__ == '__main__':
    unittest.main()
