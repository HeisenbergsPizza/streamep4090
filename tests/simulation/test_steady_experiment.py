"""Workload invariants for controlled open-arrival experiments."""
import unittest
import csv
import hashlib
import json
import gzip
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from simulation.steady_experiment import (poisson_workload, measurement_windows,
                                          build_parser, load_runtime_reference, resolve_execution)


class SteadyWorkloadTests(unittest.TestCase):
    def test_defaults_follow_reference_changes_without_executing_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'config.sh'
            sentinel = Path(directory) / 'must_not_exist'
            for cap in (512, 1024):
                source.write_text(f'MAX_BATCH_SIZE_ATTN=256\nMAX_BATCH_SIZE_EXP={cap}\n'
                                  'DEFRAG_LOOKAHEAD_STEPS=6\nDEFRAG_LOOKBACK_STEPS=2\n'
                                  f'DEFRAG_WEIGHT_DECAY=0.7\ntouch {sentinel}\n')
                with patch('simulation.steady_experiment.RUNTIME_REFERENCE_PATH', source):
                    args = build_parser().parse_args([
                        '--routing-profile', 'route', '--expert-profile', 'cost',
                        '--output-dir', 'output', '--policy', 'defrag', '--rate', '1'])
                self.assertEqual((args.attention_batch_cap, args.expert_batch_cap), (256, cap))
                self.assertEqual((args.lookahead_steps, args.lookback_steps, args.weight_decay), (6, 2, .7))
                self.assertEqual(args.expert_scheduling, 'layer')
                self.assertEqual(args.mode, 'async')
                self.assertIsNone(args.attention_profile)
                self.assertFalse(sentinel.exists())

    def test_execution_mode_policy_contract(self):
        base = ['--routing-profile', 'route', '--expert-profile', 'cost',
                '--output-dir', 'output', '--rate', '1']
        for mode in ('sync', 'tbo'):
            with self.subTest(mode=mode):
                args = build_parser().parse_args(base + ['--mode', mode])
                self.assertEqual(resolve_execution(args), mode)
                self.assertEqual(args.policy, mode)
                for extra in (['--policy', 'defrag'], ['--expert-scheduling', 'expert']):
                    args = build_parser().parse_args(base + ['--mode', mode] + extra)
                    with self.assertRaises(ValueError):
                        resolve_execution(args)
        for extra in ([], ['--policy', 'sync'], ['--policy', 'tbo']):
            args = build_parser().parse_args(base + extra)
            with self.assertRaises(ValueError):
                resolve_execution(args)

    def test_missing_or_nonliteral_runtime_reference_never_falls_back(self):
        valid = ('MAX_BATCH_SIZE_ATTN=256\nMAX_BATCH_SIZE_EXP=1024\n'
                 'DEFRAG_LOOKAHEAD_STEPS=4\nDEFRAG_LOOKBACK_STEPS=4\nDEFRAG_WEIGHT_DECAY=0.8\n')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'config.sh'
            with patch('simulation.steady_experiment.RUNTIME_REFERENCE_PATH', source):
                with self.assertRaises(FileNotFoundError):
                    load_runtime_reference()
                for invalid in (valid.replace('MAX_BATCH_SIZE_EXP=1024\n', ''),
                                valid + 'MAX_BATCH_SIZE_EXP=512\n',
                                valid.replace('MAX_BATCH_SIZE_EXP=1024', 'MAX_BATCH_SIZE_EXP=${CAP:-1024}'),
                                valid.replace('DEFRAG_WEIGHT_DECAY=0.8', 'DEFRAG_WEIGHT_DECAY=nan')):
                    source.write_text(invalid)
                    with self.assertRaises(ValueError):
                        load_runtime_reference()

    def test_poisson_positive_sorted_bounded_and_deterministic(self):
        first = poisson_workload(50, 20000, 256, 8)
        self.assertEqual(first, poisson_workload(50, 20000, 256, 8))
        arrivals, indices = first
        self.assertTrue(0 < arrivals[0] < arrivals[-1] < 20000)
        self.assertTrue(all(a < b for a, b in zip(arrivals, arrivals[1:])))
        self.assertTrue(all(0 <= index < 256 for index in indices))
        self.assertEqual(len(arrivals), len(indices))

    def test_common_random_numbers_couple_rates_without_reordering_requests(self):
        slow, slow_ids = poisson_workload(20, 20000, 32, 4)
        fast, fast_ids = poisson_workload(40, 20000, 32, 4)
        self.assertEqual(slow_ids, fast_ids[:len(slow_ids)])
        for a, b in zip(slow, fast):
            self.assertAlmostEqual(a, b * 2)

    def test_bootstrap_selects_complete_request_ordinals_not_individual_tokens(self):
        arrivals, indices = poisson_workload(100, 10000, 2, 6)
        self.assertEqual(set(indices), {0, 1})
        self.assertGreater(len(indices), 2)
        self.assertEqual(len(arrivals), len(indices))

    def test_invalid_arrival_parameters(self):
        for values in ((0, 10, 1), (-1, 10, 1), (1, 0, 1), (1, 10, 0),
                       (float('inf'), 10, 1), (1, float('nan'), 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                poisson_workload(*values, seed=0)

    def test_windows_exclude_warmup_and_cover_whole_measurement(self):
        windows = measurement_windows(10000, 40000, 5000)
        self.assertEqual(windows[0], ('measurement', 10000, 40000))
        self.assertEqual(len(windows), 7)
        self.assertEqual(windows[1][1], 10000)
        self.assertEqual(windows[-1][2], 40000)
        for a, b in zip(windows[1:], windows[2:]):
            self.assertEqual(a[2], b[1])
        with self.assertRaises(ValueError):
            measurement_windows(5000, 21000, 5000)


class SteadyRunnerIntegrationTests(unittest.TestCase):
    def test_paired_policy_runs_preserve_offered_workload_and_finish(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route, cost = root / 'routing.parquet', root / 'cost.csv'
            pq.write_table(pa.Table.from_pylist([
                dict(rid=r, token_index=t, layer=layer,
                     expert_logical_k0=r % 2, expert_logical_k1=2 + t % 2)
                for r in range(3) for t in range(2) for layer in range(2)]), route)
            cost.write_text('batch_size,time_ms\n1,0.01\n2,0.02\n3,0.03\n4,0.04\n')
            attention = root / 'attention.csv'
            with attention.open('w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['component', 'batch_size', 'context_length', 'time_ms'])
                for batch in (1, 4):
                    for context in (1024, 1026):
                        writer.writerow(['qkv', batch, context, .01 * batch])
                        writer.writerow(['attention_core', batch, context,
                                         batch * (.02 + .005 * (context - 1024))])
            base = [sys.executable, '-m', 'simulation.steady_experiment',
                    '--routing-profile', str(route), '--expert-profile', str(cost),
                    '--attention-profile', str(attention),
                    '--rate', '5000', '--duration-s', '0.1', '--warmup-s', '0.02', '--block-s', '0.02',
                    '--sample-interval-ms', '5', '--trace-duration-ms', '20',
                    '--gpus', '2', '--experts', '4', '--top-k', '2', '--layers', '2',
                    '--output-tokens', '2', '--active-request-cap', '3',
                    '--attention-batch-cap', '4', '--expert-batch-cap', '4', '--seed', '22']
            cases = (('defrag', 'defrag', 'layer', 'async'), ('fifo', 'fifo', 'layer', 'async'),
                     ('defrag_expert', 'defrag', 'expert', 'async'),
                     ('sync', 'sync', 'layer', 'sync'), ('tbo', 'tbo', 'layer', 'tbo'))
            configurations = {}
            for label, policy, granularity, mode in cases:
                command = base + ['--mode', mode, '--policy', policy, '--expert-scheduling', granularity,
                                  '--output-dir', str(root / label)]
                done = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(done.returncode, 0, done.stderr)
                result = json.loads((root / label / 'result.json').read_text())
                configurations[label] = result['configuration']['comparison_configuration']
                self.assertEqual(result['configuration']['execution_mode'], mode)
                self.assertEqual(configurations[label]['execution_mode'], mode)
                self.assertIn('sim_collective.py', result['configuration']['code_identity'])
                self.assertEqual(configurations[label]['attention_profile'], str(attention.resolve()))
                self.assertEqual(configurations[label]['attention_identity']['sha256'],
                                 hashlib.sha256(attention.read_bytes()).hexdigest())
                self.assertEqual(result['simulation']['attention_cost_model']['kind'],
                                 'decode_batch_context_profile')
                launches = [entry for entry in result['simulation']['schedule_trace']
                            if entry['stage'] == 'attention']
                self.assertTrue(launches)
                for entry in launches:
                    batch, context = entry['batch_size'], entry['attention_context_mean']
                    expected = batch * (.03 + .005 * (context - 1024)) * 10
                    self.assertAlmostEqual(entry['end'] - entry['start'], expected)
                    self.assertAlmostEqual(sum(entry['attention_components_ticks'].values()), expected)
                    self.assertGreater(expected, batch * .02 * 10)  # The fallback was not used.
                self.assertTrue(result['validation']['fully_drained'])
                reference = result['configuration']['runtime_reference']
                self.assertFalse(reference['complete_runtime_equivalence_verified'])
                self.assertEqual(reference['source'], load_runtime_reference()['source'])
                self.assertEqual(reference['overrides']['expert_batch_cap']['actual'], 4)
                self.assertEqual(reference['overrides']['attention_batch_cap']['actual'], 4)
                self.assertEqual(result['configuration']['comparison_configuration']['runtime_reference_identity'],
                                 reference['source'])
                self.assertEqual(result['configuration']['expert_scheduling'], granularity)
                self.assertEqual(result['simulation']['expert_scheduling'], granularity)
                self.assertEqual(configurations[label]['expert_scheduling'], granularity)
                self.assertEqual(reference['scheduling']['simulation_only_extension'],
                                 granularity == 'expert' or mode != 'async')
                self.assertEqual(reference['scheduling']['runtime_expert_scheduling'], 'layer')
                if mode != 'async':
                    self.assertEqual(reference['scheduling']['microbatches'], 1 if mode == 'sync' else 2)
                    self.assertEqual(reference['scheduling']['admission'], 'iteration_boundary')
                    self.assertFalse(reference['scheduling']['runtime_execution_equivalence_verified'])
                self.assertIn('steady_metrics', result['simulation'])
                self.assertEqual(result['simulation']['completed_tokens'],
                                 2 * result['configuration']['total_requests'])
                with (root / label / 'tokens.csv').open() as file:
                    tokens = list(csv.DictReader(file))
                self.assertEqual(len(tokens), result['simulation']['completed_tokens'])
                # Baselines must keep output timestamps and ITL semantics, including
                # any interval during which an iteration/microbatch gate delays work.
                previous = {}
                last_completion = -1.0
                for token in tokens:
                    rid, index = int(token['request_id']), int(token['token_index'])
                    completion = float(token['completion_ms'])
                    self.assertGreaterEqual(completion, last_completion)
                    last_completion = completion
                    if index:
                        self.assertAlmostEqual(float(token['itl_ms']), completion - previous[rid])
                    else:
                        self.assertEqual(token['itl_ms'], '')
                    previous[rid] = completion
                steady = result['simulation']['steady_metrics']
                self.assertTrue(steady['fully_drained'])
                for window in steady['windows'].values():
                    selected = [r for r in tokens if window['start_ms'] <= float(r['completion_ms']) < window['end_ms']]
                    self.assertEqual(window['counts']['completed_tokens'], len(selected))
                    itls = [float(r['itl_ms']) for r in selected if r['itl_ms']]
                    self.assertEqual(window['metrics_ms']['itl']['count'], len(itls))
                    self.assertAlmostEqual(window['metrics_ms']['itl']['sum'], sum(itls))
                    for gpu, busy in enumerate(window['gpu_busy_ms']):
                        self.assertAlmostEqual(sum(window['gpu_stage_busy_ms'][gpu].values()), busy)
                        self.assertLessEqual(window['gpu_busy_fraction'][gpu], 1 + 1e-10)
                    for values in window['receive_busy_fraction'].values():
                        self.assertTrue(all(0 <= value <= 1 + 1e-10 for value in values))
                from simulation.plot_scheduler_trace import load_trace
                trace = load_trace(root / label / 'scheduler_queue_trace.npz')
                self.assertEqual(trace.base_policy, policy)
                self.assertTrue(trace.metadata['complete'])
                self.assertEqual(trace.metadata['requested_window_ms'], {'start': 80.0, 'end': 100.0})
                with gzip.open(root / label / 'scheduler_queue_trace.decisions.jsonl.gz', 'rt') as file:
                    decisions = [json.loads(line) for line in file]
                self.assertTrue(decisions)
                self.assertEqual(len(decisions), trace.metadata['decision_count'])
                prior_end = {}
                for decision in decisions:
                    self.assertGreaterEqual(decision['start_ms'] + 1e-8,
                                            prior_end.get(decision['gpu'], 0))
                    prior_end[decision['gpu']] = decision['end_ms']
                again = subprocess.run(command, text=True, capture_output=True)
                self.assertNotEqual(again.returncode, 0)
                self.assertIn('absent or empty', again.stderr)
            self.assertEqual((root/'defrag/workload.csv').read_bytes(), (root/'fifo/workload.csv').read_bytes())
            self.assertEqual((root/'defrag/workload.csv').read_bytes(),
                             (root/'defrag_expert/workload.csv').read_bytes())
            self.assertEqual(configurations['defrag'], configurations['fifo'])
            group, expert = dict(configurations['defrag']), dict(configurations['defrag_expert'])
            self.assertNotEqual(group.pop('expert_scheduling'), expert.pop('expert_scheduling'))
            self.assertEqual(group, expert)
            for mode in ('sync', 'tbo'):
                self.assertEqual((root / 'defrag/workload.csv').read_bytes(),
                                 (root / mode / 'workload.csv').read_bytes())
                async_common, baseline_common = dict(configurations['defrag']), dict(configurations[mode])
                self.assertEqual(async_common.pop('execution_mode'), 'async')
                self.assertEqual(baseline_common.pop('execution_mode'), mode)
                self.assertEqual(async_common, baseline_common)

            invalid_output = root / 'invalid-expert-baseline'
            done = subprocess.run(base + ['--mode', 'sync', '--expert-scheduling', 'expert',
                                         '--output-dir', str(invalid_output)], text=True, capture_output=True)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn('require --expert-scheduling layer', done.stderr)
            self.assertFalse(invalid_output.exists())


if __name__ == '__main__':
    unittest.main()
