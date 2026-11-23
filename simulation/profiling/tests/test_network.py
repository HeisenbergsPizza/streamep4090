"""CPU checks of case construction, measured-transport evidence, and network fit units."""
import argparse
from pathlib import Path
import tempfile
import unittest

from simulation.profiling import network


class NetworkProfilingTests(unittest.TestCase):
    def test_each_ordered_pair_and_disjoint_assignments(self):
        for n in (2, 3, 4, 8):
            cases = list(network.transfer_cases(n))
            isolated = [pairs[0] for scenario, pairs in cases if scenario == "isolated"]
            self.assertCountEqual(isolated, [(i, j) for i in range(n) for j in range(n) if i != j])
            self.assertTrue(any(s == "bidirectional" for s, _ in cases))
            for scenario, pairs in cases:
                flat = [r for pair in pairs for r in pair]
                self.assertEqual(len(set(flat)), len(flat))

    def test_message_sizes(self):
        self.assertEqual(network.message_sizes(None, 2048)[0], 4096)
        self.assertEqual(network.message_sizes("12288,2,1024", 4), [2, 1024, 12288])
        for bad in ("10,12", "0,10,12", "9,10,12", "4,4,8"):
            with self.assertRaises(ValueError):
                network.message_sizes(bad, 4)

    def _rows(self, us):
        return [dict(src=src, dst=1-src, bytes=x, endpoint_ms=us(src, x)/1000,
                     scenario="isolated", role="receive")
                for x in (1024, 32768, 1048576) for src in (0, 1)]

    def test_decimal_gigabits_and_no_concurrent_or_sender_contamination(self):
        rows = self._rows(lambda src, x: 4 + x * 0.008 / 120)
        rows += [dict(rows[0], endpoint_ms=1e5, role="send"),
                 dict(rows[0], endpoint_ms=1e5, scenario="disjoint")]
        params, details = network.fit_network(rows, 0.05)
        self.assertAlmostEqual(params["--network-bandwidth-gbps"], 120)
        self.assertAlmostEqual(params["--network-latency-us"], 4)
        self.assertLess(details["max_link_relative_error"], 1e-8)

    def test_mixed_links_preserve_measurements_without_exporting_misleading_fit(self):
        rows = self._rows(lambda src, x: 4 + x * (0.00001 if src else 0.001))
        params, details = network.fit_network(rows, 0.35)
        self.assertEqual(params, {})
        self.assertGreater(details["max_link_relative_error"], 0.35)
        self.assertEqual(len(details["link_medians"]), 6)

    def test_flat_timing_does_not_produce_bandwidth(self):
        rows = self._rows(lambda src, x: 12)
        params, _ = network.fit_network(rows, 0.1)
        self.assertFalse(params)

    def test_nccl_transport_evidence_requires_log_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(network.transport_evidence(tmp)["channels"])
            log = Path(tmp) / "nccl-rank-0.log"
            log.write_text("NCCL INFO P2P is available\n"
                           "NCCL INFO Channel 00/0 : 0[0] -> 1[1] via SHM/direct/direct\n")
            result = network.transport_evidence(tmp)
            self.assertEqual(result["status"], "observed")
            self.assertIn("SHM/direct/direct", result["channels"][log.name][0])

    def test_command(self):
        parser = argparse.ArgumentParser()
        network.add_commands(parser.add_subparsers())
        args = parser.parse_args(["network", "--bundle", "b", "--bytes", "2,4,6"])
        self.assertEqual(args.bundle, Path("b"))
        self.assertIs(args.func, network.run_network)


if __name__ == "__main__":
    unittest.main()
