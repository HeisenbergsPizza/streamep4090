import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.scripts.ops import shared_top8


class TestSharedTop8(unittest.TestCase):
    def test_trace_is_deterministic_and_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = (Path(temp) / name for name in ("a.npy", "b.npy"))
            meta = shared_top8.prepare(a, 4, 8, 6, 128, 23)
            self.assertEqual(shared_top8.prepare(b, 4, 8, 6, 128, 23)["sha256"], meta["sha256"])
            table = np.load(a)
            self.assertEqual(table.shape, (4, 8, 6, 8))
            self.assertTrue(np.all(np.diff(np.sort(table, axis=-1), axis=-1) > 0))
            self.assertLess(int(table.max()), 128)
            self.assertGreater(len(np.unique(table[:, 0, 0, 0])), 1)
            with self.assertRaises(FileExistsError):
                shared_top8.prepare(a, 4, 8, 6, 128, 24)

    def test_trace_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            t, a, b = (root / name for name in ("trace.npy", "streaminfer.jsonl", "sglang.jsonl"))
            shared_top8.prepare(t, 2, 4, 2, 8, 0)
            table = np.load(t)
            shape = (2, 2, 2)
            events = [
                dict(request=r, step=s, layer=l, experts=table[r, s, l].tolist())
                for r, s, l in np.ndindex(shape)
            ]
            def save(path, rows):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            save(a, events)
            save(b, [dict(phase="warmup"), *reversed(events)])
            report = shared_top8.verify(t, [a], [b], shape)
            self.assertEqual(report["sglang"]["matched_unique_routes"], 8)
            for rows, msg in ((events[1:], "first missing"),
                              ([*events, events[0]], "duplicate"),
                              ([dict(events[0], experts=list(reversed(events[0]["experts"])))], "trace specifies")):
                with self.subTest(msg=msg):
                    save(b, rows)
                    with self.assertRaisesRegex(ValueError, msg):
                        shared_top8.verify(t, [a], [b], shape)


if __name__ == "__main__":
    unittest.main()
