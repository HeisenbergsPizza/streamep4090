import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from disagmoe.utils.shared_top8_replay import SharedTop8Replay


class StreamInferRouteTest(unittest.TestCase):
    def test_mixed_batch_uses_request_and_current_decode_position(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "trace.npy"
            table = np.arange(6 * 4 * 2 * 8, dtype=np.uint8).reshape(6, 4, 2, 8)
            np.save(path, table)
            replay = SharedTop8Replay(path, 64, audit_dir=root)
            try:
                # Formal IDs start at 11; each length includes the current KV slot.
                rids, seq_lens = [13, 11, 1, 12], [68, 65, 101, 66]
                indices = replay.row_indices(rids, seq_lens)
                self.assertEqual(indices, [2 * 4 + 3, 0, 0, 1 * 4 + 1])
                dispatched = table[:, :, 1].reshape(-1, 8)[indices]
                replay.audit_dispatch(1, rids, seq_lens, dispatched.ravel().tolist())
                events = [json.loads(line) for line in Path(replay.audit_file.name).read_text().splitlines()]
                self.assertEqual([(e["request"], e["step"]) for e in events], [(2, 3), (0, 0), (1, 1)])
                for event in events:
                    self.assertEqual(event["experts"], table[event["request"], event["step"], 1].tolist())
                with self.assertRaises(ValueError):
                    replay.row_indices([11], [64])
                with self.assertRaises(ValueError):
                    replay.row_indices([17], [65])
            finally:
                replay.audit_file.close()
