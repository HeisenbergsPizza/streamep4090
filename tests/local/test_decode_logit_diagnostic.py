from types import SimpleNamespace
import unittest

from experiments.scripts.rtx4090.diagnose_qwen_decode import failed_rows, replay_first_difference


class Snapshot:
    layers = 1

    def __len__(self):
        return 1

    def prompt_ids(self, index):
        return [0, 1]

    def first_token(self, index):
        return 2

    def expected_tokens(self, index):
        return [3, 4, 1]


def report():
    return {"decode_steps": 4, "requests": [{
        "snapshot_index": 0, "prefill_token": 2, "token_ids": [3, 5, 2, 1],
        "reference_prefix": [3, 4, 1], "first_different_step": 1,
    }]}


class TestDecodeLogitDiagnostic(unittest.TestCase):
    def test_report_must_match_snapshot(self):
        result = report()
        self.assertEqual(failed_rows(result, Snapshot(), 6), result["requests"])
        result["requests"][0]["reference_prefix"] = [2, 4, 1]
        with self.assertRaisesRegex(ValueError, "disagree"):
            failed_rows(result, Snapshot(), 6)
        result = report()
        result["requests"][0]["first_different_step"] = 0
        with self.assertRaisesRegex(ValueError, "first_different_step"):
            failed_rows(result, Snapshot(), 6)

    def test_saved_kv_shared_prefix_and_tied_argmax(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        snapshot = Snapshot()
        saved = torch.arange(8, dtype=torch.bfloat16).view(1, 2, 2, 2, 1)
        snapshot.load_kv = lambda index: saved
        calls = []

        class Model:
            def get_input_embeddings(self):
                return SimpleNamespace(weight=torch.zeros(1))

            def __call__(self, input_ids, past_key_values=None, **kwargs):
                calls.append(input_ids.flatten().tolist())
                logits = torch.zeros(1, 1, 6, dtype=torch.bfloat16)
                if past_key_values is None:
                    past_key_values = [(torch.zeros(1, 2, 2, 1, dtype=torch.bfloat16),
                                        torch.zeros(1, 2, 2, 1, dtype=torch.bfloat16))]
                    logits[0, 0, 2] = 1
                else:
                    for axis, tensor in enumerate(past_key_values[0]):
                        torch.testing.assert_close(tensor[0].transpose(0, 1), saved[0, axis])
                    if input_ids.item() == 2:
                        logits[0, 0, 3] = 1
                    else:
                        logits[0, 0, 4:] = 1
                return SimpleNamespace(logits=logits, past_key_values=past_key_values)

        result = replay_first_difference(Model(), snapshot, report()["requests"][0])
        self.assertEqual(calls, [[0, 1], [2], [3]])
        self.assertFalse(result["prefill_kv_bitwise_equal"])
        self.assertTrue(result["hf_reference_reproduced_through_step"])
        self.assertEqual(result["hf_greedy_token"], 4)
        self.assertEqual(result["hf_streaminfer_rank"], 1)
        self.assertEqual(result["hf_reference_minus_streaminfer_logit"], 0)
        self.assertEqual(result["hf_top1_minus_top2_logit"], 0)
