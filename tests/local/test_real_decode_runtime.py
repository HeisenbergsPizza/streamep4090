"""Check the snapshot-to-paged-cache handoff and reordered expert returns on CPU."""

import unittest

import torch

from disagmoe.frontend.real_decode import DecodeResiduals, restore_prompt_kv


class RealDecodeRuntimeTest(unittest.TestCase):
    def test_prompt_kv_uses_allocated_pages_with_writable_next_slot(self):
        source = torch.arange(2 * 2 * 5 * 2, dtype=torch.float32).to(torch.bfloat16)
        source = source.view(2, 2, 5, 2, 1)
        pages = [torch.full((2, 5, 2, 2, 1), -1., dtype=torch.bfloat16) for _ in range(2)]
        restore_prompt_kv(source, pages, [4, 1, 3])
        for i, buf in enumerate(pages):
            torch.testing.assert_close(buf[:, 4], source[i, :, :2])
            torch.testing.assert_close(buf[:, 1], source[i, :, 2:4])
            torch.testing.assert_close(buf[:, 3, :1], source[i, :, 4:])
            self.assertTrue((buf[:, 3, 1] == -1).all())
            self.assertTrue((buf[:, [0, 2]] == -1).all())
        with self.assertRaisesRegex(ValueError, "physical block"):
            restore_prompt_kv(source, pages, [4, 1, 4])

    def test_residuals_follow_requests_and_reject_stale_returns(self):
        state = DecodeResiduals([101, 13, 9], 1, "cpu")

        def values(*nums):
            return torch.tensor(nums, dtype=torch.bfloat16).view(-1, 1)

        def save(layer, reqs, data):
            state.save(layer, reqs, state.indices(reqs), data)

        def combine(layer, reqs, data):
            return state.add_after_experts(layer, reqs, state.indices(reqs), data)

        save(0, [13, 9, 101], values(10, 20, 30))
        with self.assertRaisesRegex(RuntimeError, "pending residual"):
            save(0, [101], values(90))
        with self.assertRaisesRegex(RuntimeError, "Missing residual"):
            combine(1, [13], values(0))
        # One rank runs request 9's next layer while 101 and 13 await their experts.
        early = combine(0, [9], values(7))
        save(1, [9], early)
        torch.testing.assert_close(combine(0, [101, 13], values(3, 4)), values(33, 14))
        torch.testing.assert_close(combine(1, [9], values(1)), values(28))
        with self.assertRaisesRegex(RuntimeError, "Missing residual"):
            combine(1, [9], values(0))
        # The same request can start its next decode step after the last layer.
        save(0, [9], values(12))
        torch.testing.assert_close(combine(0, [9], values(2)), values(14))
