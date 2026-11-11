"""CPU-only checks for exact-length prompts and profiler cleanup."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "experiments/scripts/sglang/profile_client_dpa4_ep4_b512.py"
spec = importlib.util.spec_from_file_location("sglang_profile_client", SCRIPT)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class FakeTokenizer:
    """Use real text characters plus distinct chat tokens, without any padding."""

    prefix = "<user>\n"
    suffix = "<end>\n<assistant>\n<think>\n\n</think>\n\n"
    special = {"<user>": 1, "<end>": 2, "<assistant>": 3, "<think>": 4, "</think>": 5}
    pad_token_id = 0

    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(kwargs)
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise AssertionError("Expected one user message")
        return self.prefix + messages[0]["content"] + self.suffix

    def encode(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("Prompt already contains chat-template special tokens")
        tokens = []
        while text:
            for special, token in self.special.items():
                if text.startswith(special):
                    tokens.append(token)
                    text = text[len(special):]
                    break
            else:
                tokens.append(ord(text[0]) + 100)
                text = text[1:]
        return tokens

    def decode(self, tokens):
        specials = {value: key for key, value in self.special.items()}
        return "".join(specials[token] if token in specials else chr(token - 100) for token in tokens)


def results(*, prompt_tokens=64, completion_tokens=128):
    return [
        {"text": "generated", "meta_info": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": 0,
        }}
        for _ in range(512)
    ]


def server_info():
    return {
        "tp_size": 4, "dp_size": 4, "ep_size": 4, "moe_dp_size": 1,
        "attn_cp_size": 1, "enable_dp_attention": True,
        "enable_dp_lm_head": False, "max_running_requests": 512,
        "cuda_graph_max_bs": 128, "disable_cuda_graph": False,
        "moe_a2a_backend": "none", "moe_runner_backend": "triton",
        "attention_backend": "triton", "dtype": "bfloat16",
        "load_balance_method": "round_robin",
        "internal_states": [
            {"effective_max_running_requests_per_dp": 128} for _ in range(4)
        ],
    }


class FakeResponse:
    def __init__(self, value=None, text="ok"):
        self.value = value
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return self.value


class FakeSession:
    def __init__(self, *, warmup=None, measurement=None):
        self.trust_env = True
        self.events = []
        self.generations = []
        self.warmup = results() if warmup is None else warmup
        self.measurement = results() if measurement is None else measurement

    def get(self, url, *, timeout):
        path = url.removeprefix(client.BASE_URL)
        self.events.append(("GET", path))
        if path == "/health":
            return FakeResponse()
        if path == "/server_info":
            return FakeResponse(server_info())
        raise AssertionError(f"Unexpected GET {path}")

    def post(self, url, *, json, timeout):
        path = url.removeprefix(client.BASE_URL)
        self.events.append(("POST", path))
        if path == "/generate":
            self.generations.append(json)
            index = len(self.generations)
            if index == 1:
                return FakeResponse({"text": "normal generation"})
            value = self.warmup if index == 2 else self.measurement
            if isinstance(value, Exception):
                raise value
            return FakeResponse(value)
        if path == "/start_profile":
            if json != {"activities": ["CUDA_PROFILER"]}:
                raise AssertionError("Unexpected profiler activity")
            return FakeResponse(text="started")
        if path == "/stop_profile":
            if json != {}:
                raise AssertionError("Unexpected stop payload")
            return FakeResponse(text="stopped")
        raise AssertionError(f"Unexpected POST {path}")


class SGLangProfileClientTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sglang-client-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.tokenizer = FakeTokenizer()

    def main_quietly(self, session, *args):
        requests = types.ModuleType("requests")
        requests.Session = mock.Mock(return_value=session)
        transformers = types.ModuleType("transformers")
        loader = mock.Mock(return_value=self.tokenizer)
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=loader)
        env = {
            "SGLANG_WORK": str(self.directory), "MODEL_DIR": "/fake/local-model",
            "SGLANG_DISABLE_CUDA_GRAPH": "0",
        }
        with mock.patch.dict(sys.modules, {"requests": requests, "transformers": transformers}), \
                mock.patch.dict(os.environ, env), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            client.main(list(args))
        loader.assert_called_once_with("/fake/local-model", local_files_only=True)

    def saved_results(self):
        paths = [path for path in (self.directory / "reports").glob("*.json")
                 if not path.name.endswith(".server-info.json")]
        return [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    def test_fixed_64_token_batch_preserves_identity_chat_boundaries_and_generation_suffix(self):
        measurement = client.batch_payload(self.tokenizer, "测量", 64)
        warmup = client.batch_payload(self.tokenizer, "预热", 64)
        self.assertNotIn("text", measurement)
        self.assertEqual(len(measurement["input_ids"]), 512)
        self.assertEqual(len(warmup["input_ids"]), 512)
        self.assertEqual(len({tuple(ids) for ids in measurement["input_ids"]}), 512)
        for i, (measured_ids, warmup_ids) in enumerate(zip(measurement["input_ids"], warmup["input_ids"])):
            for phase, ids in (("测量", measured_ids), ("预热", warmup_ids)):
                self.assertEqual(len(ids), 64)
                text = self.tokenizer.decode(ids)
                self.assertTrue(text.startswith(self.tokenizer.prefix + f"{phase}任务，编号 {i}。"))
                self.assertTrue(text.endswith(self.tokenizer.suffix))
                self.assertNotIn(self.tokenizer.pad_token_id, ids)
            self.assertNotEqual(measured_ids, warmup_ids)
        self.assertTrue(all(call == {
            "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
        } for call in self.tokenizer.template_calls))

    def test_long_body_is_truncated_and_short_body_is_extended_with_ordinary_text(self):
        for body in ("原" * 100, "短"):
            with self.subTest(body_length=len(body)), mock.patch.object(client, "PROMPT_BODY", body):
                ids = client.fixed_prompt_ids(self.tokenizer, 511, "测量", 64)
                self.assertEqual(len(ids), 64)
                text = self.tokenizer.decode(ids)
                start = self.tokenizer.prefix + "测量任务，编号 511。"
                self.assertTrue(text.startswith(start))
                self.assertTrue(text.endswith(self.tokenizer.suffix))
                resized_body = text[len(start):-len(self.tokenizer.suffix)]
                if len(body) == 100:
                    self.assertLess(len(resized_body), len(body))
                    self.assertEqual(set(resized_body), {"原"})
                else:
                    self.assertTrue(resized_body.startswith("短 请详细说明。"))
                    self.assertGreater(len(resized_body), len(body))
                self.assertNotIn(self.tokenizer.pad_token_id, ids)

    def test_default_payload_keeps_original_text_and_sampling_behavior(self):
        self.assertIsNone(client.parse_args([]).input_tokens)
        payload = client.batch_payload(self.tokenizer, "测量")
        self.assertNotIn("input_ids", payload)
        self.assertEqual(len(payload["text"]), 512)
        for i, text in enumerate(payload["text"]):
            self.assertEqual(text, self.tokenizer.prefix + f"测量任务，编号 {i}。"
                             + client.PROMPT_BODY + self.tokenizer.suffix)
        self.assertEqual(payload["sampling_params"], {
            "temperature": 0, "max_new_tokens": 128, "ignore_eos": True,
        })
        self.assertFalse(payload["stream"])
        fixed = client.batch_payload(self.tokenizer, "测量", 64)
        self.assertEqual(payload["sampling_params"], fixed["sampling_params"])
        self.assertEqual(payload["stream"], fixed["stream"])

    def test_too_short_input_fails_before_sending_generate_or_starting_profile(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            client.fixed_prompt_ids(self.tokenizer, 0, "测量", 1)
        for value in ("0", "-1"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                client.parse_args(["--input-tokens", value])
        session = FakeSession()
        with self.assertRaisesRegex(ValueError, "too short"):
            self.main_quietly(session, "--input-tokens", "1")
        self.assertEqual(session.generations, [])
        self.assertNotIn(("POST", "/start_profile"), session.events)

    def test_validate_batch_checks_all_results_and_reports_token_lengths(self):
        self.assertEqual(client.validate_batch(results(), input_tokens=64), {
            "prompt_tokens": {"min": 64, "max": 64},
            "completion_tokens": {"min": 128, "max": 128},
            "cached_tokens": {"reported_requests": 512, "min": 0, "max": 0},
        })
        for failure in ("prompt63", "missing_result", "completion127", "missing_meta"):
            with self.subTest(failure=failure):
                response = results()
                if failure == "prompt63":
                    response[-1]["meta_info"]["prompt_tokens"] = 63
                elif failure == "missing_result":
                    response.pop()
                elif failure == "completion127":
                    response[-1]["meta_info"]["completion_tokens"] = 127
                else:
                    del response[-1]["meta_info"]
                with self.assertRaises(RuntimeError):
                    client.validate_batch(response, input_tokens=64)

    def test_main_runs_fixed_length_warmup_and_profile_and_saves_inputs_and_summaries(self):
        session = FakeSession()
        self.main_quietly(session, "--input-tokens", "64")
        self.assertFalse(session.trust_env)
        self.assertEqual(session.events, [
            ("GET", "/health"), ("GET", "/server_info"),
            ("POST", "/generate"), ("POST", "/generate"),
            ("POST", "/start_profile"), ("POST", "/generate"),
            ("POST", "/stop_profile"),
        ])
        warmup, measured = session.generations[1:]
        for payload in (warmup, measured):
            self.assertEqual(len(payload["input_ids"]), 512)
            self.assertEqual({len(ids) for ids in payload["input_ids"]}, {64})
        self.assertNotEqual(warmup["input_ids"], measured["input_ids"])
        [saved] = self.saved_results()
        self.assertEqual(saved["input_tokens"], 64)
        self.assertEqual(saved["prompt_mode"], "fixed_input_ids")
        self.assertEqual(saved["request"], measured)
        self.assertEqual(saved["response"], session.measurement)
        self.assertEqual(saved["warmup_token_summary"], saved["token_summary"])
        self.assertEqual(saved["token_summary"]["prompt_tokens"], {"min": 64, "max": 64})
        self.assertEqual(saved["token_summary"]["completion_tokens"], {"min": 128, "max": 128})
        self.assertIsNone(saved["request_error"])
        self.assertIsNone(saved["profile_stop_error"])
        self.assertEqual(saved["profile_stop_response"], "stopped")
        self.assertGreaterEqual(saved["client_elapsed_seconds"], 0)

    def test_measurement_http_or_token_validation_failure_still_stops_and_saves(self):
        for failure in ("http", "prompt63"):
            with self.subTest(failure=failure):
                measurement = RuntimeError("simulated HTTP failure") if failure == "http" else results(prompt_tokens=63)
                session = FakeSession(measurement=measurement)
                with self.assertRaises(RuntimeError):
                    self.main_quietly(session, "--input-tokens", "64")
                self.assertEqual(session.events[-1], ("POST", "/stop_profile"))
                # Find by error type rather than depending on directory listing order.
                saved = next(item for item in self.saved_results()
                             if ("simulated HTTP failure" in item["request_error"]) == (failure == "http"))
                self.assertIsNone(saved["token_summary"])
                self.assertEqual(saved["profile_stop_response"], "stopped")
                self.assertEqual(saved["request"], session.generations[2])
                self.assertIsNone(saved["profile_stop_error"])
                if failure == "http":
                    self.assertIsNone(saved["response"])
                else:
                    self.assertIn("prompt_tokens=63", saved["request_error"])
                    self.assertEqual(saved["response"], measurement)

    def test_bad_warmup_fails_before_starting_profiler(self):
        session = FakeSession(warmup=results(completion_tokens=127))
        with self.assertRaisesRegex(RuntimeError, "预热"):
            self.main_quietly(session, "--input-tokens", "64")
        self.assertEqual(len(session.generations), 2)
        self.assertNotIn(("POST", "/start_profile"), session.events)
        self.assertNotIn(("POST", "/stop_profile"), session.events)
        self.assertEqual(self.saved_results(), [])


if __name__ == "__main__":
    unittest.main()
