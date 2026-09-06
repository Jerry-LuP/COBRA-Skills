from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cobras.cobras_core.config import get_dataset
from cobras.cobras_core.llm import ProviderConfig
from cobras.task_envs.alfworld.eval import run_eval
from cobras.task_envs.alfworld.skill_data import build_observations, select_rows
from cobras.task_envs.cobras_adapter import load_cobras_adapter
from cobras.task_envs.eval_types import EvalConfig

try:
    from cobras.task_envs.alfworld.vendor import alfworld_envs
except ModuleNotFoundError:  # The ALFWorld runtime is installed in a separate environment.
    alfworld_envs = None


class ALFWorldTests(unittest.TestCase):
    @unittest.skipIf(alfworld_envs is None, "ALFWorld runtime is not installed in this Python environment")
    def test_worker_hides_third_party_initialization_noise(self) -> None:
        assert alfworld_envs is not None
        class FakeBatchEnv:
            def seed(self, _seed):
                print("seed noise")

        class FakeEnvironment:
            def __init__(self, *_args, **_kwargs):
                print("Initializing AlfredTWEnv...")

            def init_env(self, *, batch_size):
                self.assert_batch_size = batch_size
                print("Overall we have 1 games in split=eval_in_distribution")
                print("100%|progress bar|", file=__import__("sys").stderr)
                print("Evaluating with 1 games")
                return FakeBatchEnv()

        commands: queue.Queue = queue.Queue()
        results: queue.Queue = queue.Queue()
        commands.put(("close", None))
        stdout = io.StringIO()
        stderr = io.StringIO()
        config = {"env": {"type": "AlfredTWEnv"}, "dataset": {}}

        with patch.object(alfworld_envs, "get_environment", return_value=FakeEnvironment):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                alfworld_envs._worker_loop(
                    commands,
                    results,
                    config,
                    seed=42,
                    is_train=False,
                    eval_dataset="eval_in_distribution",
                    gamefile=None,
                )

        self.assertEqual(results.get_nowait(), (True, "ready"))
        self.assertEqual(results.get_nowait(), (True, None))
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_fixed_split_is_complete_disjoint_and_self_contained(self) -> None:
        cfg = get_dataset("alfworld")
        train = json.loads((cfg.data_root / "train" / "items.json").read_text(encoding="utf-8"))
        test = json.loads((cfg.data_root / "test" / "items.json").read_text(encoding="utf-8"))
        self.assertEqual(len(train), 50)
        self.assertEqual(len(test), 100)
        train_paths = {str(row["gamefile"]) for row in train}
        test_paths = {str(row["gamefile"]) for row in test}
        self.assertFalse(train_paths & test_paths)
        for gamefile in train_paths | test_paths:
            self.assertTrue((cfg.data_root / "runtime" / gamefile).is_file(), gamefile)
        self.assertTrue((cfg.data_root / "runtime" / "logic" / "alfred.pddl").is_file())

    def test_adapter_disables_reasoning_and_uses_hard_reward(self) -> None:
        cfg = get_dataset("alfworld")
        adapter = load_cobras_adapter("alfworld")
        self.assertEqual(cfg.reward_field, "hard")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        self.assertEqual(adapter.target_reasoning_effort, "")
        self.assertEqual(adapter.eval_max_turns, 0)
        self.assertEqual(adapter.eval_task_timeout, 600)
        self.assertEqual(adapter.rollout_sampling, "task")

    def test_invalid_trajectory_is_retried_in_process(self) -> None:
        cfg = get_dataset("alfworld")
        item = json.loads((cfg.data_root / "train" / "items.json").read_text(encoding="utf-8"))[0]
        provider = ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:8000/v1",
            api_key_env="LOCAL_LLM_API_KEY",
            model="test-model",
        )
        calls = 0

        def fake_run_item(*_args, **kwargs):
            nonlocal calls
            calls += 1
            conversation = kwargs["out_root"] / "predictions" / "train_0000" / "conversation.json"
            conversation.parent.mkdir(parents=True, exist_ok=True)
            conversation.write_text("[]\n", encoding="utf-8")
            return {
                "id": str(item["id"]),
                "hard": 1 if calls == 2 else 0,
                "soft": 1.0 if calls == 2 else 0.0,
                "n_turns": 1,
                "fail_reason": "" if calls == 2 else "missing action tag",
                "agent_ok": calls == 2,
                "task_timed_out": False,
                "task_type": str(item.get("task_type") or "other"),
                "artifact_paths": {"conversation_json": "predictions/train_0000/conversation.json"},
            }

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "cobras.task_envs.alfworld.eval.configure_chat_target",
            return_value=("openai_chat", None, 1024, 300),
        ), patch("cobras.task_envs.alfworld.eval._run_item", side_effect=fake_run_item):
            with redirect_stdout(output):
                artifacts = run_eval(
                    EvalConfig(
                        dataset=cfg,
                        provider=provider,
                        split="train",
                        limit=1,
                        workers=1,
                        out_root=Path(tmp),
                        task_timeout=600,
                        max_turns=30,
                    )
                )
            hard_acc = artifacts.summary["hard_acc"]

        self.assertEqual(calls, 2)
        self.assertEqual(hard_acc, 1.0)
        self.assertEqual(output.getvalue().count(" complete id="), 1)
        self.assertIn("agent_ok=true reward=1", output.getvalue())

    def test_completion_log_reports_agent_ok_and_reward(self) -> None:
        cfg = get_dataset("alfworld")
        item = json.loads((cfg.data_root / "train" / "items.json").read_text(encoding="utf-8"))[0]
        provider = ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:8000/v1",
            api_key_env="LOCAL_LLM_API_KEY",
            model="test-model",
        )

        def fake_run_item(*_args, **kwargs):
            conversation = kwargs["out_root"] / "predictions" / "train_0000" / "conversation.json"
            conversation.parent.mkdir(parents=True, exist_ok=True)
            conversation.write_text("[]\n", encoding="utf-8")
            return {
                "id": str(item["id"]),
                "hard": 1,
                "soft": 1.0,
                "n_turns": 3,
                "fail_reason": "",
                "agent_ok": True,
                "task_timed_out": False,
                "task_type": str(item.get("task_type") or "other"),
                "artifact_paths": {"conversation_json": "predictions/train_0000/conversation.json"},
            }

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "cobras.task_envs.alfworld.eval.configure_chat_target",
            return_value=("openai_chat", None, 1024, 300),
        ), patch("cobras.task_envs.alfworld.eval._run_item", side_effect=fake_run_item):
            with redirect_stdout(output):
                run_eval(
                    EvalConfig(
                        dataset=cfg,
                        provider=provider,
                        split="train",
                        limit=1,
                        workers=1,
                        out_root=Path(tmp),
                        task_timeout=600,
                        max_turns=30,
                    )
                )

        self.assertIn("agent_ok=true reward=1", output.getvalue())
        self.assertNotIn(" hard=1 ", output.getvalue())

    def test_exhausted_invalid_trajectory_is_accepted_as_zero(self) -> None:
        cfg = get_dataset("alfworld")
        item = json.loads((cfg.data_root / "train" / "items.json").read_text(encoding="utf-8"))[0]
        provider = ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:8000/v1",
            api_key_env="LOCAL_LLM_API_KEY",
            model="test-model",
        )

        def fake_run_item(*_args, **kwargs):
            conversation = kwargs["out_root"] / "predictions" / "train_0000" / "conversation.json"
            conversation.parent.mkdir(parents=True, exist_ok=True)
            conversation.write_text("[]\n", encoding="utf-8")
            return {
                "id": str(item["id"]),
                "hard": 0,
                "soft": 0.0,
                "n_turns": 1,
                "fail_reason": "missing action tag",
                "agent_ok": False,
                "task_timed_out": False,
                "task_type": str(item.get("task_type") or "other"),
                "artifact_paths": {"conversation_json": "predictions/train_0000/conversation.json"},
            }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "cobras.task_envs.alfworld.eval.configure_chat_target",
            return_value=("openai_chat", None, 1024, 300),
        ), patch("cobras.task_envs.alfworld.eval._run_item", side_effect=fake_run_item):
            artifacts = run_eval(
                EvalConfig(
                    dataset=cfg,
                    provider=provider,
                    split="train",
                    limit=1,
                    workers=1,
                    out_root=Path(tmp),
                    task_timeout=600,
                    max_turns=30,
                )
            )
            row = json.loads(artifacts.results_path.read_text(encoding="utf-8").strip())
            hard_acc = artifacts.summary["hard_acc"]

        self.assertEqual(hard_acc, 0.0)
        self.assertFalse(row["agent_ok"])
        self.assertTrue(row["accepted_zero"])
        self.assertIn("retries", row["terminal_failure_reason"])

    def test_task_timeout_is_accepted_once_and_cached(self) -> None:
        cfg = get_dataset("alfworld")
        item = json.loads((cfg.data_root / "train" / "items.json").read_text(encoding="utf-8"))[0]
        provider = ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:8000/v1",
            api_key_env="LOCAL_LLM_API_KEY",
            model="test-model",
        )
        calls = 0

        def fake_run_item(*_args, **kwargs):
            nonlocal calls
            calls += 1
            conversation = kwargs["out_root"] / "predictions" / "train_0000" / "conversation.json"
            conversation.parent.mkdir(parents=True, exist_ok=True)
            conversation.write_text("[]\n", encoding="utf-8")
            return {
                "id": str(item["id"]),
                "hard": 0,
                "soft": 0.0,
                "n_turns": 4,
                "fail_reason": "Task timeout after 600s",
                "agent_ok": False,
                "task_timed_out": True,
                "task_type": str(item.get("task_type") or "other"),
                "artifact_paths": {"conversation_json": "predictions/train_0000/conversation.json"},
            }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "cobras.task_envs.alfworld.eval.configure_chat_target",
            return_value=("openai_chat", None, 1024, 300),
        ), patch("cobras.task_envs.alfworld.eval._run_item", side_effect=fake_run_item):
            config = EvalConfig(
                dataset=cfg,
                provider=provider,
                split="train",
                limit=1,
                workers=1,
                out_root=Path(tmp),
                task_timeout=600,
                max_turns=30,
            )
            first = run_eval(config)
            second = run_eval(config)
            row = json.loads(second.results_path.read_text(encoding="utf-8").strip())

        self.assertEqual(calls, 1)
        self.assertEqual(first.summary_path, second.summary_path)
        self.assertTrue(row["accepted_zero"])
        self.assertTrue(row["task_timed_out"])

    def test_eight_trace_sampling_is_uniform_deterministic_and_seeded(self) -> None:
        rows = [{"id": str(index), "hard": index % 2} for index in range(20)]
        first = [row["id"] for row in select_rows(rows, sample_size=8, seed=42)]
        repeated = [row["id"] for row in select_rows(rows, sample_size=8, seed=42)]
        different = [row["id"] for row in select_rows(rows, sample_size=8, seed=1051)]
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)
        self.assertEqual(len(first), 8)

    def test_observations_include_task_trace_and_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction = root / "predictions" / "task_1"
            prediction.mkdir(parents=True)
            (prediction / "conversation.json").write_text(
                json.dumps(
                    [
                        {
                            "type": "task",
                            "task_description": "Put an object in a receptacle.",
                            "initial_observation": "You are in a room.",
                        },
                        {
                            "step": 0,
                            "reasoning": "Inspect first.",
                            "action": "look",
                            "env_feedback": "You see a table.",
                            "reward": 0,
                            "done": False,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            results = root / "results.jsonl"
            results.write_text(
                json.dumps(
                    {
                        "id": "task:1",
                        "task_type": "pick_and_place_simple",
                        "task_description": "Put an object in a receptacle.",
                        "hard": 0,
                        "soft": 0.0,
                        "n_turns": 1,
                        "fail_reason": "timeout",
                        "artifact_paths": {"conversation_json": "predictions/task_1/conversation.json"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            observations, _ = build_observations(
                run_root=root,
                results_path=results,
                sample_size=8,
                sample_seed=42,
            )
            sample = observations["sample_observations"][0]
            self.assertEqual(sample["hard"], 0)
            self.assertEqual(sample["fail_reason"], "timeout")
            self.assertIn("Inspect first.", sample["trajectory"])
            self.assertIn("You see a table.", sample["trajectory"])


if __name__ == "__main__":
    unittest.main()
