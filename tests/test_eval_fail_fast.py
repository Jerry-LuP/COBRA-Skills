from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cobras.cobras_core.config import get_dataset
from cobras.cobras_core.llm import ProviderConfig
from cobras.task_envs.alfworld.eval import _is_resumable_row, run_eval as run_alfworld_eval
from cobras.task_envs.eval_errors import (
    EvaluationInfrastructureError,
    is_timeout_error,
)
from cobras.task_envs.eval_types import EvalConfig
from cobras.task_envs.searchqa.rollout import process_one as process_searchqa
from cobras.task_envs.spreadsheetbench.rollout import _find_test_cases
from cobras.scripts.experiment import _evaluation_artifacts_complete


class EvalFailFastTests(unittest.TestCase):
    def test_timeout_classifier_follows_wrapped_exception(self) -> None:
        try:
            try:
                raise TimeoutError("request timed out")
            except TimeoutError as exc:
                raise RuntimeError("LLM retries exhausted") from exc
        except RuntimeError as wrapped:
            self.assertTrue(is_timeout_error(wrapped))

    def test_codex_chroot_timeout_exit_is_a_scoreable_timeout(self) -> None:
        error = RuntimeError(
            "codex exec failed with exit code 124: "
            "[codex-chroot] timeout after 585s; recovered_tokens=33228"
        )

        self.assertTrue(is_timeout_error(error))

    def test_dependency_error_is_not_a_timeout(self) -> None:
        self.assertFalse(is_timeout_error(ModuleNotFoundError("No module named 'omegaconf'")))
        self.assertFalse(
            is_timeout_error(ModuleNotFoundError("No module named 'timeout_decorator'"))
        )
        self.assertFalse(is_timeout_error(OSError("libexample.so: cannot open shared object file")))

    def test_searchqa_dependency_error_aborts(self) -> None:
        item = {
            "id": "q1",
            "question": "Question?",
            "context": "[DOC] Context.",
            "answers": ["answer"],
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "cobras.task_envs.searchqa.rollout.is_target_exec_backend",
            return_value=False,
        ), patch(
            "cobras.task_envs.searchqa.rollout.chat_target",
            side_effect=ModuleNotFoundError("No module named 'required_package'"),
        ):
            with self.assertRaises(EvaluationInfrastructureError):
                process_searchqa(item, tmp, "", exec_timeout=10)

    def test_searchqa_timeout_remains_zero_scored(self) -> None:
        item = {
            "id": "q1",
            "question": "Question?",
            "context": "[DOC] Context.",
            "answers": ["answer"],
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "cobras.task_envs.searchqa.rollout.is_target_exec_backend",
            return_value=False,
        ), patch(
            "cobras.task_envs.searchqa.rollout.chat_target",
            side_effect=TimeoutError("request timed out"),
        ):
            row = process_searchqa(item, tmp, "", exec_timeout=10)

        self.assertEqual(row["hard"], 0)
        self.assertFalse(row["agent_ok"])
        self.assertEqual(row["phase"], "timeout")

    def test_spreadsheet_single_pair_tolerates_identifier_typo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "1_42930_init.xlsx").touch()
            (root / "1_43930_golden.xlsx").touch()
            cases = _find_test_cases(str(root))

        self.assertEqual(len(cases), 1)
        self.assertTrue(cases[0][1].endswith("1_42930_init.xlsx"))
        self.assertTrue(cases[0][2].endswith("1_43930_golden.xlsx"))

    def test_resume_completeness_rejects_runtime_failure_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text('{"count": 1}\n', encoding="utf-8")
            (root / "results.jsonl").write_text(
                '{"id":"1","agent_ok":false,"phase":"error",'
                '"fail_reason":"unexpected: ModuleNotFoundError"}\n',
                encoding="utf-8",
            )
            self.assertFalse(
                _evaluation_artifacts_complete(root, "results.jsonl", expected=1)
            )
            (root / "results.jsonl").write_text(
                '{"id":"1","agent_ok":false,"phase":"timeout",'
                '"fail_reason":"task-timeout-600s"}\n',
                encoding="utf-8",
            )
            self.assertTrue(
                _evaluation_artifacts_complete(root, "results.jsonl", expected=1)
            )

    def test_alfworld_worker_dependency_error_aborts(self) -> None:
        dataset = get_dataset("alfworld")
        provider = ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:8000/v1",
            api_key_env="LOCAL_LLM_API_KEY",
            model="test-model",
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "cobras.task_envs.alfworld.eval.configure_chat_target",
            return_value=("openai_chat", None, 1024, 300),
        ), patch(
            "cobras.task_envs.alfworld.eval._run_item",
            side_effect=RuntimeError(
                "Failed to start worker: ModuleNotFoundError: No module named 'omegaconf'"
            ),
        ):
            with self.assertRaises(EvaluationInfrastructureError):
                run_alfworld_eval(
                    EvalConfig(
                        dataset=dataset,
                        provider=provider,
                        split="train",
                        limit=1,
                        workers=1,
                        out_root=Path(tmp),
                        max_turns=30,
                    )
                )

    def test_alfworld_resume_rejects_old_worker_failure_zero(self) -> None:
        old_bad_row = {
            "id": "val:0001",
            "agent_ok": False,
            "accepted_zero": True,
            "task_timed_out": False,
            "terminal_failure_reason": (
                "worker failed after retries: ModuleNotFoundError: "
                "No module named 'omegaconf'"
            ),
        }
        self.assertFalse(_is_resumable_row(old_bad_row))
        self.assertTrue(
            _is_resumable_row(
                {
                    **old_bad_row,
                    "task_timed_out": True,
                    "terminal_failure_reason": "accepted after 600s task timeout",
                }
            )
        )
        self.assertTrue(
            _is_resumable_row(
                {
                    **old_bad_row,
                    "terminal_failure_reason": "accepted after 1 trajectory retries",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
