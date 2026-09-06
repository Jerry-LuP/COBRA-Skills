from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from cobras.task_envs.spreadsheetbench.evaluator import (
    _generate_cell_names,
)
from cobras.task_envs.spreadsheetbench.rollout import _get_answer_position
from cobras.task_envs.eval_errors import EvaluationInfrastructureError
from cobras.task_envs.spreadsheetbench.rollout import run_spreadsheet_batch_codegen


class SpreadsheetBenchEvaluatorTests(unittest.TestCase):
    def test_task_60_7_answer_position_fix(self) -> None:
        item = {
            "id": "60-7",
            "answer_position": "A3:E11",
            "answer_sheet": "Consolidated Tracker,Existing Task,Additions,Retired",
        }
        self.assertEqual(
            _get_answer_position(item),
            "'Consolidated Tracker'!A3:E11",
        )

    def test_task_283_32_answer_position_fix(self) -> None:
        item = {
            "id": "283-32",
            "answer_position": "Sheet3'!A:G,'Sheet4'!A:G",
            "answer_sheet": "'Sheet3','Sheet4'",
        }
        self.assertEqual(
            _get_answer_position(item),
            "'Sheet3'!A:G,'Sheet4'!A:G",
        )

    def test_other_tasks_keep_migration_behavior(self) -> None:
        item = {
            "id": "other",
            "answer_position": "A1:B2",
            "answer_sheet": "Sheet1",
        }
        self.assertEqual(_get_answer_position(item), "Sheet1!A1:B2")

    def test_column_only_range_uses_worksheet_height(self) -> None:
        self.assertEqual(
            _generate_cell_names("A:G", max_row=2),
            [f"{col}{row}" for col in "ABCDEFG" for row in (1, 2)],
        )


    def test_batch_persists_completed_rows_before_later_failure(self) -> None:
        completed = {
            "id": "ok",
            "phase": "exec",
            "hard": 1,
            "soft": 1.0,
            "n_turns": 1,
            "n_cases": 1,
            "n_pass": 1,
        }

        def fake_process(item, *_args, **_kwargs):
            if item["id"] == "ok":
                return completed
            time.sleep(0.05)
            raise RuntimeError("synthetic infrastructure failure")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "cobras.task_envs.spreadsheetbench.rollout.process_one_codegen",
            side_effect=fake_process,
        ):
            with self.assertRaises(EvaluationInfrastructureError):
                run_spreadsheet_batch_codegen(
                    items=[{"id": "ok"}, {"id": "bad"}],
                    data_root=tmp,
                    out_root=tmp,
                    skill_content="",
                    max_api_workers=1,
                    task_timeout=0,
                )

            rows = [
                json.loads(line)
                for line in (Path(tmp) / "results.jsonl").read_text().splitlines()
            ]
        self.assertEqual(rows, [completed])


if __name__ == "__main__":
    unittest.main()
