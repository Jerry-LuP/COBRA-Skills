from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cobras.cobras_core.token_usage import (
    normalize_usage,
    record_token_usage,
    summarize_token_events,
    write_usage_reports,
)


class TokenUsageTests(unittest.TestCase):
    def test_normalize_openai_and_responses_usage(self) -> None:
        self.assertEqual(
            normalize_usage({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}),
            {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        )
        self.assertEqual(
            normalize_usage({"input_tokens": 7, "output_tokens": 3}),
            {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        )

    def test_record_and_summarize_by_round_and_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            log_path = out_root / "token_usage" / "events.jsonl"
            base_env = {
                "COBRAS_TOKEN_LOG_PATH": str(log_path),
                "COBRAS_STUDENT_PROVIDER": "openrouter",
                "COBRAS_TEACHER_PROVIDER": "openrouter",
            }
            with patch.dict(os.environ, {**base_env, "COBRAS_ROUND": "0"}, clear=False):
                record_token_usage(
                    role="student",
                    model="openai/gpt-5.4-nano",
                    stage="selected_eval",
                    usage={"prompt_tokens": 100, "completion_tokens": 20},
                )
                record_token_usage(
                    role="teacher",
                    model="openai/gpt-5.4",
                    stage="mutate",
                    usage={"input_tokens": 40, "output_tokens": 10},
                )
            with patch.dict(os.environ, {**base_env, "COBRAS_ROUND": "1"}, clear=False):
                record_token_usage(
                    role="student",
                    model="openai/gpt-5.4-nano",
                    stage="selected_eval",
                    usage={"prompt_tokens": 80, "completion_tokens": 15},
                )

            events = [json.loads(line) for line in log_path.read_text().splitlines()]
            summary = summarize_token_events(events)
            self.assertEqual(summary["rounds"]["0"]["student"]["input_tokens"], 100)
            self.assertEqual(summary["rounds"]["0"]["teacher"]["output_tokens"], 10)
            self.assertEqual(summary["rounds"]["1"]["student"]["total_tokens"], 95)
            self.assertEqual(summary["overall"]["student"]["input_tokens"], 180)

            report = write_usage_reports(out_root)
            self.assertEqual(report["event_count"], 3)
            self.assertTrue((out_root / "token_usage" / "summary.json").exists())
            self.assertTrue((out_root / "token_usage" / "rounds.jsonl").exists())
            self.assertTrue((out_root / "rounds" / "round_000" / "token_usage.json").exists())
            self.assertTrue((out_root / "rounds" / "round_001" / "token_usage.json").exists())


if __name__ == "__main__":
    unittest.main()
