from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cobras.scripts.result_sampling import load_trial_rows


def _row(uid: str) -> dict:
    return {"id": uid, "question": f"question {uid}", "hard_reward": 0.0}


class ResultSamplingTests(unittest.TestCase):
    def test_loads_standard_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trial_results.jsonl"
            rows = [_row("a"), _row("b")]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertEqual(load_trial_rows(path), rows)

    def test_loads_legacy_json_array_with_jsonl_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trial_results.jsonl"
            rows = [_row("a"), _row("b")]
            path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            self.assertEqual(load_trial_rows(path), rows)

    def test_invalid_jsonl_reports_path_and_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trial_results.jsonl"
            path.write_text(json.dumps(_row("a")) + "\nnot-json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"trial_results\.jsonl:2"):
                load_trial_rows(path)


if __name__ == "__main__":
    unittest.main()
