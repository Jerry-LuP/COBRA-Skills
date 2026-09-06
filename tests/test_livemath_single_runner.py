from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cobras.task_envs.livemath.scripts.run_baseline_single import _run_trial


class LiveMathSingleRunnerTests(unittest.TestCase):
    def test_direct_prompt_is_one_turn_and_contains_task(self) -> None:
        item = {
            "id": "sample:1",
            "question": "Which statement is strongest?",
            "choices": [
                {"label": "A", "text": "First statement"},
                {"label": "B", "text": "Second statement"},
            ],
            "correct_choice": {"label": "A", "text": "First statement"},
            "theorem": "",
            "sketch": "",
            "theorem_type": ["multiple_choice"],
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "cobras.task_envs.livemath.scripts.run_baseline_single.chat_target",
            return_value=("<answer>A</answer>", {"prompt_tokens": 10, "completion_tokens": 3}),
        ) as mocked_chat:
            result = _run_trial(
                item=item,
                trial_index=1,
                run_root=Path(tmp),
                max_completion_tokens=1024,
                prompt_version="direct",
                skill_content="# Strategy\nCompare every choice.",
                use_theorem=False,
                use_sketch=False,
            )

            self.assertEqual(mocked_chat.call_count, 1)
            call = mocked_chat.call_args.kwargs
            self.assertNotIn("tools", call)
            self.assertIn("# Strategy", call["system"])
            self.assertIn("Which statement is strongest?", call["user"])
            self.assertIn("A. First statement", call["user"])
            self.assertEqual(result["n_turns"], 1)

            conversation_path = Path(result["artifact_paths"]["conversation_json"])
            conversation = json.loads(conversation_path.read_text(encoding="utf-8"))
            self.assertEqual([row["role"] for row in conversation], ["system", "user", "assistant"])
            self.assertFalse(any(row.get("type") == "tool_call" for row in conversation))
            self.assertFalse((conversation_path.parent / "task.md").exists())
            self.assertFalse((conversation_path.parent / "refs").exists())


if __name__ == "__main__":
    unittest.main()
