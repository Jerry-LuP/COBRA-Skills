from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cobras.cobras_core.best_skill import aggregate_selected_skills, materialize_best_skill


def history_row(round_index: int, arm_id: str, skill_root: Path, reward: float) -> dict:
    return {
        "round": round_index,
        "iteration": round_index + 1,
        "selected_arm": {
            "arm_id": arm_id,
            "skill_name": skill_root.name,
            "skill_root": str(skill_root),
            "workspace_dir": str(skill_root),
            "origin": "test",
        },
        "selected_eval": {
            "avg_hard_reward": reward,
            "avg_soft_reward": reward / 2,
        },
    }


class BestSkillTests(unittest.TestCase):
    def test_ranking_uses_mean_across_selected_iterations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arm_1 = root / "arm_1"
            arm_2 = root / "arm_2"
            history = [
                history_row(0, "arm_001", arm_1, 0.8),
                history_row(1, "arm_001", arm_1, 0.8),
                history_row(2, "arm_002", arm_2, 1.0),
                history_row(3, "arm_002", arm_2, 0.0),
            ]

            ranking = aggregate_selected_skills(history, "hard")

            self.assertEqual(ranking[0]["arm_id"], "arm_001")
            self.assertEqual(ranking[0]["mean_reward"], 0.8)
            self.assertEqual(ranking[0]["selection_count"], 2)

    def test_materialize_best_skill_copies_skill_and_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            arm_1 = run_root / "skills" / "arm_1"
            arm_2 = run_root / "skills" / "arm_2"
            arm_1.mkdir(parents=True)
            arm_2.mkdir(parents=True)
            (arm_1 / "skill.md").write_text("# Best\n", encoding="utf-8")
            (arm_2 / "skill.md").write_text("# Other\n", encoding="utf-8")
            rows = [
                history_row(0, "arm_001", arm_1, 0.7),
                history_row(1, "arm_001", arm_1, 0.9),
                history_row(2, "arm_002", arm_2, 0.6),
            ]
            (run_root / "history.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (run_root / "summary.json").write_text(
                json.dumps({"reward_field": "hard"}),
                encoding="utf-8",
            )

            selection = materialize_best_skill(run_root)

            self.assertEqual(selection["best_arm"]["arm_id"], "arm_001")
            self.assertEqual(
                (run_root / "optimized_skill" / "skill.md").read_text(encoding="utf-8"),
                "# Best\n",
            )
            self.assertTrue((run_root / "optimized_skill" / "selection.json").exists())


if __name__ == "__main__":
    unittest.main()
