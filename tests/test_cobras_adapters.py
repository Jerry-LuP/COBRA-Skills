from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cobras.scripts.cobras_dynamic import COBRASRunner
from cobras.task_envs.cobras_adapter import load_cobras_adapter


class CobrasAdapterTests(unittest.TestCase):
    def test_spreadsheetbench_adapter_preserves_task_defaults(self) -> None:
        adapter = load_cobras_adapter("spreadsheetbench")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        self.assertEqual(adapter.generation_results_filename, "brief_result.jsonl")
        self.assertEqual(adapter.eval_mode, "multi")
        self.assertEqual(adapter.eval_max_turns, 30)
        self.assertFalse(adapter.use_eval_feedback)
        self.assertEqual(adapter.regenerate_env["INCLUDE_REFERENCE_SKILL"], "0")

    def test_searchqa_adapter_preserves_task_defaults(self) -> None:
        adapter = load_cobras_adapter("searchqa")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        self.assertEqual(adapter.generation_results_filename, "results.jsonl")
        self.assertEqual(adapter.target_reasoning_effort, "")
        self.assertEqual(adapter.mutation_env["MUTATION_REJECT_OVER_LENGTH"], "0")
        self.assertNotIn("MUTATION_SAMPLE_POLICY", adapter.mutation_env)

    def test_livemath_adapter_preserves_existing_runtime_contract(self) -> None:
        adapter = load_cobras_adapter("livemath")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        self.assertEqual(adapter.generation_results_filename, "trial_results.jsonl")
        self.assertEqual(adapter.eval_results_filename, "trial_results.jsonl")
        self.assertEqual(adapter.eval_mode, "single")
        self.assertEqual(adapter.eval_max_turns, 1)
        self.assertEqual(adapter.eval_max_completion_tokens, 16384)
        self.assertTrue(adapter.include_prompt_version)
        self.assertEqual(adapter.rollout_sampling, "core")
        self.assertFalse(adapter.include_unevaluated_pool_arms)
        self.assertEqual(adapter.pool_summary_sort_field, "avg_hard_reward")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "count": 50,
                        "success_trials": 13,
                        "avg_hard_reward": 0.26,
                        "avg_soft_reward": 0.24,
                        "results": [{"agent_ok": True}],
                    }
                ),
                encoding="utf-8",
            )
            row = adapter.normalize_eval_summary(
                summary_path=summary_path,
                skill_name="skill",
                skill_root=root,
                workspace_dir=root,
            )
            self.assertEqual(row["avg_hard_reward"], 0.26)
            self.assertEqual(row["avg_soft_reward"], 0.24)
            self.assertEqual(Path(row["results_path"]).name, "trial_results.jsonl")

    def test_docvqa_adapter_normalizes_and_stages_mutation_inputs(self) -> None:
        adapter = load_cobras_adapter("docvqa")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "parent"
            skill_root.mkdir()
            skill_path = skill_root / "skill.md"
            skill_path.write_text("# Parent\n", encoding="utf-8")
            eval_root = root / "eval"
            eval_root.mkdir()
            (eval_root / "results.jsonl").write_text(
                json.dumps({"id": "1", "question": "q", "hard": 1, "soft": 0.75}) + "\n",
                encoding="utf-8",
            )
            (eval_root / "summary.json").write_text(
                json.dumps({"n": 1, "hard": 1, "hard_acc": 1.0, "avg_soft": 0.75}),
                encoding="utf-8",
            )
            row = adapter.normalize_eval_summary(
                summary_path=eval_root / "summary.json",
                skill_name="parent",
                skill_root=skill_root,
                workspace_dir=skill_root,
            )
            self.assertEqual(row["avg_hard_reward"], 1.0)
            self.assertEqual(row["avg_soft_reward"], 0.75)

            out_root = root / "generated" / "skill"
            env = adapter.prepare_mutation_inputs(
                skill_path=skill_path,
                eval_root=eval_root,
                out_root=out_root,
            )
            self.assertTrue((Path(env["SKILLS_ROOT"]) / "skill" / "skill.md").exists())
            self.assertTrue((Path(env["EVAL_ROOT"]) / "skill" / "results.jsonl").exists())
            self.assertEqual(Path(env["OUT_ROOT"]), out_root.parent)

    def test_docvqa_shared_runner_smoke_uses_hard_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool_root = root / "pool"
            rows = []
            for index in range(1, 6):
                skill_root = pool_root / f"skill_{index:03d}"
                skill_root.mkdir(parents=True)
                (skill_root / "skill.md").write_text(f"# Skill {index}\n", encoding="utf-8")
                rows.append(
                    {
                        "skill_name": skill_root.name,
                        "workspace_dir": str(skill_root),
                        "skill_root": str(skill_root),
                        "avg_soft_reward": 0.1 * index,
                        "avg_hard_reward": 0.0,
                    }
                )
            initial_summary = root / "init_eval.json"
            initial_summary.write_text(json.dumps({"results": rows}), encoding="utf-8")
            out_root = root / "run"
            runner = COBRASRunner(
                initial_pool_root=pool_root,
                initial_eval_summary=initial_summary,
                regenerate_baseline_root=root / "baseline",
                out_root=out_root,
                rounds=1,
                pool_size=5,
                prune_count=3,
                selected_eval_limit=4,
                embedding_backend="hash",
                embedding_dim=32,
                dataset="docvqa",
                reward_field="hard",
                test_mode=True,
                seed=7,
            )
            summary = runner.run()
            self.assertEqual(summary["dataset"], "docvqa")
            self.assertEqual(summary["reward_field"], "hard")
            self.assertEqual(len(summary["history"]), 1)
            history = summary["history"][0]
            self.assertEqual(history["reward"], history["hard_reward"])
            self.assertTrue((out_root / "rounds" / "round_000" / "pool_state.json").exists())

    def test_livemath_shared_runner_uses_adapter_and_hard_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool_root = root / "pool"
            rows = []
            for index in range(1, 3):
                skill_root = pool_root / f"skill_{index:03d}"
                skill_root.mkdir(parents=True)
                (skill_root / "skill.md").write_text(f"# Skill {index}\n", encoding="utf-8")
                rows.append(
                    {
                        "skill_name": skill_root.name,
                        "workspace_dir": str(skill_root),
                        "skill_root": str(skill_root),
                    }
                )
            initial_summary = root / "init_eval.json"
            initial_summary.write_text(json.dumps({"results": rows}), encoding="utf-8")

            runner = COBRASRunner(
                initial_pool_root=pool_root,
                initial_eval_summary=initial_summary,
                regenerate_baseline_root=root / "baseline",
                out_root=root / "livemath",
                rounds=1,
                pool_size=2,
                selected_eval_limit=4,
                embedding_backend="hash",
                embedding_dim=32,
                dataset="livemath",
                reward_field="hard",
                test_mode=True,
                seed=11,
            )
            self.assertIsNotNone(runner.dataset_adapter)
            round_dir = root / "livemath" / "round_000"
            runner._evaluate_selected_arm(round_dir, runner.arms[0])
            history = runner.history[0]
            self.assertEqual(history["reward"], history["hard_reward"])
            self.assertTrue(
                (
                    round_dir
                    / "selected_eval"
                    / "arm_001"
                    / "trial_results.jsonl"
                ).exists()
            )

    def test_legacy_socialmaze_runner_uses_soft_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool_root = root / "pool"
            skill_root = pool_root / "skill_001"
            skill_root.mkdir(parents=True)
            (skill_root / "skill.md").write_text("# Skill\n", encoding="utf-8")
            initial_summary = root / "init_eval.json"
            initial_summary.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "skill_name": skill_root.name,
                                "workspace_dir": str(skill_root),
                                "skill_root": str(skill_root),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            runner = COBRASRunner(
                initial_pool_root=pool_root,
                initial_eval_summary=initial_summary,
                regenerate_baseline_root=root / "baseline",
                out_root=root / "socialmaze_hard",
                rounds=1,
                pool_size=1,
                selected_eval_limit=4,
                embedding_backend="hash",
                embedding_dim=32,
                task_package="cobras.task_envs.socialmaze_hard.scripts",
                reward_field="soft",
                test_mode=True,
                seed=11,
            )
            runner._evaluate_selected_arm(root / "socialmaze_hard" / "round_000", runner.arms[0])
            history = runner.history[0]
            self.assertEqual(history["reward"], history["soft_reward"])


if __name__ == "__main__":
    unittest.main()
