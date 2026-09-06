from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from cobras.cobras_core.run_state import (
    ArmState,
    copy_skill_tree as _copy_skill_tree,
    deterministic_rng as _deterministic_rng,
    load_eval_rows as _load_eval_rows,
    load_json as _load_json,
    run_module as _run_module,
    skill_entrypoint_path as _skill_entrypoint_path,
    write_json,
    write_jsonl,
)


class EvaluationOpsMixin:
    """Evaluate a selected skill and normalize its reward record."""

    def _selected_eval_root(self, round_root: Path, arm: ArmState) -> tuple[Path, Path]:
        input_root = round_root / "selected_eval_input"
        workspace_dir = input_root / "workspace_001"
        _copy_skill_tree(Path(arm.skill_root), workspace_dir / "skill")
        eval_out_root = round_root / "selected_eval"
        return input_root, eval_out_root

    def _evaluate_dataset_arm(self, round_root: Path, arm: ArmState) -> tuple[ArmState, dict[str, Any]]:
        if self.dataset_adapter is None:
            raise RuntimeError("Dataset adapter is not configured")
        eval_root = round_root / "selected_eval" / arm.arm_id
        summary_path = eval_root / "summary.json"
        skill_path = _skill_entrypoint_path(Path(arm.skill_root))
        if self.test_mode:
            rng = _deterministic_rng(self.seed, self.dataset, round_root.name, arm.arm_id, "selected_eval")
            count = max(1, self._selected_eval_limit_value())
            hard_reward = float(rng.random() >= 0.5)
            soft_reward = rng.random()
            rows = [
                {
                    "id": f"smoke_{index:03d}",
                    "question": f"synthetic {self.dataset} question {index}",
                    "hard": hard_reward,
                    "soft": soft_reward,
                    "agent_ok": True,
                }
                for index in range(count)
            ]
            write_jsonl(self.dataset_adapter.eval_results_path(eval_root), rows)
            (eval_root / "predictions").mkdir(parents=True, exist_ok=True)
            write_json(
                summary_path,
                {
                    "n": count,
                    "hard": int(hard_reward * count),
                    "hard_acc": hard_reward,
                    "avg_soft": soft_reward,
                    "test_mode": True,
                },
            )
        else:
            command = [
                "--dataset",
                self.dataset,
                "--split",
                self.selected_eval_split,
                "--limit",
                str(self._selected_eval_limit_value()),
                "--workers",
                str(self.eval_parallel),
                "--model",
                self.model,
                "--provider",
                self.provider,
                "--base-url",
                self.base_url,
                "--api-key-env",
                self.api_key_env,
                "--skill-path",
                str(skill_path),
                "--out-root",
                str(eval_root),
                "--mode",
                self.dataset_adapter.eval_mode,
                "--reasoning-effort",
                self.dataset_adapter.target_reasoning_effort,
                "--seed",
                str(self.seed),
            ]
            if self.api_key:
                command.extend(["--api-key", self.api_key])
            eval_max_turns = self.max_turns or self.dataset_adapter.eval_max_turns
            if eval_max_turns > 0:
                command.extend(["--max-turns", str(eval_max_turns)])
            eval_max_completion_tokens = int(
                os.environ.get("COBRAS_TARGET_MAX_COMPLETION_TOKENS", "0") or 0
            ) or self.dataset_adapter.eval_max_completion_tokens
            if eval_max_completion_tokens > 0:
                command.extend(
                    ["--max-completion-tokens", str(eval_max_completion_tokens)]
                )
            if self.dataset_adapter.eval_task_timeout > 0:
                command.extend(["--task-timeout", str(self.dataset_adapter.eval_task_timeout)])
            if self.dataset_adapter.include_prompt_version:
                command.extend(["--prompt-version", self.prompt_version])
            if self.dataset_adapter.use_eval_feedback:
                command.append("--use-eval-feedback")
            if (split_json := self._selected_eval_split_json_path()) is not None:
                command.extend(["--data-path", str(split_json)])
            print(
                f"[cobras-eval] dataset={self.dataset} arm={arm.arm_id} "
                f"split={self.selected_eval_split} limit={self._selected_eval_limit_value()}",
                flush=True,
            )
            _run_module("cobras.scripts.run_baseline", command, stream=True)

        row = self.dataset_adapter.normalize_eval_summary(
            summary_path=summary_path,
            skill_name=arm.skill_name,
            skill_root=Path(arm.skill_root),
            workspace_dir=Path(arm.workspace_dir),
        )
        soft_reward = float(row["avg_soft_reward"])
        hard_reward = float(row["avg_hard_reward"])
        reward = hard_reward if self.reward_field == "hard" else soft_reward
        arm.latest_reward = reward
        arm.latest_soft_reward = soft_reward
        arm.latest_hard_reward = hard_reward
        arm.latest_eval_summary = str(summary_path)
        arm.latest_eval_root = str(eval_root)
        self.history.append(
            {
                "round": len(self.history),
                "arm_id": arm.arm_id,
                "skill_root": arm.skill_root,
                "reward": reward,
                "reward_field": self.reward_field,
                "soft_reward": soft_reward,
                "hard_reward": hard_reward,
                "summary_path": str(summary_path),
                "source": "selected_eval",
            }
        )
        self.dynamic_unique_skill_roots.add(arm.skill_root)
        print(
            f"[cobras-eval] dataset={self.dataset} arm={arm.arm_id} reward={reward:.6f} "
            f"field={self.reward_field} soft={soft_reward:.6f} hard={hard_reward:.6f}",
            flush=True,
        )
        return arm, row

    def _evaluate_selected_arm(self, round_root: Path, arm: ArmState) -> tuple[ArmState, dict[str, Any]]:
        if self.dataset_adapter is not None:
            return self._evaluate_dataset_arm(round_root, arm)
        input_root, eval_out_root = self._selected_eval_root(round_root, arm)
        eval_summary = (
            eval_out_root
            / f"{input_root.name}_model_{self.model.replace('/', '_')}_{self.selected_eval_split}"
            / "summary.json"
        )
        if self.test_mode:
            rng = _deterministic_rng(self.seed, round_root.name, arm.arm_id, "selected_eval")
            soft_reward = rng.random()
            hard_reward = rng.random()
            reward = hard_reward if self.reward_field == "hard" else soft_reward
            row = {
                "skill_name": arm.skill_name,
                "workspace_dir": arm.workspace_dir,
                "skill_root": arm.skill_root,
                "avg_soft_reward": soft_reward,
                "avg_hard_reward": hard_reward,
                "success_tasks": None,
                "count": None,
                "summary_path": str(eval_summary),
                "source": "selected_eval",
            }
            write_json(
                eval_summary,
                {
                    "skills_root": str(input_root),
                    "split": self.selected_eval_split,
                    "limit": self._selected_eval_limit_value(),
                    "model": self.model,
                    "provider": self.provider,
                    "parallel": self.eval_parallel,
                    "skill_parallel": 1,
                    "results": [row],
                    "test_mode": True,
                },
            )
            print(
                f"[cobras-eval] arm={arm.arm_id} done soft={soft_reward:.6f} "
                f"hard={hard_reward:.6f} summary={eval_summary} test_mode=1",
                flush=True,
            )
            arm.latest_reward = reward
            arm.latest_soft_reward = soft_reward
            arm.latest_hard_reward = hard_reward
            arm.latest_eval_summary = str(eval_summary)
            arm.latest_eval_root = str(eval_summary.parent)
            self.history.append(
                {
                    "round": len(self.history),
                    "arm_id": arm.arm_id,
                    "skill_root": arm.skill_root,
                    "reward": reward,
                    "soft_reward": soft_reward,
                    "hard_reward": hard_reward,
                    "summary_path": str(eval_summary),
                    "source": "selected_eval",
                    "test_mode": True,
                }
            )
            self.dynamic_unique_skill_roots.add(arm.skill_root)
            return arm, row
        print(
            f"[cobras-eval] arm={arm.arm_id} skill={arm.skill_name} "
            f"parallel={self.eval_parallel} input={input_root}",
            flush=True,
        )
        _run_module(
            self._module("evaluate_generated_skills"),
            [
                "--skills-root",
                str(input_root),
                "--out-root",
                str(eval_out_root),
                "--split",
                self.selected_eval_split,
                *(
                    ["--split-json", str(split_json)]
                    if (split_json := self._selected_eval_split_json_path()) is not None
                    else []
                ),
                "--parallel",
                str(self.eval_parallel),
                "--skill-parallel",
                "1",
                "--model",
                self.model,
                "--provider",
                self.provider,
                "--limit",
                str(self._selected_eval_limit_value()),
                "--reasoning-effort",
                self.teacher_reasoning_effort,
                "--base-url",
                self.base_url,
                "--api-key-env",
                self.api_key_env,
                "--api-key",
                self.api_key,
                "--openai-base-url-env",
                self.openai_base_url_env,
            ],
            stream=True,
        )
        rows = _load_eval_rows(eval_summary)
        if not rows:
            raise FileNotFoundError(f"No eval rows found in {eval_summary}")
        row = rows[0]
        soft_reward = float(row.get("avg_soft_reward", 0.0))
        hard_reward = float(row.get("avg_hard_reward", 0.0))
        reward = hard_reward if self.reward_field == "hard" else soft_reward
        print(
            f"[cobras-eval] arm={arm.arm_id} reward={reward:.6f} field={self.reward_field} "
            f"soft={soft_reward:.6f} hard={hard_reward:.6f} summary={eval_summary}",
            flush=True,
        )
        arm.latest_reward = reward
        arm.latest_soft_reward = soft_reward
        arm.latest_hard_reward = hard_reward
        arm.latest_eval_summary = str(eval_summary)
        arm.latest_eval_root = str(eval_summary.parent)
        self.history.append(
            {
                "round": len(self.history),
                "arm_id": arm.arm_id,
                "skill_root": arm.skill_root,
                "reward": reward,
                "soft_reward": soft_reward,
                "hard_reward": hard_reward,
                "summary_path": str(eval_summary),
                "source": "selected_eval",
            }
        )
        self.dynamic_unique_skill_roots.add(arm.skill_root)
        return arm, row
