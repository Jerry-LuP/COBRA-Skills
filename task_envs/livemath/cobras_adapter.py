from __future__ import annotations

import json
from pathlib import Path

from cobras.task_envs.cobras_adapter import PromptTaskCobrasAdapter


class LiveMathCobrasAdapter(PromptTaskCobrasAdapter):
    """LiveMath result layout while preserving its existing task scripts."""

    def normalize_eval_summary(
        self,
        *,
        summary_path: Path,
        skill_name: str,
        skill_root: Path,
        workspace_dir: Path,
    ) -> dict:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = payload.get("results", [])
        if isinstance(rows, list) and rows:
            agent_ok_count = sum(
                1 for row in rows if isinstance(row, dict) and bool(row.get("agent_ok"))
            )
            if agent_ok_count == 0:
                first_reason = next(
                    (
                        str(row.get("fail_reason", ""))
                        for row in rows
                        if isinstance(row, dict) and row.get("fail_reason")
                    ),
                    "unknown evaluator failure",
                )
                raise RuntimeError(
                    "LiveMath selected eval produced no successful agent executions; "
                    "refusing to record infrastructure failures as zero reward. "
                    f"First failure: {first_reason[:500]}"
                )

        count = int(payload.get("count", payload.get("task_count", payload.get("n", 0))) or 0)
        hard = payload.get("avg_hard_reward", payload.get("hard_acc"))
        if hard is None:
            hard_count = float(payload.get("success_trials", payload.get("hard", 0.0)) or 0.0)
            hard = hard_count / max(count, 1)
        soft = float(payload.get("avg_soft_reward", payload.get("avg_soft", hard)) or 0.0)
        return {
            "skill_name": skill_name,
            "workspace_dir": str(workspace_dir.resolve()),
            "skill_root": str(skill_root.resolve()),
            "avg_soft_reward": soft,
            "avg_hard_reward": float(hard or 0.0),
            "success_tasks": payload.get("success_trials", payload.get("hard")),
            "count": count,
            "summary_path": str(summary_path.resolve()),
            "results_path": str(self.eval_results_path(summary_path.parent).resolve()),
        }

    def prepare_mutation_inputs(
        self,
        *,
        skill_path: Path,
        eval_root: Path,
        out_root: Path,
    ) -> dict[str, str]:
        # The LiveMath mutator consumes the selected skill and rollout root directly.
        del skill_path, eval_root, out_root
        return {str(key): str(value) for key, value in self.mutation_env.items()}


ADAPTER = LiveMathCobrasAdapter(
    dataset="livemath",
    generation_results_filename="trial_results.jsonl",
    eval_results_filename="trial_results.jsonl",
    eval_mode="single",
    eval_max_turns=1,
    eval_max_completion_tokens=16384,
    include_prompt_version=True,
    rollout_sampling="core",
    include_unevaluated_pool_arms=False,
    pool_summary_sort_field="avg_hard_reward",
)
