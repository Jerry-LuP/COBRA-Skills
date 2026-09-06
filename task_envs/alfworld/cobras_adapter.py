from __future__ import annotations

from pathlib import Path

from cobras.task_envs.cobras_adapter import PromptTaskCobrasAdapter


class ALFWorldCobrasAdapter(PromptTaskCobrasAdapter):
    def prepare_mutation_inputs(
        self,
        *,
        skill_path: Path,
        eval_root: Path,
        out_root: Path,
    ) -> dict[str, str]:
        # The ALFWorld mutator reads the selected eval in place through RUN_ROOT.
        return {}


ADAPTER = ALFWorldCobrasAdapter(
    dataset="alfworld",
    generation_results_filename="results.jsonl",
    eval_results_filename="results.jsonl",
    target_reasoning_effort="",
    eval_mode="multi",
    eval_max_turns=0,
    eval_max_completion_tokens=16384,
    eval_task_timeout=600,
    rollout_sampling="task",
    include_unevaluated_pool_arms=True,
    pool_summary_sort_field="ucb_score",
)
