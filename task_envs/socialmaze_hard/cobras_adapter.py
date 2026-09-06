from __future__ import annotations

from cobras.task_envs.cobras_adapter import PromptTaskCobrasAdapter


ADAPTER = PromptTaskCobrasAdapter(
    dataset="socialmaze_hard",
    generation_results_filename="results.jsonl",
    eval_results_filename="results.jsonl",
    target_reasoning_effort="",
    eval_mode="single",
    eval_max_turns=1,
    eval_max_completion_tokens=2048,
    rollout_sampling="task",
    include_unevaluated_pool_arms=True,
    pool_summary_sort_field="ucb_score",
)
