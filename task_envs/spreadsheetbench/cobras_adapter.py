from __future__ import annotations

from cobras.task_envs.cobras_adapter import PromptTaskCobrasAdapter


ADAPTER = PromptTaskCobrasAdapter(
    dataset="spreadsheetbench",
    generation_results_filename="brief_result.jsonl",
    target_reasoning_effort="",
    eval_mode="multi",
    eval_max_turns=30,
    use_eval_feedback=False,
    regenerate_env={"INCLUDE_REFERENCE_SKILL": "0"},
    mutation_env={
        "MUTATION_REJECT_OVER_LENGTH": "0",
        "MUTATION_SKIP_STRONG_THRESHOLD": "0",
    },
)
