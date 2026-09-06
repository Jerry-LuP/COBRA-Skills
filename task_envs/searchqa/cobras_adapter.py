from __future__ import annotations

from cobras.task_envs.cobras_adapter import PromptTaskCobrasAdapter


ADAPTER = PromptTaskCobrasAdapter(
    dataset="searchqa",
    generation_results_filename="results.jsonl",
    target_reasoning_effort="",
    mutation_env={
        "MUTATION_REJECT_OVER_LENGTH": "0",
        "MUTATION_SKIP_STRONG_THRESHOLD": "0",
    },
)
