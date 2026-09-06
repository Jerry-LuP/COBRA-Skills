from __future__ import annotations

from cobras.task_envs.prompt_crossover import PromptCrossoverSpec, run_prompt_crossover


SPEC = PromptCrossoverSpec(
    dataset="docvqa",
    system_prompt="You create concise reusable DocVQA SKILL.md prompt instructions.",
    stage="docvqa_cobras_crossover",
    instructions=(
        "You are crossing over DocVQA prompt skills.",
        "Create one child SKILL.md for a weaker one-turn DocVQA target model.",
        "Use the backbone as the main structure. Borrow only small high-confidence rules from good parents. Use weak parents as negative controls.",
        "Do not include exact training examples, ids, gold answers, or parent-specific score text in the skill.",
        "Prefer a concise prompt skill: visual localization, label/table alignment, smallest exact span selection, and strict answer normalization.",
        "Output only the full child SKILL.md. No code fences, no notes.",
    ),
)


if __name__ == "__main__":
    run_prompt_crossover(SPEC)
