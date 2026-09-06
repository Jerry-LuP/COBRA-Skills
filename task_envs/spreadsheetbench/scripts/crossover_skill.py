from __future__ import annotations

from cobras.task_envs.prompt_crossover import PromptCrossoverSpec, run_prompt_crossover


SPEC = PromptCrossoverSpec(
    dataset="spreadsheetbench",
    system_prompt="You create concise reusable SpreadsheetBench SKILL.md prompt instructions.",
    stage="spreadsheetbench_cobras_crossover",
    instructions=(
        "You are crossing over SpreadsheetBench prompt skills.",
        "Create one child SKILL.md for a weaker SpreadsheetBench code-generation target model.",
        "Use the backbone as the main structure. Borrow only small high-confidence rules from good parents. Use weak parents as negative controls.",
        "Do not include exact training examples, ids, gold answers, or parent-specific score text in the skill.",
        "Prefer a concise prompt skill: short workflow, workbook evidence use, robust openpyxl/pandas editing, target-range discipline, and final-workbook-state checks.",
        "Output only the full child SKILL.md. No code fences, no notes.",
    ),
)


if __name__ == "__main__":
    run_prompt_crossover(SPEC)
