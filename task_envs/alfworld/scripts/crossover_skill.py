from __future__ import annotations

from cobras.task_envs.prompt_crossover import PromptCrossoverSpec, run_prompt_crossover


SPEC = PromptCrossoverSpec(
    dataset="alfworld",
    system_prompt="You create concise reusable ALFWorld SKILL.md prompt instructions.",
    stage="alfworld_cobras_crossover",
    instructions=(
        "Create one reusable ALFWorld child SKILL.md for a weaker embodied language-model agent.",
        "Treat the high-scoring backbone as the main skill. Preserve its overall strategy, useful wording, structure, and approximate length; do not rewrite it from scratch.",
        "Make at most two to four conceptual edits. Import at most two compatible ideas or bullets from another good parent, and add no new top-level section unless the interaction contract is missing.",
        "Use weak parents only as negative controls for vague, conflicting, repetitive, or counterproductive advice; do not average all parents or rewrite from scratch.",
        "Keep guidance operational and concise. Preserve reasoning in <think>...</think> and exactly one currently admissible action in <action>...</action>.",
        "Do not include task IDs, game paths, exact task descriptions, exact room layouts, numbered object instances, concrete training examples, memorized action sequences, dataset statistics, parent scores, or crossover notes.",
        "Do not invent tools, commands, observations, or environment capabilities. Let the parent evidence determine which compatible ideas matter instead of imposing a predefined strategy.",
        "Return only one complete Markdown SKILL.md with YAML frontmatter containing name and description. No code fences or commentary.",
    ),
)


if __name__ == "__main__":
    run_prompt_crossover(SPEC)
