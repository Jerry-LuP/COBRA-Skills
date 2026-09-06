from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from cobras.cobras_core.model import chat_optimizer
from cobras.task_envs.alfworld.skill_data import (
    build_observations,
    configure_optimizer,
    ensure_frontmatter,
    leakage_warnings,
    skill_quality_warnings,
    teacher_settings,
    write_json,
)
from cobras.task_envs.prompt_loader import load_prompt


def main() -> None:
    model, base_url, api_key, effort, max_tokens, timeout = teacher_settings("TEACHER")
    skill_path = Path(os.environ["SKILL_PATH"]).resolve()
    run_root = Path(os.environ["RUN_ROOT"]).resolve()
    out_dir = Path(os.environ["OUT_DIR"]).resolve()
    results_path = run_root / "results.jsonl"
    sample_size = int(os.environ.get("MUTATION_SAMPLE_SIZE", os.environ.get("SAMPLE_SIZE", "8")))
    sample_seed = int(os.environ.get("MUTATION_SAMPLE_SEED", os.environ.get("SAMPLE_SEED", "42")))
    existing_skill = skill_path.read_text(encoding="utf-8")
    observations, sampled_rows = build_observations(
        run_root=run_root,
        results_path=results_path,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )
    meta_prompt = load_prompt("meta_mutate", env="alfworld")
    user = (
        meta_prompt.rstrip()
        + "\n\n## Existing SKILL.md\n"
        + existing_skill.rstrip()
        + "\n\n## Private Rollout Observations\n"
        + json.dumps(observations, ensure_ascii=False, indent=2)
        + "\n\nReturn the full repair-mutated SKILL.md now.\n"
    )
    configure_optimizer(model, effort, base_url, api_key)
    print(
        f"[alfworld-skill-mutate] model={model} hard={observations['hard_correct']}/"
        f"{observations['count']} samples={observations['sample_size']} seed={sample_seed}",
        flush=True,
    )
    response, usage = chat_optimizer(
        system="You repair reusable ALFWorld prompt skills by converting rollout failures into general trigger-action rules.",
        user=user,
        max_completion_tokens=max_tokens,
        retries=5,
        stage="alfworld_skill_mutation",
        reasoning_effort=effort,
        timeout=timeout,
    )
    skill = ensure_frontmatter(
        response,
        name="solve-alfworld-tasks",
        description="Plan and execute reliable admissible-action sequences in ALFWorld household tasks.",
    )
    warnings = leakage_warnings(skill, sampled_rows)
    quality_warnings = skill_quality_warnings(skill, observations)
    rejected = bool(warnings)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "original_skill.md").write_text(existing_skill, encoding="utf-8")
    (out_dir / "mutation_prompt.txt").write_text(user, encoding="utf-8")
    (out_dir / "mutation_raw_response.txt").write_text(response, encoding="utf-8")
    if rejected:
        (out_dir / "rejected_skill.md").write_text(skill, encoding="utf-8")
        (out_dir / "skill.md").write_text(existing_skill.rstrip() + "\n", encoding="utf-8")
    else:
        (out_dir / "skill.md").write_text(skill, encoding="utf-8")
    write_json(out_dir / "observations.json", observations)
    write_json(
        out_dir / "mutation_summary.json",
        {
            "dataset": "alfworld",
            "model": model,
            "reasoning_effort": effort,
            "source_skill": str(skill_path),
            "run_root": str(run_root),
            "sample_size": observations["sample_size"],
            "sample_seed": sample_seed,
            "rejected": rejected,
            "warnings": warnings,
            "quality_warnings": quality_warnings,
            "usage": usage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"[alfworld-skill-mutate] skill={out_dir / 'skill.md'} usage={usage}", flush=True)


if __name__ == "__main__":
    main()
