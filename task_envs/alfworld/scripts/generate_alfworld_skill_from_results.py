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


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    model, base_url, api_key, effort, max_tokens, timeout = teacher_settings("TEACHER")
    run_root = Path(os.environ["RUN_ROOT"]).resolve()
    results_path = Path(os.environ.get("RESULTS_PATH", str(run_root / "results.jsonl"))).resolve()
    out_dir = Path(os.environ["OUT_DIR"]).resolve()
    sample_size = int(os.environ.get("SAMPLE_SIZE", "8"))
    sample_seed = int(os.environ.get("SAMPLE_SEED", "42"))
    observations, sampled_rows = build_observations(
        run_root=run_root,
        results_path=results_path,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )
    meta_prompt = load_prompt("meta_generate", env="alfworld")
    user = (
        meta_prompt.rstrip()
        + "\n\n## Private Rollout Observations\n"
        + json.dumps(observations, ensure_ascii=False, indent=2)
        + "\n\nReturn the complete ALFWorld SKILL.md now.\n"
    )
    configure_optimizer(model, effort, base_url, api_key)
    print(
        f"[alfworld-skill-gen] model={model} rows={observations['count']} "
        f"samples={observations['sample_size']} seed={sample_seed} out={out_dir}",
        flush=True,
    )
    response, usage = chat_optimizer(
        system="You write concise, reusable prompt skills for embodied language-model agents.",
        user=user,
        max_completion_tokens=max_tokens,
        retries=5,
        stage="alfworld_skill_generation",
        reasoning_effort=effort,
        timeout=timeout,
    )
    skill = ensure_frontmatter(
        response,
        name="solve-alfworld-tasks",
        description="Plan and execute reliable admissible-action sequences in ALFWorld household tasks.",
    )
    warnings = leakage_warnings(skill, sampled_rows)
    if warnings:
        raise RuntimeError("Generated skill failed leakage checks: " + ", ".join(warnings))
    quality_warnings = skill_quality_warnings(skill, observations)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "skill.md").write_text(skill, encoding="utf-8")
    (out_dir / "skill_generation_prompt.txt").write_text(user, encoding="utf-8")
    (out_dir / "skill_generation_raw_response.txt").write_text(response, encoding="utf-8")
    write_json(out_dir / "observations.json", observations)
    write_json(
        out_dir / "skill_generation_summary.json",
        {
            "dataset": "alfworld",
            "model": model,
            "base_url": base_url,
            "reasoning_effort": effort,
            "run_root": str(run_root),
            "results_path": str(results_path),
            "sample_size": observations["sample_size"],
            "sample_seed": sample_seed,
            "quality_warnings": quality_warnings,
            "usage": usage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"[alfworld-skill-gen] skill={out_dir / 'skill.md'} usage={usage}", flush=True)


if __name__ == "__main__":
    main()
