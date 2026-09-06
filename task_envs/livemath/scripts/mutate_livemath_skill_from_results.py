from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from cobras.cobras_core.model import (
    chat_optimizer,
    configure_azure_openai,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_reasoning_effort,
)
from cobras.scripts.result_sampling import load_trial_rows


ROOT = Path(__file__).resolve().parents[3]

META_MUTATION_SKILL = """You are an expert LiveMath skill editor.

You will receive:
1. An existing LiveMath SKILL.md.
2. A small random sample of rollout observations produced by a weaker target model using that skill.

Your job is to make a conservative mutation of the existing skill: preserve its overall strategy and useful wording, but refine unclear instructions, delete low-value wording, and improve multiple-choice theorem-reading behavior.

Important constraints:
- Do not rewrite from scratch.
- Make at most 2 to 4 conceptual edits.
- Prefer replacing or deleting wording over adding wording.
- Add at most 2 new bullets total.
- Do not add new top-level sections unless the existing skill is missing a final-answer section.
- If the existing skill is already strong, prefer surgical clarifications over new rules.
- Do not add concrete training examples, exact task IDs, exact question text, exact choice text, exact gold labels, or memorized answers.
- Keep the skill generic and reusable across LiveMath.
- Prefer small, high-signal edits over prompt bloat.
- Keep the mutated skill near the original length when possible.
- Preserve the final-answer contract: answer inside <answer>...</answer>.

Use the observations to infer what should change, but do not optimize for the sampled rows directly.

Output only the full mutated SKILL.md content. Do not wrap it in code fences. Do not explain your changes.
"""


def load_env_files() -> None:
    for env_path in [ROOT / ".env.livemath.local", ROOT / ".env"]:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :].strip()
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def clip(text: object, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]"


def load_rows(run_root: Path) -> list[dict]:
    return load_trial_rows(run_root)


def resolve_artifact_path(run_root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path or ""))
    return path if path.is_absolute() else run_root / path


def read_trace(row: dict, *, run_root: Path, max_trace_chars: int, max_response_chars: int) -> str:
    artifacts = row.get("artifact_paths") or {}
    conv_path = resolve_artifact_path(run_root, artifacts.get("conversation_json"))
    if not conv_path.exists():
        return ""
    try:
        conversation = json.loads(conv_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"<failed to read conversation: {exc}>"
    rendered = []
    for event in conversation:
        role = event.get("role") or event.get("type") or "event"
        turn = event.get("turn")
        label = role if turn is None else f"{role}[turn={turn}]"
        if event.get("cmd"):
            content = f"{event.get('cmd')}\n{event.get('obs', '')}"
        else:
            content = event.get("content", "")
        rendered.append(f"{label}: {clip(content, max_response_chars)}")
    return clip("\n\n".join(rendered), max_trace_chars)


def sample_rows(rows: list[dict], *, sample_size: int, seed: int | None, sample_policy: str) -> list[dict]:
    rng = random.Random(seed)
    sample_size = max(0, min(sample_size, len(rows)))
    if sample_policy == "random":
        selected = list(rows)
        rng.shuffle(selected)
        return selected[:sample_size]

    successes = [row for row in rows if int(row.get("hard") or row.get("hard_reward") or 0)]
    failures = [row for row in rows if not int(row.get("hard") or row.get("hard_reward") or 0)]
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def add(pool: list[dict]) -> None:
        candidates = [row for row in pool if str(row.get("id")) not in selected_ids]
        if not candidates:
            return
        row = rng.choice(candidates)
        selected.append(row)
        selected_ids.add(str(row.get("id")))

    if sample_size and successes and failures:
        add(successes)
        add(failures)
    elif sample_size:
        add(rows)
    remaining = [row for row in rows if str(row.get("id")) not in selected_ids]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, sample_size - len(selected))])
    rng.shuffle(selected)
    return selected


def compact_choices(row: dict, max_choice_chars: int) -> list[dict]:
    choices = []
    for choice in row.get("choices") or []:
        if isinstance(choice, dict):
            choices.append(
                {
                    "label": choice.get("label"),
                    "text_excerpt": clip(choice.get("text") or choice.get("content") or "", max_choice_chars),
                }
            )
    return choices


def build_observations(
    *,
    run_root: Path,
    sample_size: int,
    sample_seed: int | None,
    sample_policy: str,
    max_question_chars: int,
    max_choice_chars: int,
    max_response_chars: int,
    max_trace_chars: int,
) -> dict:
    rows = load_rows(run_root)
    selected = sample_rows(rows, sample_size=sample_size, seed=sample_seed, sample_policy=sample_policy)
    samples = []
    for idx, row in enumerate(selected, start=1):
        hard = int(row.get("hard") or row.get("hard_reward") or 0)
        samples.append(
            {
                "sample_label": f"sample_{idx:02d}",
                "outcome": "success" if hard else "failure",
                "task_type": row.get("task_type"),
                "question_excerpt": clip(row.get("question"), max_question_chars),
                "choices": compact_choices(row, max_choice_chars),
                "correct_label": row.get("correct_label"),
                "correct_text_excerpt": clip(row.get("correct_text"), max_choice_chars),
                "predicted_label": row.get("predicted_label") or row.get("predicted_answer"),
                "predicted_text_excerpt": clip(row.get("predicted_text"), max_choice_chars),
                "response_excerpt": clip(row.get("response"), max_response_chars),
                "hard": hard,
                "soft": float(row.get("soft") or row.get("soft_reward") or 0.0),
                "agent_ok": bool(row.get("agent_ok")),
                "n_turns": row.get("n_turns"),
                "fail_reason": row.get("fail_reason", ""),
                "trace_excerpt": read_trace(
                    row,
                    run_root=run_root,
                    max_trace_chars=max_trace_chars,
                    max_response_chars=max_response_chars,
                ),
            }
        )
    return {
        "source_run_root": str(run_root),
        "count": len(rows),
        "hard_correct": sum(int(row.get("hard") or row.get("hard_reward") or 0) for row in rows),
        "avg_soft": sum(float(row.get("soft") or row.get("soft_reward") or 0.0) for row in rows)
        / max(len(rows), 1),
        "sample_size": len(selected),
        "sample_seed": sample_seed,
        "sample_policy": sample_policy,
        "sample_observations": samples,
    }


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", text).strip()


def main() -> None:
    load_env_files()
    model = os.environ.get("SKILL_MUTATE_MODEL", os.environ.get("SKILL_GEN_MODEL", "openai/gpt-5.4"))
    base_url = os.environ.get("TEACHER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.environ.get("TEACHER_API_KEY", "")
    reasoning_effort = os.environ.get("SKILL_MUTATE_REASONING_EFFORT", "medium").strip() or None
    max_completion_tokens = int(os.environ.get("SKILL_MUTATE_MAX_COMPLETION_TOKENS", "6000"))
    timeout = int(os.environ.get("SKILL_MUTATE_TIMEOUT", "240"))

    parent_skill_path = Path(os.environ["PARENT_SKILL_PATH"])
    run_root = Path(os.environ["RUN_ROOT"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(os.environ.get("OUT_DIR", str(ROOT / "results/skills/livemath" / f"mutated_livemath_{timestamp}")))
    sample_size = int(os.environ.get("SAMPLE_SIZE", "10"))
    sample_seed_raw = os.environ.get("SAMPLE_SEED", "").strip()
    sample_seed = int(sample_seed_raw) if sample_seed_raw else None
    sample_policy = os.environ.get("SAMPLE_POLICY", "random").strip().lower() or "random"

    observations = build_observations(
        run_root=run_root,
        sample_size=sample_size,
        sample_seed=sample_seed,
        sample_policy=sample_policy,
        max_question_chars=int(os.environ.get("MAX_QUESTION_CHARS", "1400")),
        max_choice_chars=int(os.environ.get("MAX_CHOICE_CHARS", "550")),
        max_response_chars=int(os.environ.get("MAX_RESPONSE_CHARS", "750")),
        max_trace_chars=int(os.environ.get("MAX_TRACE_CHARS", "2400")),
    )
    parent_skill = parent_skill_path.read_text(encoding="utf-8")

    system = "You edit high-quality, generalizable SKILL.md instructions for LLM agents."
    user = (
        META_MUTATION_SKILL
        + "\n\n## Existing LiveMath SKILL.md\n"
        + parent_skill
        + "\n\n## Rollout Observations Using This Skill\n"
        + json.dumps(observations, ensure_ascii=False, indent=2)
        + "\n\n## Required Output\n"
        + "Write the full mutated LiveMath SKILL.md. Keep it generic and do not include exact task IDs, exact question text, exact choice text, exact gold labels, or concrete training examples.\n"
    )

    set_optimizer_backend("openai_chat")
    set_optimizer_deployment(model)
    set_reasoning_effort(reasoning_effort)
    configure_azure_openai(
        endpoint=base_url,
        api_key=api_key,
        auth_mode="openai_compatible",
        optimizer_endpoint=base_url,
        optimizer_api_key=api_key,
        optimizer_auth_mode="openai_compatible",
    )
    print(
        "[livemath-skill-mutate] "
        f"model={model} parent={parent_skill_path} run_root={run_root} "
        f"samples={observations['sample_size']} seed={sample_seed} out={out_dir}",
        flush=True,
    )
    response, usage = chat_optimizer(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=5,
        stage="livemath_skill_mutation",
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "skill.md").write_text(strip_code_fence(response) + "\n", encoding="utf-8")
    (out_dir / "mutation_prompt.txt").write_text(user, encoding="utf-8")
    (out_dir / "mutation_raw_response.txt").write_text(response, encoding="utf-8")
    (out_dir / "mutation_summary.json").write_text(
        json.dumps(
            {
                "model": model,
                "base_url": base_url,
                "parent_skill_path": str(parent_skill_path),
                "run_root": str(run_root),
                "out_dir": str(out_dir),
                "sample_size": observations["sample_size"],
                "sample_seed": sample_seed,
                "sample_policy": sample_policy,
                "parent_hard_correct": observations["hard_correct"],
                "parent_avg_soft": observations["avg_soft"],
                "usage": usage,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[livemath-skill-mutate] skill={out_dir / 'skill.md'}", flush=True)
    print(f"[livemath-skill-mutate] usage={usage}", flush=True)


if __name__ == "__main__":
    main()
