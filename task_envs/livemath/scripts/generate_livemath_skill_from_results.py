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


ROOT = Path(__file__).resolve().parents[3]

META_SKILL = """You are an expert skill writer for LiveMath.

Your job is to write a reusable SKILL.md for a weaker LiveMath target model. The target model receives a mathematical multiple-choice question and answer choices. It must return exactly one choice label inside <answer>...</answer>.

Use the observed trajectories and rewards to infer general behavioral rules. The observations may include gold/evaluation feedback. Use that feedback only to identify reusable patterns. Do not memorize, quote, or leak task IDs, exact questions, exact choices, exact gold labels, or concrete training examples.

The generated skill must:
- be generic across LiveMath theorem-reading multiple-choice tasks,
- avoid dataset-specific IDs, exact questions, exact choice text, exact gold labels, or concrete training examples,
- be useful when injected into the target model's system prompt,
- be concise enough to fit comfortably in a prompt,
- preserve enough detail to change model behavior,
- output only one valid SKILL.md with YAML frontmatter.

Do not assume any predefined list of failure modes. Infer what matters from the observations. Prefer broadly applicable operational guidance over dataset-specific rules or examples.

Be willing to choose a distinctive strategy if the observations support it. The best skill may focus on option comparison, theorem reading order, proof-sketch use, answer discipline, uncertainty handling, or another reusable behavior you infer from the data.

Output only the final SKILL.md content. Do not wrap it in code fences.
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
    results_path = run_root / "trial_results.jsonl"
    if results_path.exists():
        rows = []
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        if rows:
            return rows
    summary_path = run_root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = summary.get("results", [])
        if isinstance(rows, list) and rows:
            return [row for row in rows if isinstance(row, dict)]
    raise FileNotFoundError(f"Could not find summary.json or trial_results.jsonl under {run_root}")


def resolve_artifact_path(run_root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path or ""))
    return path if path.is_absolute() else run_root / path


def read_trace(row: dict, *, run_root: Path, max_trace_chars: int, max_response_chars: int) -> str:
    artifacts = row.get("artifact_paths") or {}
    conv_path = resolve_artifact_path(run_root, artifacts.get("conversation_json"))
    if not conv_path.exists():
        trial_root = resolve_artifact_path(run_root, artifacts.get("trial_root"))
        candidate = trial_root / "workspace" / "conversation.json"
        conv_path = candidate if candidate.exists() else conv_path

    parts = []
    if conv_path.exists():
        try:
            conversation = json.loads(conv_path.read_text(encoding="utf-8"))
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
            parts.append("Conversation:\n" + "\n\n".join(rendered))
        except Exception as exc:  # noqa: BLE001
            parts.append(f"Conversation: <failed to read: {exc}>")

    system_path = resolve_artifact_path(run_root, artifacts.get("target_system_prompt"))
    user_path = resolve_artifact_path(run_root, artifacts.get("target_user_prompt"))
    if system_path.exists():
        parts.append("System prompt excerpt:\n" + clip(system_path.read_text(encoding="utf-8"), max_response_chars))
    if user_path.exists():
        parts.append("User prompt excerpt:\n" + clip(user_path.read_text(encoding="utf-8"), max_response_chars))
    return clip("\n\n".join(parts), max_trace_chars)


def compact_choices(row: dict, max_choice_chars: int) -> list[dict]:
    compact = []
    for choice in row.get("choices") or []:
        if isinstance(choice, dict):
            compact.append(
                {
                    "label": choice.get("label"),
                    "text_excerpt": clip(choice.get("text") or choice.get("content") or "", max_choice_chars),
                }
            )
    return compact


def compact_row(
    row: dict,
    *,
    run_root: Path,
    include_trace: bool,
    max_question_chars: int,
    max_choice_chars: int,
    max_response_chars: int,
    max_trace_chars: int,
) -> dict:
    item = {
        "id": row.get("id"),
        "task_type": row.get("task_type"),
        "question_excerpt": clip(row.get("question"), max_question_chars),
        "choices": compact_choices(row, max_choice_chars),
        "correct_label": row.get("correct_label"),
        "correct_text_excerpt": clip(row.get("correct_text"), max_choice_chars),
        "predicted_label": row.get("predicted_label") or row.get("predicted_answer"),
        "predicted_text_excerpt": clip(row.get("predicted_text"), max_choice_chars),
        "response_excerpt": clip(row.get("response"), max_response_chars),
        "hard": int(row.get("hard") or row.get("hard_reward") or 0),
        "soft": float(row.get("soft") or row.get("soft_reward") or 0.0),
        "agent_ok": bool(row.get("agent_ok")),
        "n_turns": row.get("n_turns"),
        "fail_reason": row.get("fail_reason", ""),
    }
    if include_trace:
        item["trace_excerpt"] = read_trace(
            row,
            run_root=run_root,
            max_trace_chars=max_trace_chars,
            max_response_chars=max_response_chars,
        )
    return item


def build_observations(
    *,
    run_root: Path,
    sample_size: int,
    sample_seed: int | None,
    max_question_chars: int,
    max_choice_chars: int,
    max_response_chars: int,
    max_trace_chars: int,
    include_success_traces: bool,
) -> dict:
    rows = load_rows(run_root)
    successes = [row for row in rows if int(row.get("hard") or row.get("hard_reward") or 0)]
    failures = [row for row in rows if not int(row.get("hard") or row.get("hard_reward") or 0)]
    rng = random.Random(sample_seed)
    selected = []
    selected_ids = set()
    sample_size = min(max(sample_size, 0), len(rows))

    def add_one(pool: list[dict]) -> None:
        candidates = [row for row in pool if str(row.get("id")) not in selected_ids]
        if not candidates:
            return
        row = rng.choice(candidates)
        selected.append(row)
        selected_ids.add(str(row.get("id")))

    if sample_size and successes and failures:
        add_one(successes)
        add_one(failures)
    elif sample_size:
        add_one(rows)

    remaining = [row for row in rows if str(row.get("id")) not in selected_ids]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, sample_size - len(selected))])
    rng.shuffle(selected)

    compact = []
    for row in selected:
        hard = int(row.get("hard") or row.get("hard_reward") or 0)
        compact_item = compact_row(
            row,
            run_root=run_root,
            include_trace=(include_success_traces or not hard),
            max_question_chars=max_question_chars,
            max_choice_chars=max_choice_chars,
            max_response_chars=max_response_chars,
            max_trace_chars=max_trace_chars,
        )
        compact_item["outcome"] = "success" if hard else "failure"
        compact.append(compact_item)

    return {
        "source_run_root": str(run_root),
        "count": len(rows),
        "hard_correct": sum(int(row.get("hard") or row.get("hard_reward") or 0) for row in rows),
        "avg_soft": sum(float(row.get("soft") or row.get("soft_reward") or 0.0) for row in rows)
        / max(len(rows), 1),
        "n_successes": len(successes),
        "n_failures": len(failures),
        "sample_size": len(selected),
        "sample_seed": sample_seed,
        "sample_policy": (
            "Random sample from all rows. If both successful and failed rows exist, "
            "at least one success and one failure are included."
        ),
        "sample_observations": compact,
    }


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", text).strip()


def main() -> None:
    load_env_files()
    model = os.environ.get("SKILL_GEN_MODEL", "openai/gpt-5.4")
    base_url = os.environ.get("TEACHER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.environ.get("TEACHER_API_KEY", "")
    reasoning_effort = os.environ.get("SKILL_GEN_REASONING_EFFORT", "medium").strip() or None
    max_completion_tokens = int(os.environ.get("SKILL_GEN_MAX_COMPLETION_TOKENS", "6000"))
    timeout = int(os.environ.get("SKILL_GEN_TIMEOUT", "240"))

    default_run_root = ROOT / "results/cobras/livemath/livemath_openrouter_54nano_baseline_train50_parallel15_turn24_direct"
    run_root = Path(os.environ.get("RUN_ROOT", str(default_run_root)))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        os.environ.get(
            "OUT_DIR",
            str(ROOT / "results/skills/livemath" / f"generated_livemath_gpt54_medium_{timestamp}"),
        )
    )
    sample_size = int(os.environ.get("SAMPLE_SIZE", "10"))
    sample_seed_raw = os.environ.get("SAMPLE_SEED", "").strip()
    sample_seed = int(sample_seed_raw) if sample_seed_raw else None
    include_success_traces = os.environ.get("INCLUDE_SUCCESS_TRACES", "0").strip().lower() in {"1", "true", "yes"}
    max_question_chars = int(os.environ.get("MAX_QUESTION_CHARS", "1600"))
    max_choice_chars = int(os.environ.get("MAX_CHOICE_CHARS", "700"))
    max_response_chars = int(os.environ.get("MAX_RESPONSE_CHARS", "900"))
    max_trace_chars = int(os.environ.get("MAX_TRACE_CHARS", "3000"))

    if not api_key:
        raise SystemExit("Missing OpenRouter API key")
    if not run_root.exists():
        raise SystemExit(f"Baseline run root not found: {run_root}")

    observations = build_observations(
        run_root=run_root,
        sample_size=sample_size,
        sample_seed=sample_seed,
        max_question_chars=max_question_chars,
        max_choice_chars=max_choice_chars,
        max_response_chars=max_response_chars,
        max_trace_chars=max_trace_chars,
        include_success_traces=include_success_traces,
    )

    system = "You write high-quality, generalizable SKILL.md instructions for LLM agents."
    user = (
        META_SKILL
        + "\n\n## Observed Baseline Results\n"
        + json.dumps(observations, ensure_ascii=False, indent=2)
        + "\n\n## Required Output\n"
        + "Write one LiveMath SKILL.md. It must be generic and must not include any exact task IDs, exact questions, exact choice text, exact gold labels, or concrete training examples from the observations.\n"
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
        "[livemath-skill-gen] "
        f"model={model} run_root={run_root} rows={observations['count']} "
        f"failures={observations['n_failures']} successes={observations['n_successes']} "
        f"samples={observations['sample_size']} seed={observations['sample_seed']} "
        f"out={out_dir}",
        flush=True,
    )
    response, usage = chat_optimizer(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=5,
        stage="livemath_skill_generation",
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "skill.md").write_text(strip_code_fence(response) + "\n", encoding="utf-8")
    (out_dir / "skill_generation_prompt.txt").write_text(user, encoding="utf-8")
    (out_dir / "skill_generation_raw_response.txt").write_text(response, encoding="utf-8")
    (out_dir / "skill_generation_summary.json").write_text(
        json.dumps(
            {
                "model": model,
                "base_url": base_url,
                "run_root": str(run_root),
                "out_dir": str(out_dir),
                "n_rows": observations["count"],
                "n_failures": observations["n_failures"],
                "n_successes": observations["n_successes"],
                "sample_size": observations["sample_size"],
                "sample_seed": observations["sample_seed"],
                "sample_policy": observations["sample_policy"],
                "hard_correct": observations["hard_correct"],
                "avg_soft": observations["avg_soft"],
                "usage": usage,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[livemath-skill-gen] skill={out_dir / 'skill.md'}", flush=True)
    print(f"[livemath-skill-gen] usage={usage}", flush=True)


if __name__ == "__main__":
    main()
