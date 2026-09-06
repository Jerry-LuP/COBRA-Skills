from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime
from pathlib import Path

from cobras.cobras_core.model import (
    chat_optimizer,
    configure_azure_openai,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_reasoning_effort,
)


ROOT = Path(__file__).resolve().parents[3]

META_SKILL = """You are an expert skill writer for SearchQA.

Your job is to write a reusable SKILL.md for a weaker SearchQA target model. The target model receives a CONTEXT and a QUESTION, then must answer inside <answer>...</answer>.

Use the observed trajectories and rewards to infer general behavioral rules. The observations may include gold/evaluation feedback. Use that feedback only to identify reusable patterns. Do not memorize, quote, or leak task IDs, exact gold answers, exact questions, or concrete training examples.

The generated skill must:
- be generic across SearchQA examples,
- avoid dataset-specific IDs, exact gold answers, exact questions, or concrete training examples,
- focus on robust answer extraction, clue interpretation, concise exact-match formatting, and normalization,
- be concise enough to fit comfortably in a system prompt,
- preserve enough detail to change model behavior,
- output only one valid SKILL.md with YAML frontmatter.

Do not assume any predefined list of failure modes. Infer what matters from the observations. Prefer broadly applicable answer-selection and formatting guidance over dataset-specific rules or examples.

Output only the final SKILL.md content. Do not wrap it in code fences.
"""


def load_env_files() -> None:
    for env_path in [ROOT / ".env.searchqa.local", ROOT / ".env"]:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("export "):
                s = s[len("export "):].strip()
            if "=" not in s:
                continue
            key, value = s.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def clip(text: object, limit: int) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...[truncated]"


def read_trace(
    item_id: str,
    *,
    predictions_dir: Path,
    include_context: bool,
    max_trace_chars: int,
    max_response_chars: int,
) -> str:
    pred_dir = predictions_dir / item_id
    parts = []
    conv_path = pred_dir / "conversation.json"
    if conv_path.exists():
        try:
            conv = json.loads(conv_path.read_text(encoding="utf-8"))
            rendered = []
            for event in conv:
                role = event.get("role") or event.get("type") or "event"
                turn = event.get("turn")
                content = event.get("content", "")
                prefix = f"{role}"
                if turn is not None:
                    prefix += f"[turn={turn}]"
                rendered.append(f"{prefix}: {clip(content, max_response_chars)}")
            parts.append("Conversation:\n" + "\n\n".join(rendered))
        except Exception as exc:  # noqa: BLE001
            parts.append(f"Conversation: <failed to read: {exc}>")
    if include_context:
        user_path = pred_dir / "target_user_prompt.txt"
        if user_path.exists():
            parts.append(
                "Prompt excerpt:\n"
                + clip(user_path.read_text(encoding="utf-8"), max_trace_chars)
            )
    return clip("\n\n".join(parts), max_trace_chars)


def compact_row(
    row: dict,
    *,
    include_trace: bool,
    predictions_dir: Path,
    include_context: bool,
    max_trace_chars: int,
    max_response_chars: int,
) -> dict:
    item = {
        "id": row.get("id"),
        "question": row.get("question", ""),
        "gold_answers": row.get("gold_answers", row.get("gold_answer", [])),
        "predicted_answer": row.get("predicted_answer", ""),
        "hard": int(row.get("hard") or 0),
        "soft": float(row.get("soft") or 0.0),
        "agent_ok": bool(row.get("agent_ok")),
        "n_turns": row.get("n_turns"),
        "fail_reason": row.get("fail_reason", ""),
        "response_excerpt": clip(row.get("response", ""), max_response_chars),
    }
    if include_trace:
        item["trace_excerpt"] = read_trace(
            str(row.get("id")),
            predictions_dir=predictions_dir,
            include_context=include_context,
            max_trace_chars=max_trace_chars,
            max_response_chars=max_response_chars,
        )
    return item


def build_observations(
    *,
    results_path: Path,
    predictions_dir: Path,
    sample_size: int,
    sample_seed: int | None,
    max_trace_chars: int,
    max_response_chars: int,
    include_context: bool,
) -> dict:
    rows = load_rows(results_path)
    failures = [row for row in rows if not int(row.get("hard") or 0)]
    successes = [row for row in rows if int(row.get("hard") or 0)]
    rng = random.Random(sample_seed)
    sample_size = max(0, min(sample_size, len(rows)))
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def add_random(pool: list[dict]) -> None:
        candidates = [row for row in pool if str(row.get("id")) not in selected_ids]
        if not candidates:
            return
        row = rng.choice(candidates)
        selected.append(row)
        selected_ids.add(str(row.get("id")))

    if sample_size and successes and failures:
        add_random(successes)
        add_random(failures)
    elif sample_size:
        add_random(rows)

    remaining = [row for row in rows if str(row.get("id")) not in selected_ids]
    rng.shuffle(remaining)
    for row in remaining[: max(0, sample_size - len(selected))]:
        selected.append(row)
        selected_ids.add(str(row.get("id")))
    rng.shuffle(selected)

    def compact_sample(row: dict) -> dict:
        outcome = "success" if int(row.get("hard") or 0) else "failure"
        item = compact_row(
            row,
            include_trace=outcome == "failure",
            predictions_dir=predictions_dir,
            include_context=include_context,
            max_trace_chars=max_trace_chars,
            max_response_chars=max_response_chars,
        )
        item["outcome"] = outcome
        return item

    return {
        "source_results": str(results_path),
        "count": len(rows),
        "hard_correct": sum(int(row.get("hard") or 0) for row in rows),
        "avg_soft": sum(float(row.get("soft") or 0.0) for row in rows)
        / max(len(rows), 1),
        "n_failures": len(failures),
        "n_successes": len(successes),
        "sample_size": len(selected),
        "sample_seed": sample_seed,
        "sample_policy": (
            "Random sample from all rows. If both successful and failed rows exist, "
            "at least one of each is included."
        ),
        "sample_observations": [compact_sample(row) for row in selected],
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

    default_run_root = (
        ROOT
        / "results/gpt-5.4-nano/searchqa/"
        / "searchqa_openrouter_nano_baseline_direct_train50_parallel15_turn1_reasoninglow"
    )
    run_root = Path(os.environ.get("RUN_ROOT", str(default_run_root)))
    results_path = Path(os.environ.get("RESULTS_PATH", str(run_root / "results.jsonl")))
    predictions_dir = Path(os.environ.get("PREDICTIONS_DIR", str(run_root / "predictions")))
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        os.environ.get(
            "OUT_DIR",
            str(ROOT / "results/skills" / f"generated_searchqa_{timestamp}"),
        )
    )
    sample_size = int(os.environ.get("SAMPLE_SIZE", "10"))
    sample_seed_raw = os.environ.get("SAMPLE_SEED", "").strip()
    sample_seed = int(sample_seed_raw) if sample_seed_raw else None
    max_trace_chars = int(os.environ.get("MAX_TRACE_CHARS", "1800"))
    max_response_chars = int(os.environ.get("MAX_RESPONSE_CHARS", "700"))
    include_context = os.environ.get("INCLUDE_CONTEXT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    if not api_key:
        raise SystemExit("Missing OpenRouter API key")
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")

    observations = build_observations(
        results_path=results_path,
        predictions_dir=predictions_dir,
        sample_size=sample_size,
        sample_seed=sample_seed,
        max_trace_chars=max_trace_chars,
        max_response_chars=max_response_chars,
        include_context=include_context,
    )

    system = "You write high-quality, generalizable SKILL.md instructions for LLM agents."
    user = (
        META_SKILL
        + "\n\n## Observed Baseline Results\n"
        + json.dumps(observations, ensure_ascii=False, indent=2)
        + "\n\n## Required Output\n"
        + "Write one SearchQA SKILL.md. It must be generic and must not include any exact question IDs, exact gold answers, or concrete training examples from the observations.\n"
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
        "[searchqa-skill-gen] "
        f"model={model} results={results_path} rows={observations['count']} "
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
        stage="searchqa_skill_generation",
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
                "results_path": str(results_path),
                "predictions_dir": str(predictions_dir),
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

    print(f"[searchqa-skill-gen] skill={out_dir / 'skill.md'}", flush=True)
    print(f"[searchqa-skill-gen] usage={usage}", flush=True)


if __name__ == "__main__":
    main()
