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
PROJECT_ROOT = ROOT

META_SKILL = """You are an expert skill writer for SpreadsheetBench.

Your job is to write a reusable SKILL.md for a weaker SpreadsheetBench target model. The target model receives a spreadsheet manipulation instruction, an xlsx preview, and an expected answer position. It must produce one Python script using INPUT_PATH and OUTPUT_PATH. The script is executed, and the saved workbook is evaluated.

Use the observed trajectories and rewards to infer general behavioral rules. The observations may include gold/evaluation feedback. Use that feedback only to identify reusable patterns. Do not memorize, quote, or leak task IDs, exact target cells with their gold values, exact training answers, or concrete training examples.

The generated skill must:
- be generic across SpreadsheetBench tasks,
- focus on robust openpyxl/pandas spreadsheet manipulation behavior,
- be useful when injected into the target model's system prompt,
- be concise enough to fit comfortably in a prompt,
- preserve enough detail to change model behavior,
- output only one valid SKILL.md with YAML frontmatter.

Do not assume any predefined list of failure modes. Infer what matters from the observations. Prefer broadly applicable operational guidance over dataset-specific rules or examples.

Output only the final SKILL.md content. Do not wrap it in code fences.
"""

def load_env_files() -> None:
    for env_path in [
        PROJECT_ROOT / ".env.spreadsheetbench.local",
        PROJECT_ROOT / ".env",
        ROOT / ".env.spreadsheetbench.local",
        ROOT / ".env",
    ]:
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


def load_jsonl(path: Path) -> list[dict]:
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


def read_text(path: Path, limit: int) -> str:
    if not path.exists():
        return ""
    try:
        return clip(path.read_text(encoding="utf-8"), limit)
    except UnicodeDecodeError:
        return clip(path.read_text(errors="ignore"), limit)


def read_conversation(pred_dir: Path, max_chars: int, max_message_chars: int) -> str:
    conv_path = pred_dir / "conversation.json"
    if not conv_path.exists():
        return ""
    try:
        conv = json.loads(conv_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"<failed to read conversation.json: {exc}>"
    rendered = []
    for event in conv:
        role = event.get("role") or event.get("type") or "event"
        turn = event.get("turn")
        content = event.get("content", "")
        prefix = str(role)
        if turn is not None:
            prefix += f"[turn={turn}]"
        rendered.append(f"{prefix}: {clip(content, max_message_chars)}")
    return clip("\n\n".join(rendered), max_chars)


def parse_answer_position(prompt_text: str) -> str:
    for line in prompt_text.splitlines():
        if line.strip().lower().startswith("expected answer position:"):
            return line.split(":", 1)[1].strip()
    return ""


def compact_row(
    row: dict,
    *,
    predictions_dir: Path,
    include_prompt: bool,
    include_conversation: bool,
    max_prompt_chars: int,
    max_code_chars: int,
    max_raw_chars: int,
    max_conversation_chars: int,
    max_message_chars: int,
) -> dict:
    item_id = str(row.get("id"))
    pred_dir = predictions_dir / item_id
    prompt = read_text(pred_dir / "target_user_prompt.txt", max_prompt_chars)
    compact = {
        "id": item_id,
        "instruction_type": row.get("instruction_type", ""),
        "task_type": row.get("task_type", ""),
        "answer_position": parse_answer_position(prompt),
        "hard": int(row.get("hard") or 0),
        "soft": float(row.get("soft") or 0.0),
        "n_pass": int(row.get("n_pass") or 0),
        "n_cases": int(row.get("n_cases") or 0),
        "n_turns": int(row.get("n_turns") or 0),
        "llm_ok": bool(row.get("llm_ok")),
        "code_ok": bool(row.get("code_ok")),
        "exec_ok": bool(row.get("exec_ok")),
        "fail_reason": row.get("fail_reason", ""),
        "generated_code_excerpt": read_text(pred_dir / "code.py", max_code_chars),
    }
    if include_prompt:
        compact["prompt_excerpt"] = prompt
    if include_conversation:
        compact["conversation_excerpt"] = read_conversation(
            pred_dir,
            max_chars=max_conversation_chars,
            max_message_chars=max_message_chars,
        )
    raw = read_text(pred_dir / "raw.txt", max_raw_chars)
    if raw:
        compact["raw_response_excerpt"] = raw
    return compact


def select_rows(rows: list[dict], sample_size: int, sample_seed: int | None) -> list[dict]:
    rng = random.Random(sample_seed)
    successes = [row for row in rows if int(row.get("hard") or 0)]
    failures = [row for row in rows if not int(row.get("hard") or 0)]
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def add_random(pool: list[dict]) -> None:
        candidates = [row for row in pool if str(row.get("id")) not in selected_ids]
        if not candidates:
            return
        row = rng.choice(candidates)
        selected.append(row)
        selected_ids.add(str(row.get("id")))

    sample_size = max(0, min(sample_size, len(rows)))
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
    return selected


def build_observations(
    *,
    results_path: Path,
    predictions_dir: Path,
    sample_size: int,
    sample_seed: int | None,
    include_prompt: bool,
    include_conversation: bool,
    max_prompt_chars: int,
    max_code_chars: int,
    max_raw_chars: int,
    max_conversation_chars: int,
    max_message_chars: int,
) -> dict:
    rows = load_jsonl(results_path)
    selected = select_rows(rows, sample_size, sample_seed)
    failures = [row for row in rows if not int(row.get("hard") or 0)]
    successes = [row for row in rows if int(row.get("hard") or 0)]

    samples = []
    for row in selected:
        item = compact_row(
            row,
            predictions_dir=predictions_dir,
            include_prompt=include_prompt,
            include_conversation=include_conversation,
            max_prompt_chars=max_prompt_chars,
            max_code_chars=max_code_chars,
            max_raw_chars=max_raw_chars,
            max_conversation_chars=max_conversation_chars,
            max_message_chars=max_message_chars,
        )
        item["outcome"] = "success" if int(row.get("hard") or 0) else "failure"
        samples.append(item)

    return {
        "source_results": str(results_path),
        "predictions_dir": str(predictions_dir),
        "count": len(rows),
        "hard_correct": sum(int(row.get("hard") or 0) for row in rows),
        "avg_soft": sum(float(row.get("soft") or 0.0) for row in rows) / max(len(rows), 1),
        "n_failures": len(failures),
        "n_successes": len(successes),
        "sample_size": len(samples),
        "sample_seed": sample_seed,
        "sample_policy": (
            "Random sample from all rows. If both successful and failed rows exist, "
            "at least one of each is included. Full fail_reason and evaluation traces may include gold values."
        ),
        "sample_observations": samples,
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
    max_completion_tokens = int(os.environ.get("SKILL_GEN_MAX_COMPLETION_TOKENS", "7000"))
    timeout = int(os.environ.get("SKILL_GEN_TIMEOUT", "300"))

    default_run_root = (
        PROJECT_ROOT
        / "results/gpt-5.4-nano/spreadsheetbench"
        / "spreadsheetbench_openrouter_nano_baseline_multi_train50_parallel15_turn30_reasoninglow"
    )
    run_root = Path(os.environ.get("RUN_ROOT", str(default_run_root)))
    results_path = Path(os.environ.get("RESULTS_PATH", str(run_root / "brief_result.jsonl")))
    predictions_dir = Path(os.environ.get("PREDICTIONS_DIR", str(run_root / "predictions")))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        os.environ.get(
            "OUT_DIR",
            str(PROJECT_ROOT / "results/skills" / f"generated_spreadsheetbench_{timestamp}"),
        )
    )
    sample_size = int(os.environ.get("SAMPLE_SIZE", "10"))
    sample_seed_raw = os.environ.get("SAMPLE_SEED", "").strip()
    sample_seed = int(sample_seed_raw) if sample_seed_raw else None
    include_prompt = os.environ.get("INCLUDE_PROMPT", "1").strip().lower() in {"1", "true", "yes"}
    include_conversation = os.environ.get("INCLUDE_CONVERSATION", "1").strip().lower() in {"1", "true", "yes"}
    max_prompt_chars = int(os.environ.get("MAX_PROMPT_CHARS", "3500"))
    max_code_chars = int(os.environ.get("MAX_CODE_CHARS", "4500"))
    max_raw_chars = int(os.environ.get("MAX_RAW_CHARS", "1200"))
    max_conversation_chars = int(os.environ.get("MAX_CONVERSATION_CHARS", "2500"))
    max_message_chars = int(os.environ.get("MAX_MESSAGE_CHARS", "1200"))
    reference_skill_path = Path(
        os.environ.get(
            "REFERENCE_SKILL_PATH",
            str(PROJECT_ROOT / "results/skills/spreadsheetbench_manual_skills/spreadsheet_formula_value_guard_v2/skill.md"),
        )
    )

    if not api_key:
        raise SystemExit("Missing OpenRouter API key")
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")
    if not predictions_dir.exists():
        raise SystemExit(f"Predictions dir not found: {predictions_dir}")

    observations = build_observations(
        results_path=results_path,
        predictions_dir=predictions_dir,
        sample_size=sample_size,
        sample_seed=sample_seed,
        include_prompt=include_prompt,
        include_conversation=include_conversation,
        max_prompt_chars=max_prompt_chars,
        max_code_chars=max_code_chars,
        max_raw_chars=max_raw_chars,
        max_conversation_chars=max_conversation_chars,
        max_message_chars=max_message_chars,
    )

    reference_skill = ""
    if reference_skill_path.exists() and os.environ.get("INCLUDE_REFERENCE_SKILL", "0").strip().lower() in {"1", "true", "yes"}:
        reference_skill = reference_skill_path.read_text(encoding="utf-8")

    system = "You write high-quality, generalizable SKILL.md instructions for LLM coding agents."
    user = (
        META_SKILL
        + "\n\n## Observed Baseline SpreadsheetBench Results\n"
        + json.dumps(observations, ensure_ascii=False, indent=2)
    )
    if reference_skill:
        user += (
            "\n\n## Optional Reference Skill\n"
            "The following hand-written skill performed well in a prior experiment. You may borrow general ideas, "
            "but improve from the observed baseline trajectories and keep the final skill generic.\n\n"
            + reference_skill.strip()
        )
    user += (
        "\n\n## Required Output\n"
        "Write one SpreadsheetBench SKILL.md. It must be generic and must not include exact task IDs, exact gold values, "
        "or concrete training examples from the observations.\n"
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
        "[spreadsheet-skill-gen] "
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
        stage="spreadsheetbench_skill_generation",
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    skill_text = strip_code_fence(response) + "\n"
    (out_dir / "skill.md").write_text(skill_text, encoding="utf-8")
    (out_dir / "skill_generation_prompt.txt").write_text(user, encoding="utf-8")
    (out_dir / "skill_generation_raw_response.txt").write_text(response, encoding="utf-8")
    (out_dir / "observations.json").write_text(
        json.dumps(observations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "skill_generation_summary.json").write_text(
        json.dumps(
            {
                "model": model,
                "base_url": base_url,
                "results_path": str(results_path),
                "predictions_dir": str(predictions_dir),
                "out_dir": str(out_dir),
                "reference_skill_path": str(reference_skill_path) if reference_skill else "",
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

    print(f"[spreadsheet-skill-gen] skill={out_dir / 'skill.md'}", flush=True)
    print(f"[spreadsheet-skill-gen] usage={usage}", flush=True)


if __name__ == "__main__":
    main()
