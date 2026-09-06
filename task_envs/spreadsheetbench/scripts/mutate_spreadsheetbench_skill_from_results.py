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

META_MUTATION_SKILL = """You are an expert SpreadsheetBench skill editor.

You will receive:
1. An existing SpreadsheetBench SKILL.md.
2. A small random sample of rollout observations produced by a weaker target model using that skill.

Make an append-only conservative mutation. Preserve the existing skill text and behavior as much as possible. If the observations do not support a specific reusable improvement, return the original skill unchanged.

Mutation policy:
- Do not rewrite, reorder, or restyle the parent skill.
- Prefer no-op over a broad or speculative edit.
- Add at most one short section named `## Rollout Mutation Patch` near the end of the skill.
- The patch section may contain 0 to 3 markdown bullets total.
- Each added bullet must be generic, reusable, and under 180 characters.
- Do not restate rules that already appear in the parent skill.
- Do not add concrete training examples, task IDs, sheet names, cell addresses, gold values, or memorized answers.
- Do not add long checklists, broad new workflows, or instructions to inspect every sheet exhaustively.

Use the sampled observations only to identify recurring failure classes. Good patch themes include exact target range, wrong sheet, value-vs-formula, date/time/percentage type preservation, clearing only target output cells, and avoiding broad workbook edits.

Output only the full resulting SKILL.md content. If no patch is justified, output the original SKILL.md unchanged. Do not wrap it in code fences. Do not explain your changes.
"""


def load_env_files() -> None:
    for env_path in [PROJECT_ROOT / ".env.spreadsheetbench.local", PROJECT_ROOT / ".env", ROOT / ".env"]:
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


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clip(text: object, limit: int) -> str:
    s = str(text or "").strip()
    return s if len(s) <= limit else s[:limit] + "\n...[truncated]"


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
    except Exception as exc:
        return f"<failed to read conversation.json: {exc}>"
    rendered = []
    for event in conv:
        role = event.get("role") or event.get("type") or "event"
        turn = event.get("turn")
        prefix = str(role)
        if turn is not None:
            prefix += f"[turn={turn}]"
        rendered.append(f"{prefix}: {clip(event.get('content', ''), max_message_chars)}")
    return clip("\n\n".join(rendered), max_chars)


def parse_answer_position(prompt_text: str) -> str:
    for line in prompt_text.splitlines():
        if line.strip().lower().startswith("expected answer position:"):
            return line.split(":", 1)[1].strip()
    return ""


def sample_rows(rows: list[dict], *, sample_size: int, seed: int | None, failure_fraction: float) -> list[dict]:
    rng = random.Random(seed)
    sample_size = max(0, min(sample_size, len(rows)))
    successes = [row for row in rows if int(row.get("hard") or 0)]
    failures = [row for row in rows if not int(row.get("hard") or 0)]
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def add_random(pool: list[dict]) -> bool:
        candidates = [row for row in pool if str(row.get("id")) not in selected_ids]
        if not candidates:
            return False
        row = rng.choice(candidates)
        selected.append(row)
        selected_ids.add(str(row.get("id")))
        return True

    if sample_size and successes and failures:
        add_random(successes)
        add_random(failures)
    elif sample_size:
        add_random(rows)

    target_failures = round(sample_size * max(0.0, min(1.0, failure_fraction)))
    while len(selected) < sample_size:
        n_fail = sum(1 for row in selected if not int(row.get("hard") or 0))
        if failures and n_fail < target_failures and add_random(failures):
            continue
        if successes and add_random(successes):
            continue
        remaining = [row for row in rows if str(row.get("id")) not in selected_ids]
        if not remaining:
            break
        row = rng.choice(remaining)
        selected.append(row)
        selected_ids.add(str(row.get("id")))
    rng.shuffle(selected)
    return selected


def build_observations(
    *,
    results_path: Path,
    predictions_dir: Path,
    sample_size: int,
    seed: int | None,
    failure_fraction: float,
    include_prompt: bool,
    include_conversation: bool,
    include_raw: bool,
    max_prompt_chars: int,
    max_code_chars: int,
    max_raw_chars: int,
    max_conversation_chars: int,
    max_message_chars: int,
) -> dict:
    rows = load_jsonl(results_path)
    selected = sample_rows(rows, sample_size=sample_size, seed=seed, failure_fraction=failure_fraction)
    samples = []
    for idx, row in enumerate(selected, start=1):
        pred_dir = predictions_dir / str(row.get("id"))
        prompt = read_text(pred_dir / "target_user_prompt.txt", max_prompt_chars)
        item = {
            "sample_label": f"sample_{idx:02d}",
            "id": row.get("id"),
            "outcome": "success" if int(row.get("hard") or 0) else "failure",
            "instruction_type": row.get("instruction_type", ""),
            "task_type": row.get("task_type", ""),
            "answer_position": parse_answer_position(prompt),
            "hard": int(row.get("hard") or 0),
            "soft": float(row.get("soft") or 0.0),
            "n_turns": int(row.get("n_turns") or 0),
            "fail_reason": row.get("fail_reason", ""),
            "generated_code_excerpt": read_text(pred_dir / "code.py", max_code_chars),
        }
        if include_prompt:
            item["prompt_excerpt"] = prompt
        if include_conversation:
            item["conversation_excerpt"] = read_conversation(pred_dir, max_conversation_chars, max_message_chars)
        if include_raw:
            item["raw_response_excerpt"] = read_text(pred_dir / "raw.txt", max_raw_chars)
        samples.append(item)
    hard_correct = sum(int(row.get("hard") or 0) for row in rows)
    return {
        "source_results": str(results_path),
        "count": len(rows),
        "hard_correct": hard_correct,
        "hard_acc": hard_correct / max(len(rows), 1),
        "avg_soft": sum(float(row.get("soft") or 0.0) for row in rows) / max(len(rows), 1),
        "n_failures": sum(1 for row in rows if not int(row.get("hard") or 0)),
        "n_successes": sum(1 for row in rows if int(row.get("hard") or 0)),
        "sample_size": len(samples),
        "sample_seed": seed,
        "sample_observations": samples,
    }


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", text).strip()


def discover_skill_runs(skills_root: Path, eval_root: Path) -> list[tuple[str, Path, Path, Path]]:
    runs = []
    for skill_path in sorted(skills_root.glob("*/skill.md")):
        skill_name = skill_path.parent.name
        run_root = eval_root / skill_name
        results_path = run_root / "results.jsonl"
        predictions_dir = run_root / "predictions"
        if not results_path.exists():
            print(f"[spreadsheet-skill-mutate] skip {skill_name}: missing {results_path}", flush=True)
            continue
        runs.append((skill_name, skill_path, results_path, predictions_dir))
    if not runs:
        raise SystemExit(f"No skill/result pairs found under {skills_root} and {eval_root}")
    return runs


def mutate_one(*, skill_name: str, skill_path: Path, results_path: Path, predictions_dir: Path, out_dir: Path, model: str, base_url: str, api_key: str, reasoning_effort: str | None, sample_size: int, sample_seed: int | None, failure_fraction: float, max_length_ratio: float, max_abs_chars: int, reject_over_length: bool, max_completion_tokens: int, timeout: int) -> dict:
    original_skill = skill_path.read_text(encoding="utf-8")
    observations = build_observations(
        results_path=results_path,
        predictions_dir=predictions_dir,
        sample_size=sample_size,
        seed=sample_seed,
        failure_fraction=failure_fraction,
        include_prompt=env_bool("MUTATION_INCLUDE_PROMPT", True),
        include_conversation=env_bool("MUTATION_INCLUDE_CONVERSATION", True),
        include_raw=env_bool("MUTATION_INCLUDE_RAW", True),
        max_prompt_chars=int(os.environ.get("MAX_PROMPT_CHARS", "3000")),
        max_code_chars=int(os.environ.get("MAX_CODE_CHARS", "3800")),
        max_raw_chars=int(os.environ.get("MAX_RAW_CHARS", "1000")),
        max_conversation_chars=int(os.environ.get("MAX_CONVERSATION_CHARS", "2500")),
        max_message_chars=int(os.environ.get("MAX_MESSAGE_CHARS", "1200")),
    )
    original_len = len(original_skill)
    length_limit = min(max_abs_chars, max(1200, int(original_len * max_length_ratio)))
    user = (
        META_MUTATION_SKILL
        + "\n\n## Mutation Budget\n"
        + f"- Original length: {original_len} characters.\n"
        + f"- Target maximum mutated length: {length_limit} characters.\n"
        + "- Prefer deleting, replacing, or tightening confusing wording before adding new rules.\n"
        + "\n\n## Existing SKILL.md\n"
        + original_skill.strip()
        + "\n\n## Rollout Observations\n"
        + json.dumps(observations, ensure_ascii=False, indent=2)
        + "\n\n## Required Output\nReturn the full mutated SKILL.md. Keep it generic and do not include concrete sampled task IDs or gold values.\n"
    )
    set_optimizer_backend("openai_chat")
    set_optimizer_deployment(model)
    set_reasoning_effort(reasoning_effort)
    configure_azure_openai(endpoint=base_url, api_key=api_key, auth_mode="openai_compatible", optimizer_endpoint=base_url, optimizer_api_key=api_key, optimizer_auth_mode="openai_compatible")
    print(f"[spreadsheet-skill-mutate] skill={skill_name} model={model} hard={observations['hard_correct']}/{observations['count']} samples={observations['sample_size']} seed={sample_seed}", flush=True)
    response, usage = chat_optimizer(system="You conservatively edit reusable SKILL.md instructions for spreadsheet coding agents.", user=user, max_completion_tokens=max_completion_tokens, retries=5, stage="spreadsheetbench_skill_mutation", reasoning_effort=reasoning_effort, timeout=timeout)
    out_dir.mkdir(parents=True, exist_ok=True)
    mutated = strip_code_fence(response)
    rejected = reject_over_length and len(mutated) > length_limit
    (out_dir / "original_skill.md").write_text(original_skill, encoding="utf-8")
    (out_dir / "mutation_prompt.txt").write_text(user, encoding="utf-8")
    (out_dir / "mutation_raw_response.txt").write_text(response, encoding="utf-8")
    if rejected:
        (out_dir / "rejected_skill.md").write_text(mutated + "\n", encoding="utf-8")
        (out_dir / "skill.md").write_text(original_skill.rstrip() + "\n", encoding="utf-8")
    else:
        (out_dir / "skill.md").write_text(mutated + "\n", encoding="utf-8")
    summary = {"skill": skill_name, "source_skill": str(skill_path), "source_results": str(results_path), "out_dir": str(out_dir), "model": model, "sample_size": observations["sample_size"], "sample_seed": sample_seed, "hard_acc_before": observations["hard_acc"], "avg_soft_before": observations["avg_soft"], "original_chars": original_len, "mutated_chars": len(mutated), "target_max_chars": length_limit, "rejected": rejected, "usage": usage}
    (out_dir / "mutation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    load_env_files()
    model = os.environ.get("MUTATION_MODEL", "openai/gpt-5.4")
    base_url = os.environ.get("TEACHER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.environ.get("TEACHER_API_KEY", "")
    reasoning_effort = os.environ.get("MUTATION_REASONING_EFFORT", "medium").strip() or None
    sample_size = int(os.environ.get("MUTATION_SAMPLE_SIZE", "8"))
    seed_raw = os.environ.get("MUTATION_SAMPLE_SEED", "").strip()
    sample_seed = int(seed_raw) if seed_raw else None
    skills_root = Path(os.environ.get("SKILLS_ROOT", str(PROJECT_ROOT / "results/skills/generated_spreadsheetbench_gpt54_medium_sample10_seed1to5")))
    eval_root = Path(os.environ.get("EVAL_ROOT", str(PROJECT_ROOT / "results/gpt-5.4-nano/spreadsheetbench/spreadsheet_generated_gpt54_medium_sample10_seed1to5_train50_parallel15_turn30_reasoninglow_evalfb0")))
    out_root = Path(os.environ.get("OUT_ROOT", str(PROJECT_ROOT / "results/skills/mutated_spreadsheetbench" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))))
    summaries = []
    for idx, (skill_name, skill_path, results_path, predictions_dir) in enumerate(discover_skill_runs(skills_root, eval_root)):
        seed = sample_seed + idx if sample_seed is not None else None
        summaries.append(mutate_one(skill_name=skill_name, skill_path=skill_path, results_path=results_path, predictions_dir=predictions_dir, out_dir=out_root / skill_name, model=model, base_url=base_url, api_key=api_key, reasoning_effort=reasoning_effort, sample_size=sample_size, sample_seed=seed, failure_fraction=float(os.environ.get("MUTATION_FAILURE_FRACTION", "0.65")), max_length_ratio=float(os.environ.get("MUTATION_MAX_LENGTH_RATIO", "1.05")), max_abs_chars=int(os.environ.get("MUTATION_MAX_ABS_CHARS", "3800")), reject_over_length=env_bool("MUTATION_REJECT_OVER_LENGTH", False), max_completion_tokens=int(os.environ.get("MUTATION_MAX_COMPLETION_TOKENS", "6000")), timeout=int(os.environ.get("MUTATION_TIMEOUT", "240"))))
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "mutation_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[spreadsheet-skill-mutate] summary={out_root / 'mutation_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
