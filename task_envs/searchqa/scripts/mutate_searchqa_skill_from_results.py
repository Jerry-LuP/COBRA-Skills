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

META_MUTATION_SKILL = """You are an expert SearchQA skill editor.

You will receive:
1. An existing SearchQA SKILL.md.
2. A small random sample of rollout observations produced by a weaker target model using that skill.

Your job is to make a conservative mutation of the existing skill: preserve its overall structure and intent, but refine wording, replace confusing instructions, delete low-value wording, and improve exact-match behavior.

Important constraints:
- Do not rewrite from scratch.
- Make at most 2 to 4 conceptual edits.
- Prefer replacing or deleting wording over adding wording.
- Add at most 2 new bullets total.
- Do not add new top-level sections unless the existing skill is missing a final-answer section.
- If the existing skill is already strong, prefer surgical clarifications over new rules.
- Do not add concrete training examples, exact question IDs, exact gold answers, or memorized entities.
- Do not leak dataset-specific answers.
- Keep the skill generic and reusable across SearchQA.
- Prefer small, high-signal edits over large prompt bloat.
- Keep the mutated skill at or below the original length when possible.
- Preserve the final-answer contract: answer inside <answer>...</answer>.

Focus on general SearchQA improvements:
- answer type matching,
- clue interpretation,
- concise exact-match formatting,
- avoiding explanations, parentheticals, aliases, multiple candidates, and truth-judgment answers,
- preserving necessary modifiers while removing unnecessary decorations.

Output only the full mutated SKILL.md content. Do not wrap it in code fences. Do not explain your changes.
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


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def sanitize_fail_reason(text: object) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    if "expected" in s.lower():
        return "Exact-match failure; the predicted answer did not match the gold answer."
    return clip(s, 300)


def read_trace(
    item_id: str,
    *,
    predictions_dir: Path,
    max_trace_chars: int,
    max_response_chars: int,
) -> str:
    conv_path = predictions_dir / item_id / "conversation.json"
    if not conv_path.exists():
        return ""
    try:
        conv = json.loads(conv_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"<failed to read conversation: {exc}>"
    rendered = []
    for event in conv:
        role = event.get("role") or event.get("type") or "event"
        turn = event.get("turn")
        content = event.get("content", "")
        prefix = f"{role}"
        if turn is not None:
            prefix += f"[turn={turn}]"
        rendered.append(f"{prefix}: {clip(content, max_response_chars)}")
    return clip("\n\n".join(rendered), max_trace_chars)


def sample_rows(
    rows: list[dict],
    *,
    sample_size: int,
    seed: int | None,
    failure_fraction: float,
) -> list[dict]:
    rng = random.Random(seed)
    sample_size = max(0, min(sample_size, len(rows)))
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

    if sample_size and successes and failures:
        add_random(successes)
        add_random(failures)
    elif sample_size:
        add_random(rows)

    target_failures = round(sample_size * max(0.0, min(1.0, failure_fraction)))
    target_successes = sample_size - target_failures
    while len(selected) < sample_size:
        n_fail = sum(1 for row in selected if not int(row.get("hard") or 0))
        n_success = len(selected) - n_fail
        if failures and n_fail < target_failures:
            before = len(selected)
            add_random(failures)
            if len(selected) > before:
                continue
        if successes and n_success < target_successes:
            before = len(selected)
            add_random(successes)
            if len(selected) > before:
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
    include_ids: bool,
    include_gold: bool,
    include_questions: bool,
    max_trace_chars: int,
    max_response_chars: int,
) -> dict:
    rows = load_rows(results_path)
    selected = sample_rows(
        rows,
        sample_size=sample_size,
        seed=seed,
        failure_fraction=failure_fraction,
    )

    def compact(idx: int, row: dict) -> dict:
        hard = int(row.get("hard") or 0)
        item = {
            "sample_label": f"sample_{idx:02d}",
            "outcome": "success" if hard else "failure",
            "predicted_answer": row.get("predicted_answer", ""),
            "hard": hard,
            "soft": float(row.get("soft") or 0.0),
            "agent_ok": bool(row.get("agent_ok")),
            "n_turns": row.get("n_turns"),
            "response_excerpt": clip(row.get("response", ""), max_response_chars),
        }
        if include_ids:
            item["id"] = row.get("id")
        if include_questions:
            item["question"] = row.get("question", "")
        if include_gold:
            item["gold_answers"] = row.get("gold_answers", row.get("gold_answer", []))
            item["fail_reason"] = clip(row.get("fail_reason", ""), 500)
        else:
            item["fail_reason"] = sanitize_fail_reason(row.get("fail_reason", ""))
        if not hard:
            item["trace_excerpt"] = read_trace(
                str(row.get("id")),
                predictions_dir=predictions_dir,
                max_trace_chars=max_trace_chars,
                max_response_chars=max_response_chars,
            )
        return item

    hard_correct = sum(int(row.get("hard") or 0) for row in rows)
    return {
        "source_results": str(results_path),
        "count": len(rows),
        "hard_correct": hard_correct,
        "hard_acc": hard_correct / max(len(rows), 1),
        "avg_soft": sum(float(row.get("soft") or 0.0) for row in rows)
        / max(len(rows), 1),
        "n_failures": sum(1 for row in rows if not int(row.get("hard") or 0)),
        "n_successes": sum(1 for row in rows if int(row.get("hard") or 0)),
        "sample_size": len(selected),
        "sample_seed": seed,
        "sample_policy": (
            "Random sample. If both successful and failed rows exist, at least one "
            f"of each is included. Failures are softly targeted at {failure_fraction:.2f}."
        ),
        "sample_observations": [compact(idx, row) for idx, row in enumerate(selected, start=1)],
    }


def leakage_warnings(mutated_skill: str, observations: dict) -> list[str]:
    text = mutated_skill.lower()
    warnings: list[str] = []
    for item in observations.get("sample_observations", []):
        item_id = item.get("id")
        if item_id and str(item_id).lower() in text:
            warnings.append(f"sample id leaked: {item_id}")
        for answer in item.get("gold_answers", []) or []:
            answer_s = str(answer).strip()
            if len(answer_s) >= 4 and answer_s.lower() in text:
                warnings.append(f"gold answer may be leaked: {answer_s[:80]}")
    return warnings


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
            print(f"[searchqa-skill-mutate] skip {skill_name}: missing {results_path}", flush=True)
            continue
        runs.append((skill_name, skill_path, results_path, predictions_dir))
    if not runs:
        raise SystemExit(f"No skill/result pairs found under {skills_root} and {eval_root}")
    return runs


def mutate_one(
    *,
    skill_name: str,
    skill_path: Path,
    results_path: Path,
    predictions_dir: Path,
    out_dir: Path,
    model: str,
    base_url: str,
    api_key: str,
    reasoning_effort: str | None,
    sample_size: int,
    sample_seed: int | None,
    failure_fraction: float,
    include_ids: bool,
    include_gold: bool,
    include_questions: bool,
    max_length_ratio: float,
    max_abs_chars: int,
    reject_over_length: bool,
    skip_strong_threshold: float,
    max_trace_chars: int,
    max_response_chars: int,
    max_completion_tokens: int,
    timeout: int,
) -> dict:
    original_skill = skill_path.read_text(encoding="utf-8")
    observations = build_observations(
        results_path=results_path,
        predictions_dir=predictions_dir,
        sample_size=sample_size,
        seed=sample_seed,
        failure_fraction=failure_fraction,
        include_ids=include_ids,
        include_gold=include_gold,
        include_questions=include_questions,
        max_trace_chars=max_trace_chars,
        max_response_chars=max_response_chars,
    )
    original_len = len(original_skill)
    length_limit = min(max_abs_chars, max(1200, int(original_len * max_length_ratio)))
    performance_note = (
        "This skill is already relatively strong. Preserve its core behavior and "
        "make surgical wording edits only."
        if observations["hard_acc"] >= 0.8
        else "This skill has room for improvement. Make targeted edits rather than "
        "broad rewrites."
    )

    if skip_strong_threshold and observations["hard_acc"] >= skip_strong_threshold:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "skill.md").write_text(original_skill.rstrip() + "\n", encoding="utf-8")
        (out_dir / "original_skill.md").write_text(original_skill, encoding="utf-8")
        summary = {
            "skill": skill_name,
            "source_skill": str(skill_path),
            "source_results": str(results_path),
            "out_dir": str(out_dir),
            "model": model,
            "sample_size": observations["sample_size"],
            "sample_seed": sample_seed,
            "failure_fraction": failure_fraction,
            "include_ids": include_ids,
            "include_gold": include_gold,
            "include_questions": include_questions,
            "hard_correct_before": observations["hard_correct"],
            "hard_acc_before": observations["hard_acc"],
            "avg_soft_before": observations["avg_soft"],
            "original_chars": original_len,
            "mutated_chars": original_len,
            "target_max_chars": length_limit,
            "skipped": True,
            "skip_reason": (
                f"hard_acc_before {observations['hard_acc']:.4f} >= "
                f"MUTATION_SKIP_STRONG_THRESHOLD {skip_strong_threshold:.4f}"
            ),
            "rejected": False,
            "warnings": [],
            "usage": {},
        }
        (out_dir / "mutation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "[searchqa-skill-mutate] "
            f"skill={skill_name} skipped strong baseline "
            f"hard={observations['hard_correct']}/{observations['count']}",
            flush=True,
        )
        return summary

    system = "You conservatively edit reusable SKILL.md instructions for LLM agents."
    user = (
        META_MUTATION_SKILL
        + "\n\n## Mutation Budget\n"
        + f"- Original length: {original_len} characters.\n"
        + f"- Target maximum mutated length: {length_limit} characters.\n"
        + f"- Performance note: {performance_note}\n"
        + "- Prefer deleting, replacing, or tightening confusing wording before adding new rules.\n"
        + "- Preserve the original headings and ordering where possible.\n"
        + "- Do not optimize for the sampled rows directly; infer general SearchQA behavior only.\n"
        + "\n\n## Existing SKILL.md\n"
        + original_skill.strip()
        + "\n\n## Rollout Observations\n"
        + json.dumps(observations, ensure_ascii=False, indent=2)
        + "\n\n## Required Output\n"
        + "Return the full mutated SKILL.md. Keep it generic. Do not include concrete sampled questions, IDs, or gold answers. "
        + "Stay within the mutation budget; if uncertain, make fewer changes.\n"
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
        "[searchqa-skill-mutate] "
        f"skill={skill_name} model={model} rows={observations['count']} "
        f"hard={observations['hard_correct']}/{observations['count']} "
        f"samples={observations['sample_size']} seed={sample_seed}",
        flush=True,
    )
    response, usage = chat_optimizer(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=5,
        stage="searchqa_skill_mutation",
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    mutated_skill = strip_code_fence(response)
    (out_dir / "original_skill.md").write_text(original_skill, encoding="utf-8")
    (out_dir / "mutation_prompt.txt").write_text(user, encoding="utf-8")
    (out_dir / "mutation_raw_response.txt").write_text(response, encoding="utf-8")
    warnings = leakage_warnings(mutated_skill, observations)
    length_warning = len(mutated_skill) > length_limit
    if length_warning:
        warnings.append(
            f"mutated skill length {len(mutated_skill)} exceeds target {length_limit}"
        )
    rejected = reject_over_length and length_warning
    if rejected:
        (out_dir / "rejected_skill.md").write_text(mutated_skill + "\n", encoding="utf-8")
        (out_dir / "skill.md").write_text(original_skill.rstrip() + "\n", encoding="utf-8")
    else:
        (out_dir / "skill.md").write_text(mutated_skill + "\n", encoding="utf-8")
    summary = {
        "skill": skill_name,
        "source_skill": str(skill_path),
        "source_results": str(results_path),
        "out_dir": str(out_dir),
        "model": model,
        "sample_size": observations["sample_size"],
        "sample_seed": sample_seed,
        "failure_fraction": failure_fraction,
        "include_ids": include_ids,
        "include_gold": include_gold,
        "include_questions": include_questions,
        "hard_correct_before": observations["hard_correct"],
        "hard_acc_before": observations["hard_acc"],
        "avg_soft_before": observations["avg_soft"],
        "original_chars": original_len,
        "mutated_chars": len(mutated_skill),
        "target_max_chars": length_limit,
        "skipped": False,
        "rejected": rejected,
        "reject_reason": "over_length" if rejected else "",
        "warnings": warnings,
        "usage": usage,
    }
    (out_dir / "mutation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    load_env_files()
    model = os.environ.get("MUTATION_MODEL", "openai/gpt-5.4")
    base_url = os.environ.get("TEACHER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.environ.get("TEACHER_API_KEY", "")
    reasoning_effort = os.environ.get("MUTATION_REASONING_EFFORT", "medium").strip() or None
    sample_size = int(os.environ.get("MUTATION_SAMPLE_SIZE", "8"))
    sample_seed_raw = os.environ.get("MUTATION_SAMPLE_SEED", "").strip()
    sample_seed = int(sample_seed_raw) if sample_seed_raw else None
    failure_fraction = float(os.environ.get("MUTATION_FAILURE_FRACTION", "0.65"))
    include_ids = env_bool("MUTATION_INCLUDE_IDS", False)
    include_gold = env_bool("MUTATION_INCLUDE_GOLD", True)
    include_questions = env_bool("MUTATION_INCLUDE_QUESTIONS", True)
    max_length_ratio = float(os.environ.get("MUTATION_MAX_LENGTH_RATIO", "1.05"))
    max_abs_chars = int(os.environ.get("MUTATION_MAX_ABS_CHARS", "3500"))
    reject_over_length = env_bool("MUTATION_REJECT_OVER_LENGTH", False)
    skip_strong_threshold = float(os.environ.get("MUTATION_SKIP_STRONG_THRESHOLD", "0"))
    max_trace_chars = int(os.environ.get("MAX_TRACE_CHARS", "1800"))
    max_response_chars = int(os.environ.get("MAX_RESPONSE_CHARS", "700"))
    max_completion_tokens = int(os.environ.get("MUTATION_MAX_COMPLETION_TOKENS", "6000"))
    timeout = int(os.environ.get("MUTATION_TIMEOUT", "240"))

    skills_root = Path(
        os.environ.get(
            "SKILLS_ROOT",
            str(ROOT / "results/gpt-5.4-skill/generated_searchqa_sample10_seeds"),
        )
    )
    eval_root = Path(
        os.environ.get(
            "EVAL_ROOT",
            str(
                ROOT
                / "results/gpt-5.4-nano/searchqa/"
                / "searchqa_openrouter_nano_skill_sweep_generated_sample10_seed1to5_train50_parallel50_turn1_reasoninglow"
            ),
        )
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path(
        os.environ.get(
            "OUT_ROOT",
            str(ROOT / "results/gpt-5.4-skill/mutated_searchqa_sample8" / timestamp),
        )
    )

    if not api_key:
        raise SystemExit("Missing OpenRouter API key")

    summaries = []
    for idx, (skill_name, skill_path, results_path, predictions_dir) in enumerate(
        discover_skill_runs(skills_root, eval_root)
    ):
        seed = sample_seed + idx if sample_seed is not None else None
        summaries.append(
            mutate_one(
                skill_name=skill_name,
                skill_path=skill_path,
                results_path=results_path,
                predictions_dir=predictions_dir,
                out_dir=out_root / skill_name,
                model=model,
                base_url=base_url,
                api_key=api_key,
                reasoning_effort=reasoning_effort,
                sample_size=sample_size,
                sample_seed=seed,
                failure_fraction=failure_fraction,
                include_ids=include_ids,
                include_gold=include_gold,
                include_questions=include_questions,
                max_length_ratio=max_length_ratio,
                max_abs_chars=max_abs_chars,
                reject_over_length=reject_over_length,
                skip_strong_threshold=skip_strong_threshold,
                max_trace_chars=max_trace_chars,
                max_response_chars=max_response_chars,
                max_completion_tokens=max_completion_tokens,
                timeout=timeout,
            )
        )

    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "mutation_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[searchqa-skill-mutate] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
