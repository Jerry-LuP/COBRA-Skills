from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cobras.cobras_core.model import (
    chat_optimizer,
    configure_azure_openai,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_reasoning_effort,
)


ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = ROOT

META_MUTATION_SKILL = """You are an expert SocialMaze Hidden Role skill editor.

You will receive:
1. An existing SocialMaze Hidden Role SKILL.md.
2. A small random sample of rollout observations produced by a weaker target model using that skill.

Your job is to make a conservative mutation of the existing skill: preserve its overall strategy and useful wording, but refine unclear instructions, delete low-value wording, and improve hidden-role deduction behavior.

Important constraints:
- Do not rewrite from scratch.
- Make at most 2 to 4 conceptual edits.
- Prefer replacing or deleting wording over adding wording.
- Add at most 2 new bullets total.
- Do not add new top-level sections unless the existing skill is missing an answer-format or checking section.
- If the existing skill is already strong, prefer surgical clarifications over new rules.
- Do not add concrete training examples, exact task IDs, source indexes, exact questions, exact player-number answers, exact gold roles, or memorized answers.
- Do not rely on role-frequency priors from the dataset.
- Keep the skill generic and reusable across SocialMaze Hidden Role Deduction.
- Prefer small, high-signal edits over prompt bloat.
- Keep the mutated skill near the original length when possible.
- Preserve concise Markdown SKILL.md style.

Use the observations to infer what should change, but do not optimize for the sampled rows directly.

Output only the full mutated SKILL.md content. Do not wrap it in code fences. Do not explain your changes.
"""


def load_env_files() -> None:
    for env_path in [
        PROJECT_ROOT / ".env.socialmaze.local",
        PROJECT_ROOT / ".env",
        ROOT / ".env.socialmaze.local",
        ROOT / ".env",
    ]:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("export "):
                s = s[len("export ") :].strip()
            if "=" not in s:
                continue
            key, value = s.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def clip(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", text).strip()


def safe_name(value: object) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "item"


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def sample_rows(rows: list[dict[str, Any]], *, sample_size: int, seed: int | None, sample_policy: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sample_size = max(0, min(sample_size, len(rows)))
    if sample_policy == "random":
        return rng.sample(rows, sample_size)

    successes = [row for row in rows if int(row.get("hard") or 0)]
    failures = [row for row in rows if not int(row.get("hard") or 0)]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_random(pool: list[dict[str, Any]]) -> None:
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
    selected.extend(remaining[: max(0, sample_size - len(selected))])
    rng.shuffle(selected)
    return selected[:sample_size]


def read_prediction_files(run_root: Path, row_id: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    pred_dir = run_root / "predictions" / safe_name(row_id)
    result = load_json(pred_dir / "result.json") if (pred_dir / "result.json").exists() else {}
    reward = load_json(pred_dir / "reward.json") if (pred_dir / "reward.json").exists() else {}
    conversation = load_json(pred_dir / "conversation.json") if (pred_dir / "conversation.json").exists() else []
    return result, reward, conversation


def compact_conversation(conversation: list[dict[str, Any]], max_user_chars: int, max_assistant_chars: int) -> dict[str, str]:
    system = ""
    user = ""
    assistant = ""
    for msg in conversation:
        role = msg.get("role")
        content = str(msg.get("content") or "")
        if role == "system" and not system:
            system = content
        elif role == "user" and not user:
            user = content
        elif role == "assistant":
            assistant = content
    return {
        "system_excerpt": clip(system, 1000),
        "user_excerpt": clip(user, max_user_chars),
        "assistant_answer": clip(assistant, max_assistant_chars),
    }


def compact_row(
    row: dict[str, Any],
    *,
    run_root: Path,
    include_conversation: bool,
    max_user_chars: int,
    max_assistant_chars: int,
) -> dict[str, Any]:
    result, reward, conversation = read_prediction_files(run_root, str(row.get("id")))
    gold = result.get("gold_parsed") or row.get("gold_parsed") or {}
    pred = result.get("predicted_parsed") or row.get("predicted_parsed") or {}
    item = {
        "sample_label": "",
        "outcome": "success" if int(row.get("hard") or 0) else "failure",
        "hard": int(row.get("hard") or 0),
        "soft": float(row.get("soft") or 0.0),
        "gold": gold,
        "prediction": pred,
        "fail_reason": result.get("fail_reason") or row.get("fail_reason") or "",
        "reward": {
            "hard": reward.get("hard"),
            "soft": reward.get("soft"),
            "gold": reward.get("gold"),
            "prediction": reward.get("prediction"),
        },
    }
    if include_conversation:
        item["trajectory"] = compact_conversation(
            conversation,
            max_user_chars=max_user_chars,
            max_assistant_chars=max_assistant_chars,
        )
    return item


def aggregate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    role_counts = Counter()
    pred_role_counts = Counter()
    hard_by_role: dict[str, Counter] = {}
    criminal_correct = 0
    role_correct = 0
    empty = 0
    for row in rows:
        gold = row.get("gold_parsed") or {}
        pred = row.get("predicted_parsed") or {}
        gold_role = str(gold.get("role") or "")
        pred_role = str(pred.get("role") or "")
        role_counts[gold_role] += 1
        pred_role_counts[pred_role] += 1
        hard_by_role.setdefault(gold_role, Counter())
        hard_by_role[gold_role]["total"] += 1
        hard_by_role[gold_role]["hard"] += int(row.get("hard") or 0)
        criminal_correct += int(pred.get("criminal_player") == gold.get("criminal_player"))
        role_correct += int(pred.get("role") == gold.get("role"))
        empty += int(not row.get("predicted_answer"))
    return {
        "n": len(rows),
        "hard": sum(int(row.get("hard") or 0) for row in rows),
        "hard_acc": sum(int(row.get("hard") or 0) for row in rows) / max(len(rows), 1),
        "avg_soft": sum(float(row.get("soft") or 0.0) for row in rows) / max(len(rows), 1),
        "criminal_correct": criminal_correct,
        "role_correct": role_correct,
        "empty_outputs": empty,
        "gold_role_counts": dict(role_counts),
        "predicted_role_counts": dict(pred_role_counts),
        "hard_by_gold_role": {
            role: {"hard": counts["hard"], "total": counts["total"]}
            for role, counts in hard_by_role.items()
        },
    }


def build_observations(
    *,
    run_root: Path,
    sample_size: int,
    sample_seed: int | None,
    sample_policy: str,
    include_conversation: bool,
    max_user_chars: int,
    max_assistant_chars: int,
) -> dict[str, Any]:
    results_path = run_root / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.jsonl under RUN_ROOT: {run_root}")
    rows = load_jsonl(results_path)
    selected = sample_rows(rows, sample_size=sample_size, seed=sample_seed, sample_policy=sample_policy)
    samples = []
    for idx, row in enumerate(selected, start=1):
        item = compact_row(
            row,
            run_root=run_root,
            include_conversation=include_conversation,
            max_user_chars=max_user_chars,
            max_assistant_chars=max_assistant_chars,
        )
        item["sample_label"] = f"sample_{idx:02d}"
        samples.append(item)
    return {
        "source_run_root": str(run_root),
        "source_results": str(results_path),
        "aggregate_stats": aggregate_stats(rows),
        "sample_size": len(selected),
        "sample_seed": sample_seed,
        "sample_policy": sample_policy,
        "sample_observations": samples,
    }


def leakage_warnings(mutated_skill: str, observations: dict[str, Any]) -> list[str]:
    text = mutated_skill.lower()
    warnings: list[str] = []
    for item in observations.get("sample_observations", []):
        for key in ("id", "source_index"):
            value = item.get(key)
            if value and str(value).lower() in text:
                warnings.append(f"sample metadata leaked: {value}")
    return warnings


def main() -> None:
    load_env_files()
    model = os.environ.get("SKILL_MUTATE_MODEL", os.environ.get("SKILL_GEN_MODEL", "openai/gpt-5.4"))
    base_url = os.environ.get("TEACHER_BASE_URL", os.environ.get("SKILL_MUTATE_BASE_URL", "https://openrouter.ai/api/v1"))
    api_key = (
        os.environ.get("TEACHER_API_KEY")
        or os.environ.get("SKILL_MUTATE_API_KEY")
        or os.environ.get("SKILL_GEN_API_KEY")

    )
    reasoning_effort = os.environ.get("SKILL_MUTATE_REASONING_EFFORT", "medium").strip() or None
    max_completion_tokens = int(os.environ.get("SKILL_MUTATE_MAX_COMPLETION_TOKENS", "6000"))
    timeout = int(os.environ.get("SKILL_MUTATE_TIMEOUT", "240"))

    parent_skill_path = Path(os.environ["PARENT_SKILL_PATH"])
    run_root = Path(os.environ["RUN_ROOT"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        os.environ.get(
            "OUT_DIR",
            str(PROJECT_ROOT / "results/skills/socialmaze_hard" / f"mutated_hidden_role_{timestamp}"),
        )
    )
    sample_size = int(os.environ.get("SAMPLE_SIZE", "8"))
    sample_seed_raw = os.environ.get("SAMPLE_SEED", "").strip()
    sample_seed = int(sample_seed_raw) if sample_seed_raw else None
    sample_policy = os.environ.get("SAMPLE_POLICY", "random").strip().lower() or "random"
    include_conversation = env_bool("INCLUDE_CONVERSATION", True)
    max_user_chars = int(os.environ.get("MAX_USER_CHARS", "3000"))
    max_assistant_chars = int(os.environ.get("MAX_ASSISTANT_CHARS", "900"))
    max_length_ratio = float(os.environ.get("MUTATION_MAX_LENGTH_RATIO", "1.08"))
    max_abs_chars = int(os.environ.get("MUTATION_MAX_ABS_CHARS", "4200"))
    reject_over_length = env_bool("MUTATION_REJECT_OVER_LENGTH", False)

    if not api_key:
        raise SystemExit("Missing TEACHER_API_KEY")
    if not parent_skill_path.exists():
        raise SystemExit(f"PARENT_SKILL_PATH not found: {parent_skill_path}")
    if not run_root.exists():
        raise SystemExit(f"RUN_ROOT not found: {run_root}")

    parent_skill = parent_skill_path.read_text(encoding="utf-8")
    observations = build_observations(
        run_root=run_root,
        sample_size=sample_size,
        sample_seed=sample_seed,
        sample_policy=sample_policy,
        include_conversation=include_conversation,
        max_user_chars=max_user_chars,
        max_assistant_chars=max_assistant_chars,
    )
    original_len = len(parent_skill)
    length_limit = min(max_abs_chars, max(1200, int(original_len * max_length_ratio)))
    hard_acc = observations["aggregate_stats"]["hard_acc"]
    performance_note = (
        "This skill is already strong. Preserve its core behavior and make surgical wording edits only."
        if hard_acc >= 0.8
        else "This skill has room for improvement. Make targeted edits rather than broad rewrites."
    )

    system = "You conservatively edit reusable SKILL.md instructions for LLM agents."
    user = (
        META_MUTATION_SKILL
        + "\n\n## Mutation Budget\n"
        + f"- Original length: {original_len} characters.\n"
        + f"- Target maximum mutated length: {length_limit} characters.\n"
        + f"- Performance note: {performance_note}\n"
        + "- Prefer deleting, replacing, or tightening confusing wording before adding new rules.\n"
        + "- Preserve the original headings and ordering where possible.\n"
        + "- Do not optimize for the sampled rows directly; infer general Hidden Role behavior only.\n"
        + "\n\n## Existing SKILL.md\n"
        + parent_skill.strip()
        + "\n\n## Rollout Observations Using This Skill\n"
        + json.dumps(observations, ensure_ascii=False, indent=2)
        + "\n\n## Required Output\n"
        + "Return the full mutated SocialMaze Hidden Role SKILL.md. Keep it generic. Do not include concrete sampled questions, IDs, source indexes, exact player-number answers, or gold roles. Stay within the mutation budget; if uncertain, make fewer changes.\n"
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
        "[socialmaze-hidden-role-skill-mutate] "
        f"model={model} parent={parent_skill_path} run_root={run_root} "
        f"hard={observations['aggregate_stats']['hard']}/{observations['aggregate_stats']['n']} "
        f"samples={observations['sample_size']} seed={sample_seed} out={out_dir}",
        flush=True,
    )
    response, usage = chat_optimizer(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=5,
        stage="socialmaze_hidden_role_skill_mutation",
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    mutated = strip_code_fence(response)
    warnings = leakage_warnings(mutated, observations)
    length_warning = len(mutated) > length_limit
    if length_warning:
        warnings.append(f"mutated skill length {len(mutated)} exceeds target {length_limit}")
    rejected = reject_over_length and length_warning

    (out_dir / "original_skill.md").write_text(parent_skill, encoding="utf-8")
    (out_dir / "mutation_prompt.txt").write_text(user, encoding="utf-8")
    (out_dir / "mutation_raw_response.txt").write_text(response, encoding="utf-8")
    if rejected:
        (out_dir / "rejected_skill.md").write_text(mutated + "\n", encoding="utf-8")
        (out_dir / "skill.md").write_text(parent_skill.rstrip() + "\n", encoding="utf-8")
    else:
        (out_dir / "skill.md").write_text(mutated + "\n", encoding="utf-8")
    summary = {
        "model": model,
        "base_url": base_url,
        "parent_skill_path": str(parent_skill_path),
        "run_root": str(run_root),
        "out_dir": str(out_dir),
        "sample_size": observations["sample_size"],
        "sample_seed": sample_seed,
        "sample_policy": sample_policy,
        "hard_correct_before": observations["aggregate_stats"]["hard"],
        "hard_acc_before": observations["aggregate_stats"]["hard_acc"],
        "avg_soft_before": observations["aggregate_stats"]["avg_soft"],
        "original_chars": original_len,
        "mutated_chars": len(mutated),
        "target_max_chars": length_limit,
        "rejected": rejected,
        "reject_reason": "over_length" if rejected else "",
        "warnings": warnings,
        "usage": usage,
    }
    (out_dir / "mutation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "observations.json").write_text(
        json.dumps(observations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[socialmaze-hidden-role-skill-mutate] skill={out_dir / 'skill.md'}", flush=True)
    print(f"[socialmaze-hidden-role-skill-mutate] usage={usage}", flush=True)


if __name__ == "__main__":
    main()
