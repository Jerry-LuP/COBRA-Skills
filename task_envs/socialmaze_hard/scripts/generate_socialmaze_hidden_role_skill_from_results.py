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

META_SKILL = """You are an expert skill writer for SocialMaze Hidden Role Deduction.

Your job is to write a reusable SKILL.md for a weaker SocialMaze target model. The target model receives a hidden-role social deduction question and must produce the required final answer.
The task is to identify the Criminal player and Player 1's true role.
Write the skill as direct, unambiguous instructions that the target model can apply reliably.
Give explicit, actionable guidance for both required predictions; do not leave either one implicit.
When the observations reveal a recurring task-specific distinction, state it as a general rule rather than only giving high-level advice.
The target model uses no extra reasoning mode, so make the skill a compact, deterministic checklist whose decisions can be mechanically applied to the transcript.
Avoid vague decision rules such as "most likely", "cleanest fit", or "fewest assumptions" unless you define exactly how the target model should compute them.

Most observed assistant answers are failed attempts. Treat them as symptoms to correct, not demonstrations to imitate. Derive reusable rules from the game description and the differences between predictions and gold/reward feedback.

Use the observed trajectories and rewards to infer general behavioral rules. The observations may include gold/evaluation feedback. Use that feedback only to identify reusable patterns. Do not memorize, quote, or leak task IDs, exact questions, exact player-number answers, exact gold roles, or concrete training examples.

Important anti-leakage constraints:
- Do not memorize, quote, or leak exact task IDs, source indexes, player-number answers, or concrete training examples.
- Do not rely on the public dataset's answer distribution or role prior. The evaluation may be balanced so Player 1's true role can be Investigator, Criminal, Rumormonger, or Lunatic at substantial rates.
- Do not say that Rumormonger or Lunatic is usually more likely because of the dataset. Only use game rules and consistency reasoning.
- The skill must remain valid if role distributions are changed.

The generated skill must:
- be generic across Hidden Role Deduction instances,
- be useful when appended to the target model's system prompt,
- avoid dataset-specific IDs, exact questions, exact player-number answers, exact gold roles, or concrete training examples,
- clearly identify itself as reusable reasoning guidance, not task data,
- use concise Markdown structure suitable for a SKILL.md file,
- be concise enough for a long system prompt,
- preserve enough detail to change model behavior.

Do not assume any predefined list of failure modes. Infer what matters from the observations. Prefer broadly applicable operational guidance over dataset-specific rules or examples.

Be willing to choose a distinctive strategy if the observations support it. The best skill may focus on hypothesis enumeration, truth-table construction, role-count consistency, answer-format discipline, uncertainty handling, self-role resolution, statement parsing, or another reusable behavior you infer from the data.

Output only the final skill.md content. Do not wrap it in code fences.
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


def find_default_run_root() -> Path:
    root = PROJECT_ROOT / "results/cobras/socialmaze_hard"
    preferred = root / "socialmaze_hidden_role_balanced_train50_openrouter_nano_baseline_parallel15_reasoninglow"
    if preferred.exists():
        return preferred
    candidates = []
    if root.exists():
        for path in root.iterdir():
            if not path.is_dir():
                continue
            name = path.name.lower()
            if "hidden_role" in name and "baseline" in name and (path / "results.jsonl").exists():
                candidates.append(path)
    if not candidates:
        return preferred
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def select_rows(rows: list[dict[str, Any]], sample_size: int, sample_seed: int | None) -> list[dict[str, Any]]:
    rng = random.Random(sample_seed)
    sample_size = max(0, min(sample_size, len(rows)))
    return rng.sample(rows, sample_size)


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
        "system_excerpt": clip(system, 1200),
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
    compact = {
        "id": row.get("id"),
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
        compact["trajectory"] = compact_conversation(
            conversation,
            max_user_chars=max_user_chars,
            max_assistant_chars=max_assistant_chars,
        )
    return compact


def aggregate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    role_counts = Counter()
    pred_role_counts = Counter()
    hard_by_role: dict[str, Counter] = {}
    criminal_correct = 0
    role_correct = 0
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
    return {
        "n": len(rows),
        "hard": sum(int(row.get("hard") or 0) for row in rows),
        "avg_soft": sum(float(row.get("soft") or 0.0) for row in rows) / max(len(rows), 1),
        "criminal_correct": criminal_correct,
        "role_correct": role_correct,
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
    results_path: Path,
    sample_size: int,
    sample_seed: int | None,
    include_conversation: bool,
    max_user_chars: int,
    max_assistant_chars: int,
) -> dict[str, Any]:
    rows = load_jsonl(results_path)
    selected = select_rows(rows, sample_size, sample_seed)
    summary_path = run_root / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    return {
        "source_run_root": str(run_root),
        "source_results": str(results_path),
        "summary": summary,
        "aggregate_stats": aggregate_stats(rows),
        "sample_size": len(selected),
        "sample_seed": sample_seed,
        "sample_policy": (
            "Uniform random sample from all result rows, without forcing a success/failure ratio. Different seeds should "
            "produce different sample combinations. Use samples for learning generic behavior only, not for memorizing "
            "exact IDs, answers, or distribution priors."
        ),
        "sample_observations": [
            compact_row(
                row,
                run_root=run_root,
                include_conversation=include_conversation,
                max_user_chars=max_user_chars,
                max_assistant_chars=max_assistant_chars,
            )
            for row in selected
        ],
    }


def main() -> None:
    load_env_files()
    model = os.environ.get("SKILL_GEN_MODEL", "openai/gpt-5.4")
    base_url = (
        os.environ.get("SKILL_GEN_BASE_URL")
        or os.environ.get("TEACHER_BASE_URL")

        or "https://openrouter.ai/api/v1"
    )
    api_key = (
        os.environ.get("SKILL_GEN_API_KEY")
        or os.environ.get("TEACHER_API_KEY")

    )
    reasoning_effort = os.environ.get("SKILL_GEN_REASONING_EFFORT", "medium").strip() or None
    max_completion_tokens = int(os.environ.get("SKILL_GEN_MAX_COMPLETION_TOKENS", "7000"))
    timeout = int(os.environ.get("SKILL_GEN_TIMEOUT", "300"))

    run_root = Path(os.environ.get("RUN_ROOT", str(find_default_run_root())))
    results_path = Path(os.environ.get("RESULTS_PATH", str(run_root / "results.jsonl")))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        os.environ.get(
            "OUT_DIR",
            str(PROJECT_ROOT / "results/skills/socialmaze_hard" / f"generated_hidden_role_{timestamp}"),
        )
    )
    sample_size = int(os.environ.get("SAMPLE_SIZE", "8"))
    sample_seed_raw = os.environ.get("SAMPLE_SEED", "").strip()
    sample_seed = int(sample_seed_raw) if sample_seed_raw else 20260713
    include_conversation = os.environ.get("INCLUDE_CONVERSATION", "1").strip().lower() in {"1", "true", "yes"}
    max_user_chars = int(os.environ.get("MAX_USER_CHARS", "3600"))
    max_assistant_chars = int(os.environ.get("MAX_ASSISTANT_CHARS", "1200"))
    reference_skill_path_raw = os.environ.get("REFERENCE_SKILL_PATH", "").strip()
    reference_skill_path = Path(reference_skill_path_raw) if reference_skill_path_raw else None

    if not api_key:
        raise SystemExit("Missing TEACHER_API_KEY")
    if not run_root.exists():
        raise SystemExit(f"RUN_ROOT not found: {run_root}")
    if not results_path.exists():
        raise SystemExit(f"RESULTS_PATH not found: {results_path}")

    observations = build_observations(
        run_root=run_root,
        results_path=results_path,
        sample_size=sample_size,
        sample_seed=sample_seed,
        include_conversation=include_conversation,
        max_user_chars=max_user_chars,
        max_assistant_chars=max_assistant_chars,
    )

    reference_skill = ""
    if reference_skill_path is not None and reference_skill_path.is_file():
        reference_skill = reference_skill_path.read_text(encoding="utf-8")

    system = "You write high-quality, generalizable SKILL.md instructions for LLM agents."
    user = (
        META_SKILL
        + "\n\n## Observed SocialMaze Hidden Role Results\n"
        + json.dumps(observations, ensure_ascii=False, indent=2)
    )
    if reference_skill:
        user += (
            "\n\n## Optional Reference Skill\n"
            "This reference may contain useful style or ideas, but do not copy any distribution-prior advice. "
            "Prefer robust consistency reasoning over role-frequency heuristics.\n\n"
            + reference_skill.strip()
        )
    user += (
        "\n\n## Required Output\n"
        "Write one SocialMaze Hidden Role Deduction SKILL.md in concise Markdown. "
        "Use short headings and bullets where helpful. "
        "It must not include exact task IDs, source indexes, exact sample answers, or advice based on role frequency in the training data.\n"
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
        "[socialmaze-hidden-role-skill-gen] "
        f"model={model} base_url={base_url} results={results_path} "
        f"rows={observations['aggregate_stats']['n']} hard={observations['aggregate_stats']['hard']} "
        f"samples={observations['sample_size']} seed={observations['sample_seed']} out={out_dir}",
        flush=True,
    )

    response, usage = chat_optimizer(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=5,
        stage="socialmaze_hidden_role_skill_generation",
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
                "run_root": str(run_root),
                "results_path": str(results_path),
                "out_dir": str(out_dir),
                "sample_size": observations["sample_size"],
                "sample_seed": observations["sample_seed"],
                "aggregate_stats": observations["aggregate_stats"],
                "usage": usage,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[socialmaze-hidden-role-skill-gen] skill={out_dir / 'skill.md'}", flush=True)
    print(f"[socialmaze-hidden-role-skill-gen] usage={usage}", flush=True)


if __name__ == "__main__":
    main()
