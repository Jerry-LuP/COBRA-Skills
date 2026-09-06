from __future__ import annotations

from collections import Counter
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from cobras.cobras_core.model import (
    configure_azure_openai,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_reasoning_effort,
)
from cobras.task_envs.eval_runtime import safe_name


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    if not rows:
        raise ValueError(f"No result rows found in {path}")
    return rows


def clip(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    if limit < 80:
        return text[:limit]
    half = (limit - 35) // 2
    return text[:half] + "\n...[middle truncated]...\n" + text[-half:]


def strip_code_fence(text: str) -> str:
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", text).strip()


def ensure_frontmatter(text: str, *, name: str, description: str) -> str:
    text = strip_code_fence(text)
    if re.match(r"^---\s*\n.*?\n---(?:\s*\n|$)", text, flags=re.DOTALL):
        return text.rstrip() + "\n"
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {json.dumps(description, ensure_ascii=False)}\n"
        "---\n\n"
        + text.rstrip()
        + "\n"
    )


def _is_success(row: dict[str, Any]) -> bool:
    return float(row.get("hard", row.get("hard_reward", 0.0)) or 0.0) >= 1.0


def _task_type(row: dict[str, Any]) -> str:
    task_type = str(row.get("task_type") or row.get("instruction_type") or "").strip()
    if task_type:
        return task_type
    gamefile = str(row.get("gamefile") or row.get("id") or "").strip()
    return gamefile.split("-", 1)[0] if "-" in gamefile else "unknown"


def select_rows(rows: list[dict[str, Any]], *, sample_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    count = min(max(sample_size, 0), len(rows))
    if count <= 0:
        return []

    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(_task_type(row), []).append(row)

    # ALFWorld failures are highly task-structure dependent. Start with broad
    # type coverage, then fill the remaining budget with failure-heavy evidence.
    picked: list[dict[str, Any]] = []
    task_types = sorted(by_type)
    rng.shuffle(task_types)
    for task_type in task_types:
        if len(picked) >= count:
            break
        bucket = by_type[task_type]
        failures = [row for row in bucket if not _is_success(row)]
        candidates = failures or bucket
        picked.append(rng.choice(candidates))

    remaining = [row for row in rows if row not in picked]
    failure_remaining = [row for row in remaining if not _is_success(row)]
    target_failures = min(len([row for row in rows if not _is_success(row)]), max(count // 2, count - 3))
    while len(picked) < count and len([row for row in picked if not _is_success(row)]) < target_failures and failure_remaining:
        row = rng.choice(failure_remaining)
        picked.append(row)
        failure_remaining.remove(row)
        remaining.remove(row)

    fill = [row for row in remaining if row not in picked]
    if len(picked) < count:
        picked.extend(rng.sample(fill, count - len(picked)))
    rng.shuffle(picked)
    return picked


def _conversation_path(run_root: Path, row: dict[str, Any]) -> Path:
    artifacts = row.get("artifact_paths") or {}
    raw = str(artifacts.get("conversation_json") or "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else run_root / path
    return run_root / "predictions" / safe_name(row.get("id")) / "conversation.json"


def render_trace(
    run_root: Path,
    row: dict[str, Any],
    *,
    max_trace_chars: int = 24000,
    max_reasoning_chars: int = 900,
    max_feedback_chars: int = 1800,
) -> str:
    path = _conversation_path(run_root, row)
    if not path.exists():
        return "<conversation unavailable>"
    try:
        events = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"<conversation unreadable: {exc}>"
    rendered: list[str] = []
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict):
            continue
        if event.get("type") == "task":
            rendered.append(
                "TASK\n"
                f"description: {clip(event.get('task_description'), 1200)}\n"
                f"initial_observation: {clip(event.get('initial_observation'), max_feedback_chars)}"
            )
            continue
        rendered.append(
            f"STEP {event.get('step')}\n"
            f"reasoning: {clip(event.get('reasoning'), max_reasoning_chars)}\n"
            f"action: {clip(event.get('action'), 300)}\n"
            f"environment: {clip(event.get('env_feedback'), max_feedback_chars)}\n"
            f"reward: {event.get('reward')} done: {event.get('done')}"
        )
    return clip("\n\n".join(rendered), max_trace_chars)


def render_failure_tail(run_root: Path, row: dict[str, Any], *, max_steps: int = 8) -> list[dict[str, Any]]:
    path = _conversation_path(run_root, row)
    if not path.exists():
        return []
    try:
        events = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    steps = [event for event in events if isinstance(event, dict) and event.get("type") != "task"]
    tail: list[dict[str, Any]] = []
    for event in steps[-max_steps:]:
        tail.append(
            {
                "step": event.get("step"),
                "action": clip(event.get("action"), 180),
                "environment": clip(event.get("env_feedback"), 360),
                "done": event.get("done"),
            }
        )
    return tail


def build_observations(
    *,
    run_root: Path,
    results_path: Path,
    sample_size: int,
    sample_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = load_jsonl(results_path)
    selected = select_rows(rows, sample_size=sample_size, seed=sample_seed)
    task_type_counts = Counter(_task_type(row) for row in rows)
    task_type_correct = Counter(_task_type(row) for row in rows if _is_success(row))
    task_type_summary = [
        {
            "task_type": task_type,
            "count": count,
            "hard_correct": task_type_correct[task_type],
            "hard_acc": task_type_correct[task_type] / max(count, 1),
        }
        for task_type, count in sorted(task_type_counts.items())
    ]
    samples = []
    for index, row in enumerate(selected, start=1):
        hard = int(_is_success(row))
        samples.append(
            {
                "sample_label": f"sample_{index:02d}",
                "outcome": "success" if hard else "failure",
                "task_type": row.get("task_type"),
                "task_description": clip(row.get("task_description"), 1200),
                "hard": hard,
                "soft": float(row.get("soft", row.get("soft_reward", 0.0)) or 0.0),
                "n_turns": row.get("n_turns"),
                "fail_reason": clip(row.get("fail_reason"), 500),
                "failure_tail": [] if hard else render_failure_tail(run_root, row),
                "trajectory": render_trace(run_root, row),
            }
        )
    hard_total = sum(int(_is_success(row)) for row in rows)
    return (
        {
            "count": len(rows),
            "hard_correct": hard_total,
            "hard_acc": hard_total / max(len(rows), 1),
            "sample_size": len(samples),
            "sample_seed": sample_seed,
            "sample_policy": (
                "Task-type-balanced sample. Prefer at least one example from each observed "
                "ALFWorld task type, then fill remaining slots with failure-heavy evidence."
            ),
            "observed_task_type_summary": task_type_summary,
            "sample_task_types": sorted({_task_type(row) for row in selected}),
            "sample_observations": samples,
        },
        selected,
    )


def leakage_warnings(skill: str, sampled_rows: list[dict[str, Any]]) -> list[str]:
    lowered = skill.casefold()
    warnings: list[str] = []
    for row in sampled_rows:
        item_id = str(row.get("id") or "").strip()
        gamefile = str(row.get("gamefile") or "").strip()
        description = str(row.get("task_description") or "").strip()
        if len(item_id) >= 4 and item_id.casefold() in lowered:
            warnings.append(f"sample id: {item_id}")
        if gamefile and gamefile.casefold() in lowered:
            warnings.append("sample game path")
        if len(description) >= 32 and description.casefold() in lowered:
            warnings.append("exact sampled task description")
    return sorted(set(warnings))


COMMON_TASK_TOKENS = {
    "and",
    "at",
    "in",
    "obj",
    "object",
    "place",
    "pick",
    "recep",
    "receptacle",
    "simple",
    "then",
}


def _task_tokens(task_type: str) -> list[str]:
    tokens = [
        token
        for token in re.split(r"[^a-zA-Z]+", task_type.casefold())
        if len(token) >= 3 and token not in COMMON_TASK_TOKENS
    ]
    return sorted(set(tokens))


def skill_quality_warnings(skill: str, observations: dict[str, Any]) -> list[str]:
    lowered = skill.casefold()
    warnings: list[str] = []
    body_words = re.findall(r"[a-zA-Z][a-zA-Z'-]+", re.sub(r"^---.*?---", "", skill, flags=re.DOTALL))
    if len(body_words) < 700:
        warnings.append(f"skill is short for multi-structure ALFWorld guidance: {len(body_words)} body words")
    if "<action>" not in lowered or "</action>" not in lowered:
        warnings.append("skill does not explicitly preserve <action>...</action> output contract")
    for phrase in ("source-qualified", "visible contents", "nothing happens", "final receptacle"):
        if phrase not in lowered:
            warnings.append(f"skill may miss action-level recovery phrase: {phrase!r}")

    for row in observations.get("observed_task_type_summary", []):
        if not isinstance(row, dict):
            continue
        count = int(row.get("count") or 0)
        hard_acc = float(row.get("hard_acc") or 0.0)
        task_type = str(row.get("task_type") or "")
        if count <= 0 or hard_acc >= 0.5:
            continue
        tokens = _task_tokens(task_type)
        if tokens and not any(token in lowered for token in tokens):
            warnings.append(
                f"weak observed task_type {task_type!r} has hard_acc={hard_acc:.3f}, "
                f"but none of its structural tokens appear in the skill: {tokens}"
            )
    return warnings


def configure_optimizer(model: str, reasoning_effort: str | None, base_url: str, api_key: str) -> None:
    if not api_key:
        raise SystemExit("Missing teacher API key")
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


def teacher_settings(prefix: str) -> tuple[str, str, str, str | None, int, int]:
    model = os.environ.get(f"{prefix}_MODEL", "openai/gpt-5.5")
    base_url = (
        os.environ.get(f"{prefix}_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENROUTER_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )
    api_key = (
        os.environ.get(f"{prefix}_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("YUNWU_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    effort = os.environ.get(f"{prefix}_REASONING_EFFORT", "medium").strip() or None
    max_tokens = int(os.environ.get(f"{prefix}_MAX_COMPLETION_TOKENS", "6000"))
    timeout = int(os.environ.get(f"{prefix}_TIMEOUT", "300"))
    return model, base_url, api_key, effort, max_tokens, timeout


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
