from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _usage_value(usage: Any, *names: str) -> int:
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def normalize_usage(usage: Any) -> dict[str, int]:
    input_tokens = _usage_value(usage, "input_tokens", "prompt_tokens")
    output_tokens = _usage_value(usage, "output_tokens", "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens") or input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def record_token_usage(
    *,
    role: str,
    model: str,
    stage: str,
    usage: Any,
    provider: str = "",
) -> None:
    """Append one successful LLM request to the active COBRAS token ledger."""

    log_path = os.environ.get("COBRAS_TOKEN_LOG_PATH", "").strip()
    if not log_path:
        return
    normalized_role = "teacher" if role in {"teacher", "optimizer", "skill"} else "student"
    normalized = normalize_usage(usage)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "round": os.environ.get("COBRAS_ROUND", "setup"),
        "role": normalized_role,
        "provider": provider
        or os.environ.get(
            "COBRAS_TEACHER_PROVIDER" if normalized_role == "teacher" else "COBRAS_STUDENT_PROVIDER",
            os.environ.get("PROVIDER", ""),
        ),
        "model": model,
        "stage": stage,
        **normalized,
        "pid": os.getpid(),
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _empty_totals() -> dict[str, Any]:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "models": {},
        "stages": {},
    }


def _add(bucket: dict[str, Any], event: dict[str, Any]) -> None:
    calls = 1
    input_tokens = int(event.get("input_tokens") or 0)
    output_tokens = int(event.get("output_tokens") or 0)
    total_tokens = int(event.get("total_tokens") or input_tokens + output_tokens)
    bucket["calls"] += calls
    bucket["input_tokens"] += input_tokens
    bucket["output_tokens"] += output_tokens
    bucket["total_tokens"] += total_tokens

    for group_name, key in (("models", "model"), ("stages", "stage")):
        name = str(event.get(key) or "unknown")
        grouped = bucket[group_name].setdefault(
            name,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        grouped["calls"] += calls
        grouped["input_tokens"] += input_tokens
        grouped["output_tokens"] += output_tokens
        grouped["total_tokens"] += total_tokens


def _add_totals(bucket: dict[str, Any], totals: dict[str, Any], *, model: str = "", stage: str = "") -> None:
    calls = int(totals.get("calls") or 1)
    event = {
        "calls": calls,
        "input_tokens": int(totals.get("input_tokens") or totals.get("prompt_tokens") or 0),
        "output_tokens": int(totals.get("output_tokens") or totals.get("completion_tokens") or 0),
        "total_tokens": int(totals.get("total_tokens") or 0),
        "model": model or totals.get("model") or "unknown",
        "stage": stage or totals.get("stage") or "unknown",
    }
    if not event["total_tokens"]:
        event["total_tokens"] = event["input_tokens"] + event["output_tokens"]
    bucket["calls"] += calls
    bucket["input_tokens"] += event["input_tokens"]
    bucket["output_tokens"] += event["output_tokens"]
    bucket["total_tokens"] += event["total_tokens"]

    for group_name, key in (("models", "model"), ("stages", "stage")):
        name = str(event.get(key) or "unknown")
        grouped = bucket[group_name].setdefault(
            name,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        grouped["calls"] += calls
        grouped["input_tokens"] += event["input_tokens"]
        grouped["output_tokens"] += event["output_tokens"]
        grouped["total_tokens"] += event["total_tokens"]


def load_token_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def summarize_token_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_round: dict[str, dict[str, Any]] = {}
    overall = {"student": _empty_totals(), "teacher": _empty_totals()}
    algorithm = {"student": _empty_totals(), "teacher": _empty_totals()}
    non_algorithm = {"student": _empty_totals(), "teacher": _empty_totals()}
    for event in events:
        round_name = str(event.get("round", "setup"))
        role = "teacher" if event.get("role") == "teacher" else "student"
        round_row = by_round.setdefault(
            round_name,
            {"round": round_name, "student": _empty_totals(), "teacher": _empty_totals()},
        )
        _add(round_row[role], event)
        _add(overall[role], event)
        if round_name.isdigit():
            _add(algorithm[role], event)
        else:
            _add(non_algorithm[role], event)
    return {
        "rounds": by_round,
        "overall": overall,
        "algorithm": algorithm,
        "non_algorithm": non_algorithm,
        "event_count": len(events),
    }


def summarize_init_skill_generation(initial_pool_root: Path | None) -> dict[str, Any]:
    """Sum generation-time teacher tokens stored next to an initial skill pool."""

    payload = {
        "source_root": str(initial_pool_root.resolve()) if initial_pool_root else "",
        "found_metadata_files": 0,
        "missing_metadata": True,
        "teacher": _empty_totals(),
        "files": [],
    }
    if initial_pool_root is None:
        return payload
    root = initial_pool_root.resolve()
    if not root.exists():
        return payload
    seen: set[Path] = set()
    for path in sorted(root.rglob("skill_generation_summary.json")):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        usage = data.get("usage")
        if not isinstance(usage, dict):
            continue
        normalized = normalize_usage(usage)
        model = str(data.get("model") or data.get("skill_generation_model") or "unknown")
        stage = str(data.get("stage") or "init_skill_generation")
        _add_totals(payload["teacher"], normalized, model=model, stage=stage)
        payload["files"].append(
            {
                "path": str(path),
                "model": model,
                "stage": stage,
                **normalized,
            }
        )
    payload["found_metadata_files"] = len(payload["files"])
    payload["missing_metadata"] = len(payload["files"]) == 0
    return payload


def combine_totals(*buckets: dict[str, Any]) -> dict[str, Any]:
    combined = _empty_totals()
    for bucket in buckets:
        for key in ("calls", "input_tokens", "output_tokens", "total_tokens"):
            combined[key] += int(bucket.get(key) or 0)
        for group_name in ("models", "stages"):
            for name, row in (bucket.get(group_name) or {}).items():
                grouped = combined[group_name].setdefault(
                    name,
                    {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                for key in ("calls", "input_tokens", "output_tokens", "total_tokens"):
                    grouped[key] += int(row.get(key) or 0)
    return combined


def _round_sort_key(name: str) -> tuple[int, int | str]:
    if name == "setup":
        return (0, 0)
    try:
        return (1, int(name))
    except ValueError:
        return (2, name)


def write_usage_reports(out_root: Path, initial_pool_root: Path | None = None) -> dict[str, Any]:
    out_root = out_root.resolve()
    usage_root = out_root / "token_usage"
    events_path = usage_root / "events.jsonl"
    summary = summarize_token_events(load_token_events(events_path))
    usage_root.mkdir(parents=True, exist_ok=True)

    ordered_rounds = [
        summary["rounds"][name]
        for name in sorted(summary["rounds"], key=_round_sort_key)
    ]
    init_skill_generation = summarize_init_skill_generation(initial_pool_root)
    teacher_with_init_skill_generation = combine_totals(
        summary["algorithm"]["teacher"],
        init_skill_generation["teacher"],
    )
    rounds_path = usage_root / "rounds.jsonl"
    rounds_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered_rounds),
        encoding="utf-8",
    )
    summary_payload = {
        "event_count": summary["event_count"],
        "student": summary["algorithm"]["student"],
        "teacher": summary["algorithm"]["teacher"],
        "algorithm": summary["algorithm"],
        "non_algorithm": summary["non_algorithm"],
        "runtime_all_events": summary["overall"],
        "init_skill_generation": init_skill_generation,
        "teacher_with_init_skill_generation": teacher_with_init_skill_generation,
        "notes": {
            "student": "Top-level student totals include only numeric COBRAS rounds. init_eval/setup events are excluded from the algorithm total and reported under non_algorithm.",
            "teacher": "Top-level teacher totals include numeric COBRAS rounds. init skill generation is reported separately and combined under teacher_with_init_skill_generation.",
        },
        "rounds_path": str(rounds_path),
        "events_path": str(events_path),
    }
    (usage_root / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for row in ordered_rounds:
        round_name = str(row["round"])
        if round_name == "setup":
            target = usage_root / "setup.json"
        elif round_name.isdigit():
            target = out_root / "rounds" / f"round_{int(round_name):03d}" / "token_usage.json"
        else:
            target = usage_root / f"round_{round_name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_payload
