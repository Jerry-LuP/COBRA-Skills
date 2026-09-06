from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def load_trial_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        if path.name == "summary.json":
            return _load_rows_from_summary(path)
        if path.suffix == ".jsonl":
            return _load_rows_from_jsonl(path)
        if path.name == "result.json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return [payload] if _looks_like_trial_row(payload) else []
        raise FileNotFoundError(f"Unsupported trial input file: {path}")

    summary_path = path / "summary.json"
    if summary_path.exists():
        rows = _load_rows_from_summary(summary_path)
        if rows:
            return rows
    jsonl_path = path / "trial_results.jsonl"
    if jsonl_path.exists():
        rows = _load_rows_from_jsonl(jsonl_path)
        if rows:
            return rows

    rows: list[dict[str, Any]] = []
    for result_path in sorted(path.glob("**/result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if _looks_like_trial_row(payload):
            rows.append(payload)
    if rows:
        return rows
    raise FileNotFoundError(f"Could not find trial-level result rows under {path}")


def _load_rows_from_summary(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and _looks_like_trial_row(row)]


def _load_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in legacy JSONL file {path}")
        return [
            row
            for row in payload
            if isinstance(row, dict) and _looks_like_trial_row(row)
        ]

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if isinstance(payload, dict) and _looks_like_trial_row(payload):
            rows.append(payload)
    return rows


def _looks_like_trial_row(payload: dict[str, Any]) -> bool:
    return "question" in payload and any(
        field in payload for field in ("hard_reward", "soft_reward", "hard", "soft")
    )


def sample_trials(
    *,
    rng: random.Random,
    rows: list[dict[str, Any]],
    sample_size: int,
    min_success: int,
    min_fail: int,
    success_field: str,
    success_threshold: float,
) -> list[dict[str, Any]]:
    fallback_field = success_field.replace("_reward", "")
    successes = [
        row
        for row in rows
        if float(row.get(success_field, row.get(fallback_field, 0.0))) >= success_threshold
    ]
    failures = [
        row
        for row in rows
        if float(row.get(success_field, row.get(fallback_field, 0.0))) < success_threshold
    ]
    if len(successes) < min_success:
        raise ValueError(f"Not enough successful trials: need {min_success}, found {len(successes)}")
    if len(failures) < min_fail:
        raise ValueError(f"Not enough failed trials: need {min_fail}, found {len(failures)}")
    if sample_size < min_success + min_fail:
        raise ValueError("sample_size must be >= min_success + min_fail")

    picked = rng.sample(successes, min_success) + rng.sample(failures, min_fail)
    remaining = [row for row in rows if row not in picked]
    extra = sample_size - len(picked)
    if extra > len(remaining):
        raise ValueError(f"Not enough remaining trials to fill sample_size={sample_size}")
    picked.extend(rng.sample(remaining, extra))
    rng.shuffle(picked)
    return picked
