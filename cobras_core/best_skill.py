from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_history(history_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(history_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {history_path}:{line_number}") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def selected_reward(row: dict[str, Any], reward_field: str) -> float:
    selected_eval = row.get("selected_eval", {}) or {}
    if reward_field == "soft":
        value = selected_eval.get("avg_soft_reward", selected_eval.get("soft_reward"))
    else:
        value = selected_eval.get("avg_hard_reward", selected_eval.get("hard_reward"))
    if value is None:
        value = selected_eval.get("reward", 0.0)
    return float(value or 0.0)


def aggregate_selected_skills(
    history: list[dict[str, Any]],
    reward_field: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    metadata: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(history):
        selected_arm = row.get("selected_arm", {}) or {}
        arm_id = str(selected_arm.get("arm_id") or row.get("arm_id") or "").strip()
        skill_root = str(selected_arm.get("skill_root") or row.get("skill_root") or "").strip()
        if not arm_id or not skill_root:
            continue
        round_index = int(row.get("round", index))
        iteration = int(row.get("iteration", round_index + 1))
        grouped[arm_id].append((round_index, iteration, selected_reward(row, reward_field)))
        metadata[arm_id] = {
            "arm_id": arm_id,
            "skill_name": str(selected_arm.get("skill_name") or Path(skill_root).name),
            "skill_root": skill_root,
            "workspace_dir": str(selected_arm.get("workspace_dir") or skill_root),
            "origin": str(selected_arm.get("origin") or "unknown"),
        }

    ranking: list[dict[str, Any]] = []
    for arm_id, observations in grouped.items():
        rewards = [reward for _, _, reward in observations]
        mean_reward = sum(rewards) / len(rewards)
        ranking.append(
            {
                **metadata[arm_id],
                "reward_field": reward_field,
                "mean_reward": mean_reward,
                "selection_count": len(rewards),
                "min_reward": min(rewards),
                "max_reward": max(rewards),
                "rounds": [round_index for round_index, _, _ in observations],
                "iterations": [iteration for _, iteration, _ in observations],
                "rewards": rewards,
            }
        )
    ranking.sort(
        key=lambda row: (
            -float(row["mean_reward"]),
            -int(row["selection_count"]),
            str(row["arm_id"]),
        )
    )
    return ranking


def reward_field_for_run(run_root: Path) -> str:
    summary_path = run_root / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        reward_field = str(payload.get("reward_field", "hard")).strip().lower()
        if reward_field in {"hard", "soft"}:
            return reward_field
    return "soft" if "reward_soft" in str(run_root) else "hard"


def materialize_best_skill(
    run_root: Path,
    *,
    reward_field: str | None = None,
    destination_name: str = "optimized_skill",
) -> dict[str, Any]:
    run_root = run_root.resolve()
    history_path = run_root / "history.jsonl"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history: {history_path}")
    objective = reward_field or reward_field_for_run(run_root)
    ranking = aggregate_selected_skills(load_history(history_path), objective)
    if not ranking:
        raise ValueError(f"No selected skills found in {history_path}")

    best = ranking[0]
    skill_root = Path(str(best["skill_root"]))
    skill_path = skill_root / "skill.md"
    if not skill_path.exists():
        skill_path = skill_root / "SKILL.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"Missing skill.md under {skill_root}")

    destination = run_root / destination_name
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(skill_path, destination / "skill.md")
    selection = {
        "selection_rule": "highest_mean_reward_across_selected_iterations",
        "reward_field": objective,
        "best_arm": best,
        "source_skill_path": str(skill_path.resolve()),
        "ranking": ranking,
    }
    (destination / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return selection
