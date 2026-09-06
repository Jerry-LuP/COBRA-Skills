from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def format_score(row: dict[str, Any]) -> str:
    reward = row.get("latest_reward")
    reward_text = "NA" if reward is None else f"{float(reward):.4f}"
    return (
        f"{row['arm_id']} pred={float(row['pred']):.4f} "
        f"bonus={float(row['bonus']):.4f} score={float(row['ucb_score']):.4f} "
        f"latest={reward_text} origin={row.get('origin', '')}"
    )


def run_module(module: str, args: list[str], *, timeout: int = 7200, stream: bool = False) -> tuple[str, str]:
    cmd = [sys.executable, "-m", module, *args]
    if stream:
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            check=False,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{module} failed with exit code {completed.returncode}")
        return "", ""
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{module} failed with exit code {completed.returncode}: "
            f"{(completed.stderr or completed.stdout).strip()[:4000]}"
        )
    return completed.stdout, completed.stderr


def copy_skill_tree(src_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval_rows(summary_path: Path) -> list[dict[str, Any]]:
    payload = load_json(summary_path)
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise ValueError(f"Expected results list in {summary_path}")
    return [row for row in rows if isinstance(row, dict)]


def load_initial_arms(initial_eval_summary: Path) -> list[dict[str, Any]]:
    rows = load_eval_rows(initial_eval_summary)
    arms: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        skill_root = Path(str(row.get("skill_root", ""))).resolve()
        workspace_dir = Path(str(row.get("workspace_dir", skill_root.parent))).resolve()
        arms.append(
            {
                "arm_id": f"arm_{index:03d}",
                "skill_name": str(row.get("skill_name") or workspace_dir.name or f"arm_{index:03d}"),
                "skill_root": str(skill_root),
                "workspace_dir": str(workspace_dir),
                "origin": "init",
                "parent_ids": [],
                "generation": 0,
            }
        )
    if not arms:
        raise FileNotFoundError(f"No initial arms found in {initial_eval_summary}")
    return arms


def arm_serial(arm_id: str) -> int:
    try:
        return int(str(arm_id).rsplit("_", 1)[-1])
    except Exception:
        return 0


def load_generated_child(summary_path: Path) -> dict[str, Any]:
    payload = load_json(summary_path)
    generated = payload.get("generated", [])
    if not isinstance(generated, list) or not generated:
        raise FileNotFoundError(f"No generated skills found in {summary_path}")
    row = generated[0]
    if not isinstance(row, dict):
        raise ValueError(f"Expected dict generated row in {summary_path}")
    return row


def skill_entrypoint_path(skill_root: Path) -> Path:
    lower = skill_root / "skill.md"
    upper = skill_root / "SKILL.md"
    if lower.exists():
        return lower
    if upper.exists():
        return upper
    raise FileNotFoundError(f"Could not find skill.md or SKILL.md under {skill_root}")


def deterministic_rng(*parts: Any) -> random.Random:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


@dataclass
class ArmState:
    arm_id: str
    skill_name: str
    skill_root: str
    workspace_dir: str
    latest_reward: float | None = None
    latest_soft_reward: float | None = None
    latest_hard_reward: float | None = None
    latest_eval_summary: str | None = None
    latest_eval_root: str | None = None
    latest_train_rollout_root: str | None = None
    origin: str = "init"
    parent_ids: list[str] | None = None
    generation: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "skill_name": self.skill_name,
            "skill_root": self.skill_root,
            "workspace_dir": self.workspace_dir,
            "latest_reward": self.latest_reward,
            "latest_soft_reward": self.latest_soft_reward,
            "latest_hard_reward": self.latest_hard_reward,
            "latest_eval_summary": self.latest_eval_summary,
            "latest_eval_root": self.latest_eval_root,
            "latest_train_rollout_root": self.latest_train_rollout_root,
            "origin": self.origin,
            "parent_ids": self.parent_ids or [],
            "generation": self.generation,
        }
