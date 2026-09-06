#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cobras.scripts.eval_command import build_eval_command


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "socialmaze_hard"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated SocialMaze skills for COBRAS.")
    parser.add_argument("--skills-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--parallel", type=int, default=15)
    parser.add_argument("--skill-parallel", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="openai/gpt-5.4-nano")
    parser.add_argument("--provider", type=str, default="yunwu")
    parser.add_argument("--reasoning-effort", type=str, default="")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key-env", type=str, default="TEACHER_API_KEY")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--openai-base-url-env", type=str, default="OPENAI_BASE_URL")
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--task-timeout", type=int, default=240)
    return parser.parse_args()


def discover_skills(skills_root: Path) -> list[dict[str, str]]:
    summary_path = skills_root / "summary.json"
    found: list[dict[str, str]] = []
    if summary_path.exists():
        payload = load_json(summary_path)
        for index, row in enumerate(payload.get("generated", []) or payload.get("results", []), start=1):
            if not isinstance(row, dict):
                continue
            skill_root = Path(str(row.get("skill_root", ""))).resolve()
            workspace_dir = Path(str(row.get("workspace_dir") or skill_root.parent)).resolve()
            if (skill_root / "skill.md").exists() or (skill_root / "SKILL.md").exists():
                found.append(
                    {
                        "name": workspace_dir.name or f"workspace_{index:03d}",
                        "workspace_dir": str(workspace_dir),
                        "skill_root": str(skill_root),
                    }
                )
    if not found:
        for index, workspace_dir in enumerate(sorted(skills_root.glob("workspace_*")), start=1):
            skill_root = workspace_dir / "skill"
            if (skill_root / "skill.md").exists() or (skill_root / "SKILL.md").exists():
                found.append(
                    {
                        "name": workspace_dir.name or f"workspace_{index:03d}",
                        "workspace_dir": str(workspace_dir.resolve()),
                        "skill_root": str(skill_root.resolve()),
                    }
                )
    if not found and ((skills_root / "skill.md").exists() or (skills_root / "SKILL.md").exists()):
        found.append({"name": "workspace_001", "workspace_dir": str(skills_root.resolve()), "skill_root": str(skills_root.resolve())})
    if not found:
        raise FileNotFoundError(f"No skill.md found under {skills_root}")
    return found


def data_path_for(args: argparse.Namespace) -> str:
    if args.split_json:
        return str(args.split_json.resolve())
    candidate = DEFAULT_DATA_ROOT / f"{args.split}.jsonl"
    if candidate.exists():
        return str(candidate.resolve())
    return ""


def runner_provider(provider: str, base_url: str) -> str:
    text = f"{provider} {base_url}".lower()
    if "openrouter" in text:
        return "openrouter"
    if "yunwu" in text:
        return "yunwu"
    return provider.lower()


def prepare_env(args: argparse.Namespace, skill_root: str, out_root: Path, name: str) -> dict[str, str]:
    env = os.environ.copy()
    provider = runner_provider(args.provider, args.base_url)
    env["PROVIDER"] = provider
    env["MODEL"] = args.model
    env["TARGET_MODEL"] = args.model
    env["SPLIT"] = args.split
    env["LIMIT"] = str(args.limit)
    env["NUM_TASKS"] = str(args.limit)
    env["WORKERS"] = str(max(1, args.parallel))
    env["SKILL_PATH"] = skill_root
    env["OUT_ROOT"] = str(out_root)
    env["RUN_ID"] = name
    env["COBRAS_TARGET_REASONING_EFFORT"] = args.reasoning_effort or ""
    env["MAX_COMPLETION_TOKENS"] = str(args.max_completion_tokens)
    env["TASK_TIMEOUT"] = str(args.task_timeout)
    if path := data_path_for(args):
        env["DATA_PATH"] = path
    if args.api_key:
        env[args.api_key_env] = args.api_key
    if args.base_url:
        env[args.openai_base_url_env] = args.base_url
        if provider == "openrouter":
            env["TEACHER_BASE_URL"] = args.base_url
        elif provider == "yunwu":
            env["TEACHER_BASE_URL"] = args.base_url
    return env


def convert_trials(workspace_eval_dir: Path, skill: dict[str, str]) -> list[dict[str, Any]]:
    rows = load_jsonl(workspace_eval_dir / "results.jsonl")
    trials: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("id", ""))
        prediction_dir = workspace_eval_dir / "predictions" / row_id
        user_prompt = ""
        response = str(row.get("predicted_answer") or "")
        if (prediction_dir / "target_user_prompt.txt").exists():
            user_prompt = (prediction_dir / "target_user_prompt.txt").read_text(encoding="utf-8", errors="ignore")
        if (prediction_dir / "raw.txt").exists():
            response = (prediction_dir / "raw.txt").read_text(encoding="utf-8", errors="ignore")
        trial = {
            "id": row_id,
            "uid": row_id,
            "question": user_prompt,
            "task_type": row.get("task", "hidden_role_deduction"),
            "task_description": user_prompt,
            "response": response,
            "predicted_answer": response,
            "predicted_parsed": row.get("predicted_parsed", {}),
            "ground_truth": row.get("gold_answer", ""),
            "gold_parsed": row.get("gold_parsed", {}),
            "hard_reward": float(row.get("hard", 0.0)),
            "soft_reward": float(row.get("soft", 0.0)),
            "hard": int(row.get("hard", 0)),
            "soft": float(row.get("soft", 0.0)),
            "fail_reason": row.get("fail_reason", ""),
            "artifact_paths": {
                "prediction_dir": str(prediction_dir),
                "conversation": str(prediction_dir / "conversation.json"),
                "reward": str(prediction_dir / "reward.json"),
                "result": str(prediction_dir / "result.json"),
            },
            "skill_root": skill["skill_root"],
            "skill_name": skill["name"],
            "workspace_dir": skill["workspace_dir"],
        }
        trials.append(trial)
    with (workspace_eval_dir / "trial_results.jsonl").open("w", encoding="utf-8") as handle:
        for trial in trials:
            handle.write(json.dumps(trial, ensure_ascii=False) + "\n")
    return trials


def evaluate_one(args: argparse.Namespace, run_root: Path, skill: dict[str, str], index: int) -> dict[str, Any]:
    workspace_name = f"workspace_{index:03d}"
    workspace_eval_dir = run_root / workspace_name
    if workspace_eval_dir.exists():
        shutil.rmtree(workspace_eval_dir)
    workspace_eval_dir.mkdir(parents=True, exist_ok=True)
    env = prepare_env(args, skill["skill_root"], workspace_eval_dir, workspace_name)
    provider = runner_provider(args.provider, args.base_url)
    data_path = Path(data_path_for(args)) if data_path_for(args) else None
    completed = subprocess.run(
        build_eval_command(
            dataset="socialmaze_hard",
            split=args.split,
            limit=args.limit,
            workers=max(1, args.parallel),
            skill_path=Path(skill["skill_root"]),
            out_root=workspace_eval_dir,
            provider=provider,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            reasoning_effort=args.reasoning_effort or None,
            max_completion_tokens=args.max_completion_tokens,
            task_timeout=args.task_timeout,
            data_path=data_path,
        ),
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=max(7200, args.task_timeout * max(1, args.limit) + 600),
    )
    (workspace_eval_dir / "runner_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (workspace_eval_dir / "runner_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"SocialMaze runner failed for {skill['name']}: {(completed.stderr or completed.stdout)[-4000:]}")
    summary = load_json(workspace_eval_dir / "summary.json")
    trials = convert_trials(workspace_eval_dir, skill)
    return {
        "skill_name": skill["name"],
        "workspace_dir": skill["workspace_dir"],
        "skill_root": skill["skill_root"],
        "avg_soft_reward": float(summary.get("avg_soft", 0.0)),
        "avg_hard_reward": float(summary.get("hard_acc", 0.0)),
        "success_tasks": int(summary.get("hard", 0)),
        "count": int(summary.get("n", len(trials))),
        "summary_path": str(workspace_eval_dir / "summary.json"),
        "trial_results_path": str(workspace_eval_dir / "trial_results.jsonl"),
        "criminal_acc": float(summary.get("criminal_acc", 0.0)),
        "role_acc": float(summary.get("role_acc", 0.0)),
    }


def main() -> None:
    args = parse_args()
    args.skills_root = args.skills_root.resolve()
    args.out_root = args.out_root.resolve()
    skills = discover_skills(args.skills_root)
    run_name = args.run_name or f"{args.skills_root.name}_model_{args.model.replace('/', '_')}_{args.split}"
    run_root = args.out_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    max_skill_workers = max(1, args.skill_parallel)
    print(f"[socialmaze-eval] skills={len(skills)} run_root={run_root}", flush=True)
    with ThreadPoolExecutor(max_workers=max_skill_workers) as executor:
        future_map = {
            executor.submit(evaluate_one, args, run_root, skill, index): (skill, index)
            for index, skill in enumerate(skills, start=1)
        }
        for future in as_completed(future_map):
            skill, _ = future_map[future]
            row = future.result()
            results.append(row)
            print(
                f"[socialmaze-eval] done skill={skill['name']} hard={row['avg_hard_reward']:.4f} soft={row['avg_soft_reward']:.4f}",
                flush=True,
            )
    results.sort(key=lambda row: row["workspace_dir"])
    payload = {
        "skills_root": str(args.skills_root),
        "split": args.split,
        "limit": args.limit,
        "model": args.model,
        "provider": args.provider,
        "parallel": args.parallel,
        "skill_parallel": args.skill_parallel,
        "results": results,
    }
    write_json(run_root / "summary.json", payload)
    print(str(run_root / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
