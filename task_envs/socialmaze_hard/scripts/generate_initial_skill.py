#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GENERATOR = PROJECT_ROOT / "task_envs" / "socialmaze_hard" / "scripts" / "generate_socialmaze_hidden_role_skill_from_results.py"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SocialMaze initial skills for COBRAS.")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--provider", type=str, default="yunwu")
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--min-success", type=int, default=0)
    parser.add_argument("--min-fail", type=int, default=0)
    parser.add_argument("--num-skills", type=int, default=1)
    parser.add_argument("--parallel-skills", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="openai/gpt-5.4")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key-env", type=str, default="TEACHER_API_KEY")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--openai-base-url-env", type=str, default="OPENAI_BASE_URL")
    return parser.parse_args()


def child_env(args: argparse.Namespace, out_dir: Path, seed: int) -> dict[str, str]:
    env = os.environ.copy()
    skill_model = env.get("SOCIALMAZE_SKILL_GEN_MODEL") or env.get("SKILL_GEN_MODEL") or "openai/gpt-5.4"
    env["RUN_ROOT"] = str(args.baseline_root.resolve())
    env["RESULTS_PATH"] = str((args.baseline_root / "results.jsonl").resolve())
    env["OUT_DIR"] = str(out_dir)
    env["SAMPLE_SIZE"] = str(args.sample_size)
    env["SAMPLE_SEED"] = str(seed)
    env["SKILL_GEN_MODEL"] = skill_model
    env["SKILL_GEN_REASONING_EFFORT"] = env.get("SKILL_GEN_REASONING_EFFORT") or "medium"
    if args.base_url:
        env["SKILL_GEN_BASE_URL"] = args.base_url
        env[args.openai_base_url_env] = args.base_url
    if args.api_key:
        env["SKILL_GEN_API_KEY"] = args.api_key
        env[args.api_key_env] = args.api_key
    return env


def main() -> None:
    args = parse_args()
    summary_root = args.out_root.resolve() / f"sample{args.sample_size}_skills{args.num_skills}"
    summary_root.mkdir(parents=True, exist_ok=True)
    skill_model = os.environ.get("SOCIALMAZE_SKILL_GEN_MODEL") or os.environ.get("SKILL_GEN_MODEL") or "openai/gpt-5.4"
    generated: list[dict[str, Any]] = []
    for index in range(1, args.num_skills + 1):
        workspace_dir = summary_root / f"workspace_{index:03d}"
        skill_root = workspace_dir / "skill"
        skill_root.mkdir(parents=True, exist_ok=True)
        env = child_env(args, skill_root, args.seed + index)
        completed = subprocess.run(
            [sys.executable, str(GENERATOR)],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=1800,
        )
        (workspace_dir / "generator_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
        (workspace_dir / "generator_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"SocialMaze skill generation failed: {(completed.stderr or completed.stdout)[-4000:]}")
        if not (skill_root / "skill.md").exists():
            raise FileNotFoundError(f"Generator did not produce {skill_root / 'skill.md'}")
        generated.append(
            {
                "skill_name": f"generated_socialmaze_{index:03d}",
                "workspace_dir": str(workspace_dir),
                "skill_root": str(skill_root),
                "avg_soft_reward": 0.0,
                "avg_hard_reward": 0.0,
                "count": 0,
                "summary_path": str(summary_root / "summary.json"),
                "origin": "regenerate",
            }
        )
        print(f"[socialmaze-generate-initial] generated workspace_{index:03d}", flush=True)
    write_json(
        summary_root / "summary.json",
        {
            "baseline_root": str(args.baseline_root.resolve()),
            "sample_size": args.sample_size,
            "num_skills": args.num_skills,
            "model": args.model,
            "skill_generation_model": skill_model,
            "generated": generated,
        },
    )
    print(str(summary_root / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
