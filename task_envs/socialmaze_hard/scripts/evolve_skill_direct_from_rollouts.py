#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]

from cobras.cobras_core.model import chat_optimizer, configure_azure_openai, set_optimizer_backend, set_optimizer_deployment, set_reasoning_effort  # noqa: E402


META_PROMPT = """You are mutating a reusable Markdown SKILL.md for SocialMaze Hidden Role Deduction.

You receive an existing skill and a small random rollout sample from the target model using that skill. Write a conservative but meaningful mutation.

Constraints:
- Output only the full Markdown skill.md content.
- Do not wrap the answer in code fences.
- Do not copy concrete task IDs, exact questions, source indexes, exact player-number answers, or gold labels.
- Do not rely on answer distribution priors.
- Preserve useful existing guidance, but you may change the strategy if the rollout evidence suggests a better generic behavior.
- Keep the skill concise enough to inject into an already long system prompt.
- Prefer operational reasoning steps over long explanations.
"""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", text).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mutate a SocialMaze skill directly from COBRAS rollout samples.")
    parser.add_argument("--rollouts-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--provider", type=str, default="yunwu")
    parser.add_argument("--model", type=str, default="openai/gpt-5.4")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key-env", type=str, default="TEACHER_API_KEY")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--openai-base-url-env", type=str, default="OPENAI_BASE_URL")
    return parser.parse_args()


def load_rollouts(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_path = root / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if not isinstance(rows, list) or not rows:
        raise FileNotFoundError(f"No rollout rows in {summary_path}")
    return [row for row in rows if isinstance(row, dict)], payload


def read_skill(skill_root: Path) -> str:
    for name in ("skill.md", "SKILL.md"):
        path = skill_root / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"No skill.md under {skill_root}")


def compact_rollout(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row.get("question") or "")
    response = str(row.get("response") or row.get("predicted_answer") or "")
    return {
        "outcome": "success" if float(row.get("hard_reward", row.get("hard", 0.0))) >= 1.0 else "failure",
        "hard_reward": row.get("hard_reward", row.get("hard", 0.0)),
        "soft_reward": row.get("soft_reward", row.get("soft", 0.0)),
        "gold_parsed": row.get("gold_parsed", {}),
        "predicted_parsed": row.get("predicted_parsed", {}),
        "fail_reason": row.get("fail_reason", ""),
        "question_excerpt": question[:3200],
        "assistant_answer": response[:1000],
    }


def main() -> None:
    args = parse_args()
    rows, rollout_summary = load_rollouts(args.rollouts_root.resolve())
    parent_skill_root = Path(str(rows[0]["skill_root"])).resolve()
    parent_skill = read_skill(parent_skill_root)
    model = os.environ.get("SOCIALMAZE_SKILL_MUTATE_MODEL") or os.environ.get("SKILL_MUTATE_MODEL") or os.environ.get("SKILL_GEN_MODEL") or "openai/gpt-5.4"
    reasoning_effort = os.environ.get("SKILL_MUTATE_REASONING_EFFORT") or os.environ.get("SKILL_GEN_REASONING_EFFORT") or "medium"
    base_url = args.base_url or os.environ.get("TEACHER_BASE_URL") or "https://openrouter.ai/api/v1"
    api_key = args.api_key or os.environ.get(args.api_key_env) or os.environ.get("TEACHER_API_KEY")
    if not api_key:
        raise SystemExit(f"Missing API key via --api-key or {args.api_key_env}")
    os.environ[args.openai_base_url_env] = base_url
    os.environ[args.api_key_env] = api_key
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
    observations = {
        "rollouts_root": str(args.rollouts_root.resolve()),
        "sample_size": len(rows),
        "rollout_summary": rollout_summary,
        "samples": [compact_rollout(row) for row in rows],
    }
    user = (
        META_PROMPT
        + "\n\n## Existing SKILL.md\n"
        + parent_skill.strip()
        + "\n\n## Rollout Observations\n"
        + json.dumps(observations, ensure_ascii=False, indent=2)
        + "\n\nReturn the full mutated Markdown skill.md now.\n"
    )
    response, usage = chat_optimizer(
        system="You edit reusable benchmark skills.",
        user=user,
        max_completion_tokens=int(os.environ.get("SKILL_MUTATE_MAX_COMPLETION_TOKENS", "6000")),
        retries=5,
        stage="socialmaze_cobras_rollout_mutate",
        reasoning_effort=reasoning_effort,
        timeout=300,
    )
    out_root = args.out_root.resolve() / f"{args.rollouts_root.name}_direct_model_{args.model.replace('/', '_')}"
    workspace_dir = out_root / "workspace_001"
    skill_root = workspace_dir / "skill"
    skill_root.mkdir(parents=True, exist_ok=True)
    mutated = strip_code_fence(response)
    (skill_root / "skill.md").write_text(mutated.rstrip() + "\n", encoding="utf-8")
    (workspace_dir / "parent_skill.md").write_text(parent_skill, encoding="utf-8")
    (workspace_dir / "mutation_prompt.txt").write_text(user, encoding="utf-8")
    (workspace_dir / "mutation_raw_response.txt").write_text(response, encoding="utf-8")
    write_json(workspace_dir / "metadata.json", {"parent_skill_root": str(parent_skill_root), "usage": usage})
    row = {
        "skill_name": "socialmaze_rollout_mutated_001",
        "workspace_dir": str(workspace_dir),
        "skill_root": str(skill_root),
        "parent_skill_root": str(parent_skill_root),
        "avg_soft_reward": 0.0,
        "avg_hard_reward": 0.0,
        "count": 0,
        "summary_path": str(out_root / "summary.json"),
        "origin": "rollout_mutate",
    }
    write_json(
        out_root / "summary.json",
        {
            "rollouts_root": str(args.rollouts_root.resolve()),
            "model": args.model,
            "skill_mutation_model": model,
            "generated": [row],
            "usage": usage,
        },
    )
    print(str(out_root / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
