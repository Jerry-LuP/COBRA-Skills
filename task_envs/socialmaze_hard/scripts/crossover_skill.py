#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]

from cobras.cobras_core.model import (  # noqa: E402
    chat_optimizer,
    configure_azure_openai,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_reasoning_effort,
)


META_PROMPT = """You are creating one reusable Markdown SKILL.md for SocialMaze Hidden Role Deduction by crossing over parent skills.

You receive higher-scoring parent skills and lower-scoring parent skills. Use higher-scoring skills as positive evidence about useful reasoning. Use lower-scoring skills as negative evidence about wording or strategies to avoid.

Constraints:
- Output only the full Markdown skill.md content.
- Do not wrap the answer in code fences.
- Do not copy concrete task IDs, exact questions, source indexes, exact player-number answers, or gold labels.
- Do not rely on answer distribution priors.
- Keep the skill concise enough to inject into an already long system prompt.
- Prefer operational hidden-role reasoning guidance over broad advice.
- It is acceptable to preserve the best parent's core strategy and make only a small high-confidence edit.
"""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", text).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SocialMaze skills by crossing over good and bad parent skills.")
    parser.add_argument("--eval-summary", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--num-children", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--top-pool-size", type=int, default=4)
    parser.add_argument("--bottom-pool-size", type=int, default=4)
    parser.add_argument("--score-field", type=str, default="avg_soft_reward")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider", type=str, default="yunwu")
    parser.add_argument("--model", type=str, default="openai/gpt-5.4")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key-env", type=str, default="TEACHER_API_KEY")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--openai-base-url-env", type=str, default="OPENAI_BASE_URL")
    return parser.parse_args()


def skill_path(skill_root: Path) -> Path:
    for name in ("skill.md", "SKILL.md"):
        path = skill_root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No skill.md under {skill_root}")


def read_skill(skill_root: Path) -> str:
    return skill_path(skill_root).read_text(encoding="utf-8", errors="ignore")


def load_parent_rows(summary_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise ValueError(f"Expected results list in {summary_path}")
    parents: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        skill_root = Path(str(row.get("skill_root", ""))).resolve()
        if skill_root.exists() and (skill_root / "skill.md").exists():
            item = dict(row)
            item["skill_root"] = str(skill_root)
            parents.append(item)
    if not parents:
        raise FileNotFoundError(f"No usable parent skills found in {summary_path}")
    return parents


def score(row: dict[str, Any], field: str) -> float:
    if field in row:
        return float(row.get(field) or 0.0)
    return float(row.get("avg_soft_reward") or row.get("avg_hard_reward") or 0.0)


def choose_parents(rows: list[dict[str, Any]], args: argparse.Namespace, rng: random.Random) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(rows, key=lambda row: score(row, args.score_field), reverse=True)
    top_pool = ranked[: max(1, min(args.top_pool_size, len(ranked)))]
    bottom_pool = ranked[-max(1, min(args.bottom_pool_size, len(ranked))) :]
    backbone = top_pool[0]
    good_candidates = [row for row in top_pool if row is not backbone]
    good_k = min(max(0, args.k - 1), len(good_candidates))
    bad_k = min(max(1, args.k), len(bottom_pool))
    good = rng.sample(good_candidates, good_k) if good_k else []
    bad = rng.sample(bottom_pool, bad_k)
    return backbone, good, bad


def parent_digest(label: str, role: str, row: dict[str, Any], score_field: str) -> dict[str, Any]:
    skill_root = Path(str(row["skill_root"]))
    text = read_skill(skill_root)
    return {
        "label": label,
        "role": role,
        "skill_name": row.get("skill_name", skill_root.parent.name),
        "score": score(row, score_field),
        "avg_soft_reward": row.get("avg_soft_reward"),
        "avg_hard_reward": row.get("avg_hard_reward"),
        "success_tasks": row.get("success_tasks"),
        "count": row.get("count"),
        "skill_root": str(skill_root),
        "skill_md": text[:6000],
    }


def build_prompt(backbone: dict[str, Any], good: list[dict[str, Any]], bad: list[dict[str, Any]], score_field: str) -> str:
    parents = [parent_digest("backbone", "best", backbone, score_field)]
    parents.extend(parent_digest(f"good_{idx:02d}", "good", row, score_field) for idx, row in enumerate(good, start=1))
    parents.extend(parent_digest(f"bad_{idx:02d}", "bad", row, score_field) for idx, row in enumerate(bad, start=1))
    return (
        META_PROMPT
        + "\n\n## Parent Skills\n"
        + json.dumps(parents, ensure_ascii=False, indent=2)
        + "\n\n## Crossover Instructions\n"
        + "- Start from the backbone/best parent's useful structure.\n"
        + "- Borrow at most a few concrete reasoning rules from good parents.\n"
        + "- Avoid patterns that appear mainly in bad parents, especially vague answer-format-only guidance or overconfident shortcuts.\n"
        + "- Produce one complete Markdown skill.md for future SocialMaze Hidden Role tasks.\n"
        + "- Do not include a separate explanation.\n"
    )


def configure_optimizer(args: argparse.Namespace) -> tuple[str, str, str]:
    model = os.environ.get("SOCIALMAZE_SKILL_CROSSOVER_MODEL") or os.environ.get("SKILL_CROSSOVER_MODEL") or os.environ.get("SKILL_GEN_MODEL") or "openai/gpt-5.4"
    reasoning_effort = os.environ.get("SKILL_CROSSOVER_REASONING_EFFORT") or os.environ.get("SKILL_GEN_REASONING_EFFORT") or "medium"
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
    return model, reasoning_effort, base_url


def generate_child(child_index: int, args: argparse.Namespace, rows: list[dict[str, Any]], out_root: Path) -> dict[str, Any]:
    rng = random.Random(args.seed + child_index)
    backbone, good, bad = choose_parents(rows, args, rng)
    prompt = build_prompt(backbone, good, bad, args.score_field)
    response, usage = chat_optimizer(
        system="You write concise reusable benchmark skills.",
        user=prompt,
        max_completion_tokens=int(os.environ.get("SKILL_CROSSOVER_MAX_COMPLETION_TOKENS", "6000")),
        retries=5,
        stage="socialmaze_cobras_crossover",
        reasoning_effort=os.environ.get("SKILL_CROSSOVER_REASONING_EFFORT") or os.environ.get("SKILL_GEN_REASONING_EFFORT") or "medium",
        timeout=300,
    )
    workspace_dir = out_root / f"workspace_{child_index:03d}"
    skill_root = workspace_dir / "skill"
    skill_root.mkdir(parents=True, exist_ok=True)
    skill_text = strip_code_fence(response)
    (skill_root / "skill.md").write_text(skill_text.rstrip() + "\n", encoding="utf-8")
    (workspace_dir / "crossover_prompt.txt").write_text(prompt, encoding="utf-8")
    (workspace_dir / "crossover_raw_response.txt").write_text(response, encoding="utf-8")
    metadata = {
        "workspace_dir": str(workspace_dir),
        "skill_root": str(skill_root),
        "score_field": args.score_field,
        "backbone_parent": backbone,
        "good_parents": good,
        "bad_parents": bad,
        "usage": usage,
    }
    write_json(workspace_dir / "metadata.json", metadata)
    return {
        "skill_name": f"socialmaze_crossover_{child_index:03d}",
        "workspace_dir": str(workspace_dir),
        "skill_root": str(skill_root),
        "avg_soft_reward": 0.0,
        "avg_hard_reward": 0.0,
        "count": 0,
        "summary_path": str(out_root / "summary.json"),
        "origin": "crossover",
        "backbone_skill_root": str(backbone["skill_root"]),
    }


def main() -> None:
    args = parse_args()
    args.eval_summary = args.eval_summary.resolve()
    rows = load_parent_rows(args.eval_summary)
    crossover_model, reasoning_effort, base_url = configure_optimizer(args)
    out_root = (args.out_root / f"children{args.num_children}_k{args.k}_model_{args.model.replace('/', '_')}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        future_map = {
            executor.submit(generate_child, child_index, args, rows, out_root): child_index
            for child_index in range(1, args.num_children + 1)
        }
        for future in as_completed(future_map):
            child_index = future_map[future]
            row = future.result()
            generated.append(row)
            print(f"[socialmaze-crossover] workspace_{child_index:03d} completed", flush=True)
    generated.sort(key=lambda row: row["workspace_dir"])
    write_json(
        out_root / "summary.json",
        {
            "model": args.model,
            "crossover_model": crossover_model,
            "reasoning_effort": reasoning_effort,
            "base_url": base_url,
            "eval_summary": str(args.eval_summary),
            "score_field": args.score_field,
            "k": args.k,
            "top_pool_size": args.top_pool_size,
            "bottom_pool_size": args.bottom_pool_size,
            "num_children": args.num_children,
            "generated": generated,
        },
    )
    print(str(out_root / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
