from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cobras.cobras_core.llm import resolve_provider
from cobras.cobras_core.model import (
    chat_optimizer,
    configure_azure_openai,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_reasoning_effort,
)


@dataclass(frozen=True)
class PromptCrossoverSpec:
    dataset: str
    system_prompt: str
    stage: str
    instructions: tuple[str, ...]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _skill_path(skill_root: Path) -> Path:
    for name in ("skill.md", "SKILL.md"):
        path = skill_root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No skill.md under {skill_root}")


def _strip_code_fence(text: str) -> str:
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", text).strip()


def _parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--eval-summary", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--num-children", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--top-pool-size", type=int, default=4)
    parser.add_argument("--bottom-pool-size", type=int, default=4)
    parser.add_argument("--score-field", default="ucb_score")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model", default="openai/gpt-5.4")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--openai-base-url-env", default="OPENAI_BASE_URL")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise ValueError(f"Expected results list in {path}")
    usable = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        skill_root = Path(str(row.get("skill_root", ""))).resolve()
        if skill_root.exists():
            item = dict(row)
            item["skill_root"] = str(skill_root)
            usable.append(item)
    if not usable:
        raise FileNotFoundError(f"No scored parent skills found in {path}")
    return usable


def _select_parents(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    top_pool_size: int,
    bottom_pool_size: int,
    k: int,
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    required = max(top_pool_size, bottom_pool_size)
    if len(rows) < required:
        raise ValueError(f"Need at least {required} scored parent skills, found {len(rows)}")
    if k < 1 or k > top_pool_size or k > bottom_pool_size:
        raise ValueError("k must be between 1 and both pool sizes")
    ranked = sorted(rows, key=lambda row: float(row.get(score_field, 0.0)), reverse=True)
    top_pool = ranked[:top_pool_size]
    bottom_pool = ranked[-bottom_pool_size:]
    backbone = rng.choice(top_pool)
    good_pool = [row for row in top_pool if row is not backbone]
    good = rng.sample(good_pool, min(k - 1, len(good_pool)))
    bad = rng.sample(bottom_pool, min(k, len(bottom_pool)))
    return backbone, good, bad


def _parent_block(label: str, row: dict[str, Any], score_field: str) -> str:
    skill_root = Path(str(row["skill_root"]))
    return "\n".join(
        [
            f"## {label}",
            f"score={float(row.get(score_field, 0.0)):.6f}",
            f"hard={float(row.get('avg_hard_reward', 0.0)):.6f}",
            "```markdown",
            _skill_path(skill_root).read_text(encoding="utf-8", errors="ignore"),
            "```",
        ]
    )


def _configure_optimizer(args: argparse.Namespace) -> None:
    provider = resolve_provider(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    api_key = args.api_key or os.environ.get(provider.api_key_env, "")
    if not api_key and provider.provider != "local":
        raise RuntimeError(f"Missing API key: {provider.api_key_env}")
    set_optimizer_deployment(provider.model)
    set_reasoning_effort(args.reasoning_effort)
    set_optimizer_backend("openai_chat")
    configure_azure_openai(
        endpoint=provider.base_url,
        api_key=api_key,
        auth_mode="openai_compatible",
        optimizer_endpoint=provider.base_url,
        optimizer_api_key=api_key,
        optimizer_auth_mode="openai_compatible",
    )


def _generate_one(
    *,
    child_index: int,
    rows: list[dict[str, Any]],
    out_root: Path,
    args: argparse.Namespace,
    spec: PromptCrossoverSpec,
) -> dict[str, Any]:
    backbone, good, bad = _select_parents(
        rows,
        score_field=args.score_field,
        top_pool_size=args.top_pool_size,
        bottom_pool_size=args.bottom_pool_size,
        k=args.k,
        rng=random.Random(args.seed + child_index),
    )
    blocks = [_parent_block("Backbone parent", backbone, args.score_field)]
    blocks.extend(
        _parent_block(f"Good parent {index}", row, args.score_field)
        for index, row in enumerate(good, start=1)
    )
    blocks.extend(
        _parent_block(f"Weak parent {index}", row, args.score_field)
        for index, row in enumerate(bad, start=1)
    )
    prompt = "\n\n".join([*spec.instructions, *blocks])
    workspace_dir = out_root / f"workspace_{child_index:03d}"
    skill_root = workspace_dir / "skill"
    skill_root.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "crossover_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    response, usage = chat_optimizer(
        system=spec.system_prompt,
        user=prompt,
        max_completion_tokens=int(os.environ.get("SKILL_CROSSOVER_MAX_COMPLETION_TOKENS", "6000")),
        retries=5,
        stage=spec.stage,
        reasoning_effort=args.reasoning_effort,
        timeout=300,
    )
    (skill_root / "skill.md").write_text(_strip_code_fence(response) + "\n", encoding="utf-8")
    metadata = {
        "skill_name": f"crossover_{child_index:03d}",
        "workspace_dir": str(workspace_dir),
        "skill_root": str(skill_root),
        "child_skill_root": str(skill_root),
        "backbone_parent": backbone,
        "good_parents": good,
        "bad_parents": bad,
        "score_field": args.score_field,
        "usage": usage,
    }
    _write_json(workspace_dir / "metadata.json", metadata)
    return metadata


def run_prompt_crossover(spec: PromptCrossoverSpec) -> None:
    args = _parse_args(f"Create {spec.dataset} skills by crossing over scored parents.")
    args.eval_summary = args.eval_summary.resolve()
    rows = _load_rows(args.eval_summary)
    _configure_optimizer(args)
    model_dir = f"children{args.num_children}_k{args.k}_model_{args.model.replace('/', '_')}"
    out_root = (args.out_root / model_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        futures = {
            executor.submit(
                _generate_one,
                child_index=index,
                rows=rows,
                out_root=out_root,
                args=args,
                spec=spec,
            ): index
            for index in range(1, args.num_children + 1)
        }
        for future in as_completed(futures):
            generated.append(future.result())
    generated.sort(key=lambda row: str(row["workspace_dir"]))
    _write_json(
        out_root / "summary.json",
        {
            "dataset": spec.dataset,
            "model": args.model,
            "eval_summary": str(args.eval_summary),
            "score_field": args.score_field,
            "k": args.k,
            "top_pool_size": args.top_pool_size,
            "bottom_pool_size": args.bottom_pool_size,
            "generated": generated,
        },
    )
    print(f"[crossover-skill] dataset={spec.dataset} out={out_root}", flush=True)
