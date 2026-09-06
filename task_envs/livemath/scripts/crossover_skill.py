#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cobras.scripts.env_utils import load_env_files, resolve_chat_provider
from cobras.task_envs.livemath.common import (
    DEFAULT_PROVIDER,
    copy_skill_tree,
    ensure_clean_dir,
    ensure_skill_md,
    skill_file_manifest,
    write_json,
)
from cobras.cobras_core.model import (
    chat_optimizer,
    configure_azure_openai,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_reasoning_effort,
)

LOADED_ENV_FILES = load_env_files(
    [
        Path(__file__).resolve().parent.parent.parent / ".env.livemath.local",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]
)
DEFAULT_MODEL = os.environ.get("LIVEMATH_MODEL", "gpt-5.5")
DEFAULT_BASE_URL = os.environ.get("LIVEMATH_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create new LiveMath skills by crossing over good and bad parent skills.")
    parser.add_argument("--eval-summary", type=Path, required=True, help="Evaluation summary.json for scored parent skills.")
    parser.add_argument("--out-root", type=Path, default=Path("output") / "livemath" / "skill_crossover")
    parser.add_argument("--num-children", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--k", type=int, default=2, help="Randomly sample k parents from the top pool and k from the bottom pool.")
    parser.add_argument("--top-pool-size", type=int, default=3, help="Take the top-N scored skills as the good pool.")
    parser.add_argument("--bottom-pool-size", type=int, default=3, help="Take the bottom-N scored skills as the bad pool.")
    parser.add_argument("--score-field", type=str, default="avg_soft_reward", choices=["avg_soft_reward", "avg_hard_reward"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--provider",
        type=str,
        choices=["local", "openrouter", "yunwu", "custom"],
        default=DEFAULT_PROVIDER,
    )
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key-env", type=str, default="TEACHER_API_KEY")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--openai-base-url-env", type=str, default="OPENAI_BASE_URL")
    return parser.parse_args()


def _parent_score_table(backbone: dict[str, Any], good: list[dict[str, Any]], bad: list[dict[str, Any]], score_field: str) -> str:
    rows = ["label | role | skill_name | score | hard | successes/count"]
    rows.append(
        " | ".join(
            [
                "backbone",
                "backbone",
                str(backbone.get("skill_name", "")),
                f"{float(backbone.get(score_field, 0.0)):.6f}",
                f"{float(backbone.get('avg_hard_reward', 0.0)):.6f}",
                f"{backbone.get('success_tasks', '?')}/{backbone.get('count', '?')}",
            ]
        )
    )
    for role, parents in (("good", good), ("bad", bad)):
        for index, row in enumerate(parents, start=1):
            label = f"{role}_{index:02d}"
            rows.append(
                " | ".join(
                    [
                        label,
                        role,
                        str(row.get("skill_name", "")),
                        f"{float(row.get(score_field, 0.0)):.6f}",
                        f"{float(row.get('avg_hard_reward', 0.0)):.6f}",
                        f"{row.get('success_tasks', '?')}/{row.get('count', '?')}",
                    ]
                )
            )
    return "\n".join(rows)


def _prompt_text(backbone: dict[str, Any], good: list[dict[str, Any]], bad: list[dict[str, Any]], score_field: str) -> str:
    parent_table = _parent_score_table(backbone, good, bad, score_field)
    return "\n".join(
        [
            "You are creating a new LiveMath skill by crossing over high-scoring and low-scoring parent skills.",
            "",
            "Parent scores:",
            parent_table,
            "",
            "Goal:",
            "- Produce one reusable LiveMath SKILL.md by combining the parent skills below.",
            "- The output will be injected directly into a weaker LiveMath target model's system prompt.",
            "- Return only the complete child SKILL.md content.",
            "",
            "Crossover method:",
            "1. Treat the backbone parent as the main skill to preserve and patch.",
            "2. Preserve the backbone's structure and wording unless there is concrete evidence to change it.",
            "3. Use the other good parents only to suggest one small missing patch.",
            "4. Use bad parents only as negative controls: identify advice that is vague, redundant, overbroad, or likely to cause wrong theorem, hypothesis, boundary, or option choices.",
            "5. Do not average all parents, do not rewrite from scratch, and do not make the child more generic than the backbone.",
            "",
            "LiveMath-specific priorities:",
            "- Emphasize exact theorem statement reading before any reasoning.",
            "- Emphasize hypotheses, quantifiers, strict versus non-strict inequalities, boundary cases, and choice wording.",
            "- Keep answer selection precise: compare all options line by line before choosing a label.",
            "- Prefer concise imperative workflow steps over long explanations.",
            "- Do not include benchmark-specific answers, source file names, UID values, or one-off facts.",
            "",
            "Important:",
            "- A small high-confidence edit to the best parent is better than an ambitious full rewrite.",
            "- If a bad parent contains an instruction that also appears in a good parent, do not reject it solely because it appears in a bad parent.",
            "",
            "Required output:",
            "- Output only the complete child SKILL.md content.",
            "- Do not include crossover notes, explanations, markdown code fences, or any commentary outside the skill.",
        ]
    )


def _load_scored_parents(summary_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise ValueError(f"Expected results list in {summary_path}")
    parents: list[dict[str, Any]] = []
    for row in rows:
        skill_root = Path(str(row.get("skill_root", "")))
        if not skill_root.exists():
            continue
        parents.append(row)
    if not parents:
        raise FileNotFoundError(f"No scored skill roots found in {summary_path}")
    return parents


def _select_parents(
    rows: list[dict[str, Any]],
    *,
    k: int,
    score_field: str,
    top_pool_size: int,
    bottom_pool_size: int,
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(rows, key=lambda row: float(row.get(score_field, 0.0)), reverse=True)
    if len(ranked) < max(top_pool_size, bottom_pool_size):
        raise ValueError(
            f"Need at least {max(top_pool_size, bottom_pool_size)} scored parent skills for crossover, found {len(ranked)}"
        )
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > top_pool_size or k > bottom_pool_size:
        raise ValueError("k cannot exceed top-pool-size or bottom-pool-size")
    best_pool = ranked[:top_pool_size]
    worst_pool = ranked[-bottom_pool_size:]
    backbone = rng.choice(best_pool)
    remaining_good_pool = [row for row in best_pool if row is not backbone]
    good = rng.sample(remaining_good_pool, min(max(k - 1, 0), len(remaining_good_pool)))
    bad = rng.sample(worst_pool, k)
    return backbone, good, bad


def _copy_parent_bundle(dst_root: Path, label: str, row: dict[str, Any], score_field: str) -> None:
    parent_dir = dst_root / label
    skill_root = Path(str(row["skill_root"]))
    copy_skill_tree(skill_root, parent_dir / "skill")
    (parent_dir / "score.txt").write_text(str(row.get(score_field, 0.0)), encoding="utf-8")
    (parent_dir / "manifest.txt").write_text(
        "\n".join(skill_file_manifest(parent_dir / "skill")) + "\n",
        encoding="utf-8",
    )
    write_json(parent_dir / "metadata.json", row)


def _generate_one_child(
    *,
    child_index: int,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    out_root: Path,
) -> dict[str, Any]:
    rng = random.Random(args.seed + child_index)
    backbone, good, bad = _select_parents(
        rows,
        k=args.k,
        score_field=args.score_field,
        top_pool_size=args.top_pool_size,
        bottom_pool_size=args.bottom_pool_size,
        rng=rng,
    )
    workspace_dir = out_root / f"workspace_{child_index:03d}"
    ensure_clean_dir(workspace_dir)
    _copy_parent_bundle(workspace_dir, "backbone", backbone, args.score_field)

    for index, row in enumerate(good, start=1):
        _copy_parent_bundle(workspace_dir, f"good_{index:02d}", row, args.score_field)
    for index, row in enumerate(bad, start=1):
        _copy_parent_bundle(workspace_dir, f"bad_{index:02d}", row, args.score_field)

    parent_table = _parent_score_table(backbone, good, bad, args.score_field)
    (workspace_dir / "parent_scores.md").write_text(parent_table + "\n", encoding="utf-8")
    prompt = _prompt_text(backbone, good, bad, args.score_field)
    (workspace_dir / "crossover_prompt.txt").write_text(prompt, encoding="utf-8")
    parent_sections: list[str] = []
    for label in ["backbone", *[f"good_{i:02d}" for i in range(1, len(good) + 1)], *[f"bad_{i:02d}" for i in range(1, len(bad) + 1)]]:
        skill_path = workspace_dir / label / "skill" / "skill.md"
        if not skill_path.exists():
            skill_path = workspace_dir / label / "skill" / "SKILL.md"
        parent_sections.append(f"## Parent {label}\n{skill_path.read_text(encoding='utf-8')}")
    chat_prompt = (
        prompt
        + "\n\nParent skill contents:\n\n"
        + "\n\n".join(parent_sections)
        + "\n\nReturn only the complete child SKILL.md content. Do not use code fences and do not add commentary."
    )
    response, usage = chat_optimizer(
        system="You create concise, generalizable SKILL.md instructions for LiveMath agents.",
        user=chat_prompt,
        max_completion_tokens=int(os.environ.get("SKILL_CROSSOVER_MAX_COMPLETION_TOKENS", "6000")),
        retries=5,
        stage="livemath_skill_crossover",
        reasoning_effort=args.reasoning_effort,
        timeout=300,
    )
    skill_text = re.sub(r"^```(?:markdown|md)?\s*", "", response.strip(), flags=re.IGNORECASE)
    skill_text = re.sub(r"\s*```$", "", skill_text).strip()
    (workspace_dir / "skill").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "skill" / "skill.md").write_text(skill_text + "\n", encoding="utf-8")
    (workspace_dir / "crossover_last_message.txt").write_text(response, encoding="utf-8")
    write_json(workspace_dir / "crossover_usage.json", usage)
    ensure_skill_md(workspace_dir / "skill")
    metadata = {
        "workspace_dir": str(workspace_dir),
        "skill_root": str(workspace_dir / "skill"),
        "child_skill_root": str(workspace_dir / "skill"),
        "score_field": args.score_field,
        "top_pool_size": args.top_pool_size,
        "bottom_pool_size": args.bottom_pool_size,
        "backbone_parent": backbone,
        "good_parents": good,
        "bad_parents": bad,
    }
    write_json(workspace_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    args = parse_args()
    if args.provider == "local":
        args.model = args.model or os.environ.get("TEACHER_MODEL", "") or DEFAULT_MODEL
        args.base_url = (
            args.base_url
            or os.environ.get("TEACHER_BASE_URL", "")
            or "http://127.0.0.1:8000/v1"
        )
        args.api_key = args.api_key or os.environ.get(args.api_key_env, "")
    else:
        provider_cfg = resolve_chat_provider(args.provider)
        args.provider = provider_cfg["provider"]
        args.model = args.model or provider_cfg["model"] or DEFAULT_MODEL
        args.base_url = args.base_url or provider_cfg["base_url"] or DEFAULT_BASE_URL
        env_api_key = os.environ.get(args.api_key_env, "").strip()
        args.api_key = args.api_key or env_api_key or provider_cfg["api_key"]
    if args.base_url:
        os.environ[args.openai_base_url_env] = args.base_url
    if args.api_key:
        os.environ[args.api_key_env] = args.api_key
    print(
        f"[livemath-crossover] provider={args.provider} model={args.model} "
        f"base_url={args.base_url} api_key_env={args.api_key_env} "
        f"api_key_set={bool(args.api_key)}",
        flush=True,
    )
    set_optimizer_backend("openai_chat")
    set_optimizer_deployment(args.model)
    set_reasoning_effort(args.reasoning_effort)
    configure_azure_openai(
        endpoint=args.base_url,
        api_key=args.api_key,
        auth_mode="openai_compatible",
        optimizer_endpoint=args.base_url,
        optimizer_api_key=args.api_key,
        optimizer_auth_mode="openai_compatible",
    )
    args.eval_summary = args.eval_summary.resolve()
    rows = _load_scored_parents(args.eval_summary)
    out_root = (args.out_root / f"children{args.num_children}_k{args.k}_model_{args.model.replace('/', '_')}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    max_workers = max(1, args.parallel)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _generate_one_child,
                child_index=child_index,
                args=args,
                rows=rows,
                out_root=out_root,
            ): child_index
            for child_index in range(1, args.num_children + 1)
        }
        for future in as_completed(future_map):
            child_index = future_map[future]
            result = future.result()
            generated.append(result)
            print(f"[crossover-skill] workspace_{child_index:03d} completed")

    generated.sort(key=lambda row: row["workspace_dir"])
    summary = {
        "provider": args.provider,
        "model": args.model,
        "eval_summary": str(args.eval_summary),
        "score_field": args.score_field,
        "k": args.k,
        "top_pool_size": args.top_pool_size,
        "bottom_pool_size": args.bottom_pool_size,
        "num_children": args.num_children,
        "generated": generated,
    }
    write_json(out_root / "summary.json", summary)
    print(str(out_root))


if __name__ == "__main__":
    main()
