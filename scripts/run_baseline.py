from __future__ import annotations

import argparse
import os
from pathlib import Path

from cobras.cobras_core.config import DATASETS, get_dataset
from cobras.cobras_core.llm import resolve_provider
from cobras.cobras_core.paths import COBRAS_ROOT, ensure_project_paths, load_default_env_files, model_slug
from cobras.task_envs.eval_types import EvalConfig
from cobras.task_envs.registry import load_evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a COBRAS dataset baseline or skill-injected eval.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--model", default="")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--skill-path", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=int(os.environ.get("COBRAS_TARGET_MAX_COMPLETION_TOKENS", "0") or 0),
    )
    parser.add_argument("--task-timeout", type=int, default=0)
    parser.add_argument("--exec-timeout", type=int, default=0)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--mode", default="multi", choices=["single", "multi"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-detail", default="auto")
    parser.add_argument("--use-eval-feedback", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--use-theorem", action="store_true")
    parser.add_argument("--use-sketch", action="store_true")
    parser.add_argument("--prompt-version", default="direct", choices=["direct", "analysis"])
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    return parser.parse_args()


def default_out_root(args: argparse.Namespace, model: str) -> Path:
    cfg = get_dataset(args.dataset)
    limit = args.limit or (cfg.train_limit if args.split == "train" else cfg.test_limit)
    kind = "skill_eval" if args.skill_path else "baseline"
    skill_slug = ""
    if args.skill_path:
        skill_slug = "_skill_" + model_slug(args.skill_path.parent.name if args.skill_path.name == "skill.md" else args.skill_path.stem)
    return (
        COBRAS_ROOT
        / "results"
        / "cobras"
        / cfg.name
        / kind
        / f"target_{model_slug(model)}"
        / f"{args.split}{limit}{skill_slug}"
        / "trial_001"
    )


def main() -> None:
    ensure_project_paths()
    os.chdir(COBRAS_ROOT)
    args = parse_args()
    load_default_env_files(args.dataset)
    cfg = get_dataset(args.dataset)
    provider_cfg = resolve_provider(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    limit = args.limit or (cfg.train_limit if args.split == "train" else cfg.test_limit)
    out_root = (args.out_root or default_out_root(args, provider_cfg.model)).resolve()
    skill_path = args.skill_path.resolve() if args.skill_path else None
    eval_config = EvalConfig(
        dataset=cfg,
        provider=provider_cfg,
        split=args.split,
        limit=limit,
        workers=args.workers,
        out_root=out_root,
        skill_path=skill_path,
        api_key=args.api_key,
        max_turns=args.max_turns,
        max_completion_tokens=args.max_completion_tokens,
        task_timeout=args.task_timeout,
        reasoning_effort=args.reasoning_effort,
        mode=args.mode,
        seed=args.seed,
        options={
            "exec_timeout": args.exec_timeout or None,
            "image_detail": args.image_detail,
            "use_eval_feedback": args.use_eval_feedback,
            "trials": args.trials,
            "use_theorem": args.use_theorem,
            "use_sketch": args.use_sketch,
            "prompt_version": args.prompt_version,
            "data_path": str(args.data_path.resolve()) if args.data_path else "",
            "sample_seed": args.sample_seed,
        },
    )
    artifacts = load_evaluator(args.dataset)(eval_config)
    artifacts.validate()


if __name__ == "__main__":
    main()
