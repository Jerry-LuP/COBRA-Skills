from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cobras.cobras_core.runner import COBRASRunner
from cobras.cobras_core.run_state import ArmState
from cobras.cobras_core.update_schedule import add_schedule_arguments
from cobras.scripts.embeddings import DEFAULT_OPENROUTER_BASE_URL, DEFAULT_OPENROUTER_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the shared COBRAS loop with NN + LinearUCB.")
    parser.add_argument("--initial-pool-root", type=Path, required=True)
    parser.add_argument("--initial-eval-summary", type=Path, required=True)
    parser.add_argument("--regenerate-baseline-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--prune-count", type=int, default=3)
    parser.add_argument("--crossover-threshold", type=int, default=8)
    parser.add_argument("--train-rollout-tasks", type=int, default=6)
    parser.add_argument("--train-rollout-parallel", type=int, default=24)
    parser.add_argument("--eval-parallel", type=int, default=24)
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--selected-eval-limit", type=int, default=0)
    parser.add_argument("--selected-eval-split", type=str, default="val")
    parser.add_argument("--selected-eval-split-json", type=Path, default=None)
    parser.add_argument("--generate-parallel", type=int, default=3)
    add_schedule_arguments(parser)
    parser.add_argument("--crossover-num-children", type=int, default=1)
    parser.add_argument("--crossover-parallel", type=int, default=1)
    parser.add_argument("--crossover-k", type=int, default=2)
    parser.add_argument("--crossover-top-pool-size", type=int, default=4)
    parser.add_argument("--crossover-bottom-pool-size", type=int, default=4)
    parser.add_argument("--crossover-score-field", type=str, default="avg_soft_reward")
    parser.add_argument("--regen-sample-size", type=int, default=6)
    parser.add_argument("--regen-min-success", type=int, default=1)
    parser.add_argument("--regen-min-fail", type=int, default=1)
    parser.add_argument("--rollout-sample-size", type=int, default=6)
    parser.add_argument("--rollout-sample-min-success", type=int, default=1)
    parser.add_argument("--rollout-sample-min-fail", type=int, default=1)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--lambda_", type=float, default=0.03)
    parser.add_argument("--mlp-l2", type=float, default=1e-4)
    parser.add_argument("--embedding-backend", type=str, default="openrouter", choices=["openrouter", "hash"])
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--embedding-base-url", type=str, default=DEFAULT_OPENROUTER_BASE_URL)
    parser.add_argument("--embedding-api-key-env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--embedding-dim", type=int, default=2560)
    parser.add_argument("--model", type=str, default="gpt-5.5")
    providers = ["openrouter", "yunwu", "custom", "local"]
    parser.add_argument("--provider", type=str, choices=providers, default="yunwu")
    parser.add_argument("--teacher-model", type=str, default="")
    parser.add_argument("--teacher-provider", type=str, choices=providers, default="")
    parser.add_argument("--teacher-reasoning-effort", type=str, default="medium")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--teacher-base-url", type=str, default="")
    parser.add_argument("--teacher-api-key-env", type=str, default="")
    parser.add_argument("--teacher-api-key", type=str, default="")
    parser.add_argument("--openai-base-url-env", type=str, default="OPENAI_BASE_URL")
    parser.add_argument("--task-package", type=str, default="scripts")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--reward-field", type=str, choices=["hard", "soft"], default="soft")
    parser.add_argument("--prompt-version", type=str, default="direct", choices=["direct", "analysis"])
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main() -> None:
    runner = COBRASRunner(**vars(parse_args()))
    runner.run()


__all__ = ["ArmState", "COBRASRunner", "build_parser", "main", "parse_args"]


if __name__ == "__main__":
    main()
