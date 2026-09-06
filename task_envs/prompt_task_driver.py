from __future__ import annotations

import sys

from cobras.task_envs.cobras_types import CobrasDriverContext


def build_prompt_task_command(
    context: CobrasDriverContext,
    *,
    crossover_score_field: str = "ucb_score",
) -> list[str]:
    """Map the unified CLI context onto the shared COBRAS core runner."""

    args = context.args
    cfg = context.dataset
    teacher_reasoning_effort = getattr(args, "teacher_reasoning_effort", "medium")
    embedding_base_url = getattr(args, "embedding_base_url", "")
    embedding_api_key_env = getattr(args, "embedding_api_key_env", "EMBEDDING_API_KEY")
    command = [
        sys.executable,
        "-m",
        "cobras.scripts.cobras_dynamic",
        "--dataset",
        cfg.name,
        "--initial-pool-root",
        str(context.initial_pool_root),
        "--initial-eval-summary",
        str(context.initial_eval_summary),
        "--regenerate-baseline-root",
        str(context.baseline_root),
        "--out-root",
        str(context.out_root),
        "--rounds",
        str(args.rounds),
        "--pool-size",
        str(args.pool_size),
        "--prune-count",
        str(args.prune_count),
        "--prune-interval",
        str(args.prune_interval),
        "--prune-schedule",
        args.prune_schedule,
        "--prune-log-threshold",
        str(args.prune_log_threshold),
        "--regen-sample-size",
        str(args.regen_sample_size),
        "--rollout-sample-size",
        str(args.rollout_sample_size),
        "--eval-parallel",
        str(args.eval_workers),
        "--selected-eval-split",
        "train",
        "--selected-eval-limit",
        str(args.eval_limit or cfg.train_limit),
        "--generate-parallel",
        str(args.generate_parallel),
        "--reward-field",
        args.reward_field or cfg.reward_field,
        "--crossover-score-field",
        crossover_score_field,
        "--nu",
        str(args.nu),
        "--lambda_",
        str(args.lambda_),
        "--embedding-backend",
        args.embedding_backend,
        "--embedding-model",
        args.embedding_model,
        "--embedding-dim",
        str(args.embedding_dim),
        "--model",
        context.provider.model,
        "--provider",
        context.provider.provider,
        "--base-url",
        context.provider.base_url,
        "--api-key-env",
        context.provider.api_key_env,
        "--teacher-model",
        context.teacher_provider.model,
        "--teacher-provider",
        context.teacher_provider.provider,
        "--teacher-base-url",
        context.teacher_provider.base_url,
        "--teacher-api-key-env",
        context.teacher_provider.api_key_env,
        "--teacher-reasoning-effort",
        teacher_reasoning_effort,
        "--seed",
        str(args.seed),
    ]
    if embedding_base_url:
        command.extend(["--embedding-base-url", embedding_base_url])
    if embedding_api_key_env:
        command.extend(["--embedding-api-key-env", embedding_api_key_env])
    if args.max_turns > 0:
        command.extend(["--max-turns", str(args.max_turns)])

    teacher_api_key = getattr(args, "teacher_api_key", "")
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    if teacher_api_key:
        command.extend(["--teacher-api-key", teacher_api_key])
    if args.prune_cooldown_rounds is not None:
        command.extend(["--prune-cooldown-rounds", str(args.prune_cooldown_rounds)])
    if args.prune_max_interval is not None:
        command.extend(["--prune-max-interval", str(args.prune_max_interval)])
    if args.resume:
        command.append("--resume")
    if args.test_mode:
        command.append("--test-mode")
    return command
