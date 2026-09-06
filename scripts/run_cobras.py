from __future__ import annotations

import argparse
from importlib import import_module
import json
import os
import subprocess
import sys
from pathlib import Path

from cobras.cobras_core.best_skill import materialize_best_skill
from cobras.cobras_core.config import DATASETS, get_dataset
from cobras.cobras_core.llm import child_env_for_provider, resolve_provider
from cobras.cobras_core.paths import COBRAS_ROOT, ensure_project_paths, load_default_env_files, model_slug
from cobras.cobras_core.token_usage import write_usage_reports
from cobras.cobras_core.update_schedule import add_schedule_arguments, schedule_from_args
from cobras.task_envs.cobras_types import CobrasDriverContext


LIVEMATH_EVAL_PROTOCOL = {"harness": "direct", "prompt_version": "direct"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone COBRAS for a configured dataset.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--prune-count", type=int, default=3)
    add_schedule_arguments(parser)
    parser.add_argument("--regen-sample-size", type=int, default=8)
    parser.add_argument("--rollout-sample-size", type=int, default=8)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-workers", type=int, default=15)
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--max-completion-tokens", type=int, default=0)
    parser.add_argument("--task-timeout", type=int, default=0)
    parser.add_argument("--exec-timeout", type=int, default=0)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi")
    parser.add_argument("--prompt-version", choices=["direct", "analysis"], default="direct")
    parser.add_argument("--image-detail", default="auto")
    parser.add_argument("--generate-parallel", type=int, default=3)
    parser.add_argument("--reward-field", default="")
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--lambda_", type=float, default=0.03)
    parser.add_argument("--model", default="openai/gpt-5.4-nano")
    parser.add_argument("--teacher-model", default="openai/gpt-5.4")
    parser.add_argument("--teacher-reasoning-effort", default="medium")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--teacher-provider", default="")
    parser.add_argument("--teacher-base-url", default="")
    parser.add_argument("--teacher-api-key-env", default="")
    parser.add_argument("--embedding-backend", default="openrouter")
    parser.add_argument("--embedding-model", default="qwen/qwen3-embedding-4b")
    parser.add_argument("--embedding-base-url", default="")
    parser.add_argument("--embedding-api-key-env", default="EMBEDDING_API_KEY")
    parser.add_argument("--embedding-dim", type=int, default=2560)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--initial-pool-root", type=Path, default=None)
    parser.add_argument("--initial-eval-summary", type=Path, default=None)
    parser.add_argument(
        "--skip-initial-eval",
        action="store_true",
        help="Build the initial arm manifest without evaluating every init skill.",
    )
    parser.add_argument("--regenerate-baseline-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-mode", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str]) -> None:
    completed = subprocess.run(cmd, cwd=COBRAS_ROOT, env=env, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def result_root(args: argparse.Namespace, reward_field: str) -> Path:
    if args.out_root:
        return args.out_root.resolve()
    target = model_slug(args.model)
    teacher = model_slug(args.teacher_model)
    emb = model_slug(args.embedding_model)
    schedule = schedule_from_args(args)
    exp = (
        f"lam_{args.lambda_}_nu_{args.nu}_r_{args.rounds}_emb_{emb}_"
        f"init{args.pool_size}_{schedule.tag}_drop{args.prune_count}_"
        f"sample{args.rollout_sample_size}_reward_{reward_field}"
    )
    return COBRAS_ROOT / "results" / "cobras" / args.dataset / "cobras" / f"target_{target}__teacher_{teacher}" / exp / "trial_001"


def load_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def shared_baseline_root(dataset: str, model: str, provider: str) -> Path:
    target = model_slug(model)
    provider_name = model_slug(provider)
    leaf = "train50_direct" if dataset == "livemath" else "train50"
    return (
        COBRAS_ROOT
        / "results"
        / "baselines"
        / dataset
        / f"target_{target}__provider_{provider_name}"
        / leaf
    )


def task_eval_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--mode",
        args.mode,
        "--prompt-version",
        args.prompt_version,
        "--image-detail",
        args.image_detail,
    ]
    if args.max_turns > 0:
        values.extend(["--max-turns", str(args.max_turns)])
    if args.max_completion_tokens > 0:
        values.extend(["--max-completion-tokens", str(args.max_completion_tokens)])
    if args.task_timeout > 0:
        values.extend(["--task-timeout", str(args.task_timeout)])
    if args.exec_timeout > 0:
        values.extend(["--exec-timeout", str(args.exec_timeout)])
    if args.reasoning_effort:
        values.extend(["--reasoning-effort", args.reasoning_effort])
    return values


def _assert_livemath_protocol(payload: dict, *, source: Path) -> None:
    protocol = payload.get("eval_protocol", payload)
    if not isinstance(protocol, dict) or any(
        protocol.get(key) != value for key, value in LIVEMATH_EVAL_PROTOCOL.items()
    ):
        raise SystemExit(
            "Refusing to reuse a LiveMath artifact from an incompatible legacy harness: "
            f"{source}. Start a new trial with direct-answer artifacts."
        )


def _assert_livemath_skill_protocol(skill_paths: list[Path]) -> None:
    forbidden = ("./refs/", "./task.md", "`find`", "`grep`", "local workspace tools")
    contaminated: list[str] = []
    for path in skill_paths:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(marker.lower() in text for marker in forbidden):
            contaminated.append(str(path))
    if contaminated:
        preview = "\n".join(f"- {path}" for path in contaminated[:10])
        raise SystemExit(
            "LiveMath init skills were generated for an incompatible legacy harness. "
            "Regenerate the init pool from direct-answer baseline trajectories before running COBRAS:\n"
            f"{preview}"
        )


def copy_optimized_skill(out_root: Path, reward_field: str) -> None:
    history_path = out_root / "history.jsonl"
    if not history_path.exists():
        return
    selection = materialize_best_skill(out_root, reward_field=reward_field)
    best = selection["best_arm"]
    print(
        f"[cobras] optimized_skill arm={best['arm_id']} "
        f"mean_{reward_field}={best['mean_reward']:.6f} "
        f"selections={best['selection_count']} path={out_root / 'optimized_skill' / 'skill.md'}",
        flush=True,
    )


def ensure_baseline(args: argparse.Namespace, env: dict[str, str], out_root: Path) -> Path:
    cfg = get_dataset(args.dataset)
    train_limit = getattr(args, "train_limit", None) or cfg.train_limit
    baseline_root = args.regenerate_baseline_root or shared_baseline_root(
        args.dataset,
        args.model,
        args.provider,
    )
    summary = baseline_root / "summary.json"
    if summary.exists():
        if args.dataset == "livemath":
            config_path = baseline_root / "config.json"
            if not config_path.exists():
                _assert_livemath_protocol({}, source=baseline_root)
            _assert_livemath_protocol(load_summary(config_path), source=baseline_root)
        return baseline_root
    cmd = [
        sys.executable,
        "-m",
        "cobras.scripts.run_baseline",
        "--dataset",
        args.dataset,
        "--split",
        "train",
        "--limit",
        str(train_limit),
        "--workers",
        str(args.eval_workers),
        "--model",
        args.model,
        "--provider",
        args.provider,
        "--base-url",
        args.base_url,
        "--api-key-env",
        args.api_key_env,
        "--out-root",
        str(baseline_root),
        *task_eval_args(args),
    ]
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    run(cmd, env)
    return baseline_root


def initial_pool_root(args: argparse.Namespace, cfg) -> Path:
    return (args.initial_pool_root or cfg.init_skills_root).resolve()


def ensure_initial_eval(args: argparse.Namespace, env: dict[str, str], out_root: Path) -> Path:
    cfg = get_dataset(args.dataset)
    train_limit = getattr(args, "train_limit", None) or cfg.train_limit
    summary_path = args.initial_eval_summary or (out_root / "init_eval" / "summary.json")
    if summary_path.exists():
        if args.dataset == "livemath":
            _assert_livemath_protocol(load_summary(summary_path), source=summary_path)
        return summary_path.resolve()
    skills_root = initial_pool_root(args, cfg)
    if not skills_root.exists():
        raise SystemExit(f"No init skills directory found at {skills_root}")
    skills = sorted(path for path in skills_root.iterdir() if (path / "skill.md").exists())
    if not skills:
        raise SystemExit(f"No init skills found under {skills_root}")
    if args.dataset == "livemath":
        _assert_livemath_skill_protocol([path / "skill.md" for path in skills[: args.pool_size]])
    if args.skip_initial_eval:
        rows = [
            {
                "skill_name": skill_dir.name,
                "workspace_dir": str(skill_dir.resolve()),
                "skill_root": str(skill_dir.resolve()),
                "count": None,
                "summary_path": "",
            }
            for skill_dir in skills[: args.pool_size]
        ]
        payload: dict[str, object] = {
            "results": rows,
            "evaluation_skipped": True,
            "note": "Initial skills are unevaluated; COBRAS history starts empty.",
        }
        if args.dataset == "livemath":
            payload["eval_protocol"] = LIVEMATH_EVAL_PROTOCOL
        write_json(summary_path, payload)
        print(
            f"[cobras] initial eval skipped; manifest={summary_path} arms={len(rows)}",
            flush=True,
        )
        return summary_path.resolve()
    rows = []
    init_root = summary_path.parent
    for index, skill_dir in enumerate(skills[: args.pool_size], start=1):
        eval_root = init_root / f"skill_{index:03d}"
        cmd = [
            sys.executable,
            "-m",
            "cobras.scripts.run_baseline",
            "--dataset",
            args.dataset,
            "--split",
            "train",
            "--limit",
            str(train_limit),
            "--workers",
            str(args.eval_workers),
            "--model",
            args.model,
            "--provider",
            args.provider,
            "--base-url",
            args.base_url,
            "--api-key-env",
            args.api_key_env,
            "--skill-path",
            str(skill_dir / "skill.md"),
            "--out-root",
            str(eval_root),
            *task_eval_args(args),
        ]
        if args.api_key:
            cmd.extend(["--api-key", args.api_key])
        run(cmd, env)
        payload = load_summary(eval_root / "summary.json")
        hard = payload.get("hard_acc", payload.get("avg_hard_reward", 0.0))
        soft = payload.get("avg_soft", payload.get("avg_soft_reward", hard))
        rows.append(
            {
                "skill_name": skill_dir.name,
                "workspace_dir": str(skill_dir.resolve()),
                "skill_root": str(skill_dir.resolve()),
                "avg_soft_reward": float(soft or 0.0),
                "avg_hard_reward": float(hard or 0.0),
                "count": payload.get("count", payload.get("n")),
                "summary_path": str((eval_root / "summary.json").resolve()),
            }
        )
    payload: dict[str, object] = {"results": rows}
    if args.dataset == "livemath":
        payload["eval_protocol"] = LIVEMATH_EVAL_PROTOCOL
    write_json(summary_path, payload)
    return summary_path.resolve()


def main() -> None:
    ensure_project_paths()
    args = parse_args()
    load_default_env_files(args.dataset)
    cfg = get_dataset(args.dataset)
    reward_field = args.reward_field or cfg.reward_field
    provider_cfg = resolve_provider(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    teacher_cfg = resolve_provider(
        provider=args.teacher_provider or args.provider,
        model=args.teacher_model,
        base_url=args.teacher_base_url or args.base_url,
        api_key_env=args.teacher_api_key_env or args.api_key_env,
    )
    out_root = result_root(args, reward_field)
    env = os.environ.copy()
    teacher_api_key = env.get(teacher_cfg.api_key_env, "").strip()
    env.update(child_env_for_provider(provider_cfg, api_key=args.api_key))
    env["TEACHER_MODEL"] = teacher_cfg.model
    env["TEACHER_BASE_URL"] = teacher_cfg.base_url
    if teacher_api_key:
        env[teacher_cfg.api_key_env] = teacher_api_key
        env["TEACHER_API_KEY"] = teacher_api_key
    env["PYTHONPATH"] = (
        f"{COBRAS_ROOT.parent}:{COBRAS_ROOT}:{env.get('PYTHONPATH', '')}"
    )
    env["PROVIDER"] = provider_cfg.provider
    if teacher_cfg.base_url:
        env["OPENAI_BASE_URL"] = teacher_cfg.base_url
        env["SKILL_GEN_BASE_URL"] = teacher_cfg.base_url
        env["SKILL_MUTATE_BASE_URL"] = teacher_cfg.base_url
    env["SKILL_GEN_MODEL"] = teacher_cfg.model
    env["SKILL_MUTATE_MODEL"] = teacher_cfg.model
    env["COBRAS_TOKEN_LOG_PATH"] = str((out_root / "token_usage" / "events.jsonl").resolve())
    env["COBRAS_ROUND"] = "setup"
    env["COBRAS_STUDENT_MODEL"] = provider_cfg.model
    env["COBRAS_TEACHER_MODEL"] = teacher_cfg.model
    env["COBRAS_STUDENT_PROVIDER"] = provider_cfg.provider
    env["COBRAS_TEACHER_PROVIDER"] = teacher_cfg.provider
    if args.api_key:
        env[teacher_cfg.api_key_env] = args.api_key
    # Keep secrets in environment variables. Dataset drivers only need the
    # variable name and must never expose API keys in process arguments.
    args.teacher_api_key = ""

    baseline_root = ensure_baseline(args, env, out_root)
    initial_eval_summary = ensure_initial_eval(args, env, out_root)
    init_pool_root = initial_pool_root(args, cfg)
    write_usage_reports(out_root, initial_pool_root=init_pool_root)
    common = [
        "--initial-eval-summary",
        str(initial_eval_summary),
        "--regenerate-baseline-root",
        str(baseline_root),
        "--out-root",
        str(out_root),
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
        "--eval-limit",
        str(args.eval_limit or args.train_limit or cfg.train_limit),
        "--eval-workers",
        str(args.eval_workers),
        "--generate-parallel",
        str(args.generate_parallel),
        "--reward-field",
        reward_field,
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
        "--base-url",
        provider_cfg.base_url,
        "--api-key-env",
        provider_cfg.api_key_env,
        "--teacher-base-url",
        teacher_cfg.base_url,
        "--teacher-api-key-env",
        teacher_cfg.api_key_env,
        "--seed",
        str(args.seed),
        "--prompt-version",
        args.prompt_version,
    ]
    if args.prune_cooldown_rounds is not None:
        common.extend(["--prune-cooldown-rounds", str(args.prune_cooldown_rounds)])
    if args.prune_max_interval is not None:
        common.extend(["--prune-max-interval", str(args.prune_max_interval)])
    if args.resume:
        common.append("--resume")
    if args.api_key:
        common.extend(["--api-key", args.api_key])
    if args.test_mode:
        common.append("--test-mode")

    driver = import_module(cfg.cobras_driver_module)
    build_command = getattr(driver, "build_command", None)
    if not callable(build_command):
        raise RuntimeError(f"{cfg.cobras_driver_module} must export build_command(context)")
    context = CobrasDriverContext(
        args=args,
        dataset=cfg,
        provider=provider_cfg,
        teacher_provider=teacher_cfg,
        common_args=tuple(common),
        out_root=out_root,
        baseline_root=baseline_root,
        initial_eval_summary=initial_eval_summary,
        initial_pool_root=init_pool_root,
    )
    try:
        run(build_command(context), env)
        copy_optimized_skill(out_root, reward_field)
    finally:
        usage_summary = write_usage_reports(out_root, initial_pool_root=init_pool_root)
        print(
            f"[cobras] token_usage={out_root / 'token_usage' / 'summary.json'} "
            f"events={usage_summary['event_count']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
