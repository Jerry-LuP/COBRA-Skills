from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping

import yaml

from cobras.cobras_core.paths import COBRAS_ROOT
from cobras.cobras_core.token_usage import write_usage_reports
from cobras.task_envs.eval_errors import is_valid_result_row
from cobras.scripts.release_config import (
    DATASETS,
    HARNESS_BACKENDS,
    ExperimentConfig,
    load_runtime_env,
    resolve_config,
    trial_name,
    write_json,
)


DEFAULT_CONFIG = COBRAS_ROOT / "configs" / "default.yaml"
DEFAULT_ENV = COBRAS_ROOT / ".env"


def _yaml_value(value: str) -> Any:
    return yaml.safe_load(value)


def _set_overrides(values: Iterable[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--set expects KEY=VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--set key cannot be empty")
        overrides[key] = _yaml_value(raw)
    return overrides


def _parse_trials(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    resolved: list[int] = []
    for value in values:
        for part in value.split(","):
            if not part.strip():
                continue
            trial = int(part)
            if trial < 1:
                raise ValueError("Trial numbers must be positive")
            resolved.append(trial)
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate trial numbers are not allowed")
    return resolved


def _parse_datasets(positional: list[str], option: list[str] | None) -> list[str] | None:
    values = list(positional)
    for item in option or []:
        values.extend(part.strip() for part in item.split(",") if part.strip())
    if not values:
        return None
    unknown = sorted(set(values) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unsupported datasets: {', '.join(unknown)}")
    return list(dict.fromkeys(values))


def _config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides = _set_overrides(args.set_values)
    direct = {
        "experiment.name": args.run_name,
        "experiment.output_root": args.output_root,
        "experiment.seed": args.seed,
        "experiment.resume": args.resume,
        "experiment.initial_eval": args.initial_eval,
        "target.model": args.target_model,
        "target.provider": args.target_provider,
        "target.reasoning_effort": args.target_reasoning_effort,
        "target.enable_thinking": args.target_enable_thinking,
        "teacher.model": args.teacher_model,
        "teacher.provider": args.teacher_provider,
        "teacher.reasoning_effort": args.teacher_reasoning_effort,
        "teacher.max_completion_tokens": args.teacher_max_tokens,
        "embedding.backend": args.embedding_backend,
        "embedding.model": args.embedding_model,
        "embedding.dimension": args.embedding_dimension,
        "harness.backend": args.harness,
        "harness.timeout_seconds": args.harness_timeout,
        "algorithm.rounds": args.rounds,
        "algorithm.pool_size": args.pool_size,
        "algorithm.prune_count": args.prune_count,
        "algorithm.nu": args.nu,
        "algorithm.lambda": args.lambda_,
        "algorithm.eval_limit": args.train_size,
        "init_skills.count": args.init_count,
        "init_skills.sample_size": args.init_sample_size,
        "init_skills.parallel": args.init_parallel,
    }
    for key, value in direct.items():
        if value is not None:
            overrides[key] = value

    datasets = _parse_datasets(args.dataset, args.datasets)
    trials = _parse_trials(args.trials)
    if datasets is not None:
        overrides["experiment.datasets"] = datasets
    if trials is not None:
        overrides["experiment.trials"] = trials
    selected = datasets or []
    if args.workers is not None:
        for dataset in selected or DATASETS:
            overrides[f"datasets.{dataset}.workers"] = args.workers
    if args.train_size is not None:
        for dataset in selected or DATASETS:
            overrides[f"datasets.{dataset}.train_size"] = args.train_size
    if args.test_size is not None:
        for dataset in selected or DATASETS:
            overrides[f"datasets.{dataset}.test_size"] = args.test_size
    if args.target_max_tokens is not None:
        overrides["target.max_completion_tokens"] = args.target_max_tokens
        for dataset in selected or DATASETS:
            overrides[f"datasets.{dataset}.max_completion_tokens"] = args.target_max_tokens
    return overrides


def _endpoint_default(provider: str, role: str) -> str:
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    if provider == "yunwu":
        return "https://yunwu.ai/v1"
    if provider == "local":
        return "http://127.0.0.1:8000/v1"
    if role == "embedding":
        return "https://openrouter.ai/api/v1"
    return ""


def _identity_diff(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                differences.append(f"{path}: unexpected {actual[key]!r}")
            elif key not in actual:
                differences.append(f"{path}: missing (expected {expected[key]!r})")
            else:
                differences.extend(_identity_diff(expected[key], actual[key], path))
        return differences
    if expected != actual:
        return [f"{prefix}: expected {expected!r}, found {actual!r}"]
    return []


def _score(summary_path: Path) -> tuple[float | None, float | None]:
    if not summary_path.exists():
        return None, None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    hard = payload.get("hard_acc", payload.get("avg_hard_reward"))
    soft = payload.get("avg_soft", payload.get("avg_soft_reward", hard))
    return (
        None if hard is None else float(hard),
        None if soft is None else float(soft),
    )


def _evaluation_artifacts_complete(
    root: Path,
    results_file: str,
    *,
    expected: int,
) -> bool:
    summary_path = root / "summary.json"
    results_path = root / results_file
    if not summary_path.exists() or not results_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if len(rows) != expected or any(
        not isinstance(row, dict) or "id" not in row or not is_valid_result_row(row)
        for row in rows
    ):
        return False
    keys = {
        (str(row["id"]), int(row.get("trial_index") or 0))
        for row in rows
    }
    if len(keys) != expected:
        return False
    reported = summary.get("count", summary.get("n"))
    if reported is not None:
        try:
            if int(reported) != expected:
                return False
        except (TypeError, ValueError):
            return False
    return True


class Pipeline:
    def __init__(self, config: ExperimentConfig, *, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run
        self.env = load_runtime_env(config.env_path)
        source_path = str(COBRAS_ROOT)
        current_pythonpath = self.env.get("PYTHONPATH", "")
        self.env["PYTHONPATH"] = source_path + (f":{current_pythonpath}" if current_pythonpath else "")
        self.target_url = self.env.get("TARGET_BASE_URL", "").strip() or _endpoint_default(config.target.provider, "target")
        self.teacher_url = self.env.get("TEACHER_BASE_URL", "").strip() or _endpoint_default(config.teacher.provider, "teacher")
        self.embedding_url = self.env.get("EMBEDDING_BASE_URL", "").strip() or _endpoint_default(config.embedding.backend, "embedding")
        self._configure_environment()

    def _configure_environment(self) -> None:
        cfg = self.config
        env = self.env
        env.update(
            {
                "TARGET_MODEL": cfg.target.model,
                "TARGET_BASE_URL": self.target_url,
                "TARGET_API_KEY": env.get("TARGET_API_KEY", "dummy" if cfg.target.provider == "local" else ""),
                "TEACHER_MODEL": cfg.teacher.model,
                "TEACHER_BASE_URL": self.teacher_url,
                "EMBEDDING_MODEL": cfg.embedding.model,
                "EMBEDDING_BASE_URL": self.embedding_url,
                "OPENAI_TARGET_TEMPERATURE": str(cfg.target.temperature),
                "OPENAI_CHAT_TEMPLATE_ENABLE_THINKING": str(cfg.target.enable_thinking).lower(),
                "TARGET_ENABLE_THINKING": str(cfg.target.enable_thinking).lower(),
                "COBRAS_TARGET_REASONING_EFFORT": cfg.target.reasoning_effort,
                "SKILL_GEN_REASONING_EFFORT": cfg.teacher.reasoning_effort,
                "SKILL_MUTATE_REASONING_EFFORT": cfg.teacher.reasoning_effort,
                "SKILL_CROSSOVER_REASONING_EFFORT": cfg.teacher.reasoning_effort,
                "TEACHER_MAX_COMPLETION_TOKENS": str(cfg.teacher.max_completion_tokens),
                "SKILL_GEN_MAX_COMPLETION_TOKENS": str(cfg.teacher.max_completion_tokens),
                "SKILL_MUTATE_MAX_COMPLETION_TOKENS": str(cfg.teacher.max_completion_tokens),
                "MUTATION_MAX_COMPLETION_TOKENS": str(cfg.teacher.max_completion_tokens),
                "SKILL_CROSSOVER_MAX_COMPLETION_TOKENS": str(cfg.teacher.max_completion_tokens),
                "COBRAS_STUDENT_MODEL": cfg.target.model,
                "COBRAS_TEACHER_MODEL": cfg.teacher.model,
                "COBRAS_STUDENT_PROVIDER": cfg.target.provider,
                "COBRAS_TEACHER_PROVIDER": cfg.teacher.provider,
                "COBRAS_EXEC_ALLOWED_ROOT": str(cfg.output_root),
                "EXEC_EMPTY_RESPONSE_RETRIES": str(cfg.harness.empty_response_retries),
                "ALFWORLD_EXEC_SESSION_MODE": "episode",
            }
        )
        if cfg.teacher.provider == "local":
            env.setdefault("TEACHER_API_KEY", "dummy")
        if cfg.embedding.backend == "hash":
            env.setdefault("EMBEDDING_API_KEY", "")
        if cfg.harness.backend == "native":
            env["TARGET_BACKEND"] = "openai_chat"
        elif cfg.harness.backend == "codex":
            codex = cfg.harness.codex
            env.update(
                {
                    "TARGET_BACKEND": "codex_exec",
                    "CODEX_EXEC_PATH": self._executable(codex.get("executable", "codex")),
                    "CODEX_EXEC_USE_SDK": "cli",
                    "CODEX_EXEC_SANDBOX": str(codex.get("sandbox", "read-only")),
                    "CODEX_EXEC_REASONING_EFFORT": cfg.target.reasoning_effort or "none",
                    "CODEX_EXEC_NETWORK_ACCESS": str(bool(codex.get("network_access", False))).lower(),
                    "CODEX_EXEC_WEB_SEARCH": str(bool(codex.get("web_search", False))).lower(),
                    "CODEX_EXEC_APPROVAL_POLICY": str(codex.get("approval_policy", "never")),
                    "CODEX_EXEC_CONTEXT_WINDOW": str(cfg.harness.context_window),
                    "COBRAS_CODEX_ALLOWED_ROOT": str(cfg.output_root),
                    "COBRAS_CODEX_ROOTFS": self._project_path(
                        codex.get("rootfs", ".codex_chroot/rootfs")
                    ),
                }
            )
        else:
            claude = cfg.harness.claude_code
            env.update(
                {
                    "TARGET_BACKEND": "claude_code_exec",
                    "CLAUDE_CODE_EXEC_PATH": self._executable(claude.get("executable", "claude")),
                    "CLAUDE_CODE_EXEC_USE_SDK": str(claude.get("mode", "cli")),
                    "CLAUDE_CODE_EXEC_EFFORT": cfg.target.reasoning_effort or "none",
                    "CLAUDE_CODE_EXEC_MAX_THINKING_TOKENS": str(int(claude.get("max_thinking_tokens", 0))),
                    "CLAUDE_CODE_EXEC_CONTEXT_WINDOW": str(cfg.harness.context_window),
                    "CLAUDE_CODE_EXEC_TIMEOUT_SECONDS": str(cfg.harness.timeout_seconds),
                }
            )

    @staticmethod
    def _executable(value: Any) -> str:
        path = str(value).strip()
        if not path:
            raise ValueError("Harness executable cannot be empty")
        candidate = Path(path).expanduser()
        if not candidate.is_absolute() and "/" in path:
            candidate = COBRAS_ROOT / candidate
        return str(candidate.resolve()) if candidate.is_absolute() or "/" in path else path

    @staticmethod
    def _project_path(value: Any) -> str:
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = COBRAS_ROOT / candidate
        return str(candidate.resolve())

    def _validate_harness(self) -> None:
        if self.config.harness.backend != "codex":
            return
        executable = self.env["CODEX_EXEC_PATH"]
        resolved = Path(executable) if "/" in executable else Path(shutil.which(executable) or "")
        if not resolved.is_file():
            raise RuntimeError(f"Codex executable not found: {executable}")
        if resolved.name != "codex_chroot_exec.py":
            return
        if os.geteuid() != 0:
            raise RuntimeError("The Codex chroot harness must run as root on Linux")
        marker = Path(self.env["COBRAS_CODEX_ROOTFS"]) / ".cobras-codex-rootfs.json"
        if not marker.is_file():
            setup = COBRAS_ROOT / "scripts" / "harness_scripts" / "setup_codex_chroot.py"
            raise RuntimeError(
                f"Codex chroot is not initialized. Run: {sys.executable} {setup}"
            )

    def _task_timeouts(self, dataset: str) -> tuple[int, int]:
        task = self.config.dataset(dataset)
        if self.config.harness.backend == "native":
            return task.task_timeout, task.exec_timeout
        exec_timeout = self.config.harness.timeout_seconds
        return max(task.task_timeout, exec_timeout + 60), exec_timeout

    def _validate_credentials(self, *, target: bool, teacher: bool, embedding: bool) -> None:
        if self.dry_run:
            return
        if target:
            self._validate_harness()
        missing: list[str] = []
        if target:
            if not self.target_url:
                missing.append("TARGET_BASE_URL")
            if self.config.target.provider != "local" and not self.env.get("TARGET_API_KEY", "").strip():
                missing.append("TARGET_API_KEY")
        if teacher:
            if not self.teacher_url:
                missing.append("TEACHER_BASE_URL")
            if self.config.teacher.provider != "local" and not self.env.get("TEACHER_API_KEY", "").strip():
                missing.append("TEACHER_API_KEY")
        if embedding and self.config.embedding.backend != "hash":
            if not self.embedding_url:
                missing.append("EMBEDDING_BASE_URL")
            if not self.env.get("EMBEDDING_API_KEY", "").strip():
                missing.append("EMBEDDING_API_KEY")
        if missing:
            raise RuntimeError(
                f"Missing runtime variables in {self.config.env_path} or process environment: "
                + ", ".join(missing)
            )

    def _stage_env(self, dataset: str, token_root: Path, *, round_name: str) -> dict[str, str]:
        task = self.config.dataset(dataset)
        task_timeout, exec_timeout = self._task_timeouts(dataset)
        env = self.env.copy()
        env.update(
            {
                "COBRAS_TOKEN_LOG_PATH": str(token_root / "token_usage" / "events.jsonl"),
                "COBRAS_ROUND": round_name,
                "COBRAS_TARGET_MAX_COMPLETION_TOKENS": str(task.max_completion_tokens),
                "COBRAS_TARGET_TASK_TIMEOUT_SECONDS": str(task_timeout),
                "ALFWORLD_REQUEST_TIMEOUT": str(exec_timeout),
                "LIVEMATH_EXEC_TIMEOUT": str(exec_timeout),
                "CLAUDE_CODE_EXEC_MAX_TOKENS": str(task.max_completion_tokens),
            }
        )
        return env

    def _run(self, command: list[str], *, env: dict[str, str], label: str) -> None:
        rendered = shlex.join(command)
        print(f"[cobra-skills] {label}\n  {rendered}", flush=True)
        if self.dry_run:
            return
        completed = subprocess.run(command, cwd=COBRAS_ROOT, env=env, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"{label} failed with exit code {completed.returncode}")

    def _ensure_manifest(self, path: Path, identity: dict[str, Any]) -> None:
        if self.dry_run:
            return
        if path.exists():
            actual = json.loads(path.read_text(encoding="utf-8"))
            differences = _identity_diff(identity, actual)
            if differences:
                preview = "\n".join(f"- {row}" for row in differences[:20])
                raise ValueError(
                    f"Artifact parameters do not match {path}:\n{preview}\n"
                    "Use a different experiment name/output root or remove that artifact."
                )
            return
        write_json(path, identity)

    def _resume_or_raise(self, complete: bool, path: Path, label: str) -> bool:
        if not complete:
            return False
        if self.config.resume:
            print(f"[cobra-skills] resume {label}: {path}", flush=True)
            return True
        raise FileExistsError(f"{label} already exists at {path}; enable resume or use a new output root")

    def _eval_args(self, dataset: str, *, split: str, limit: int, out_root: Path, seed: int) -> list[str]:
        task = self.config.dataset(dataset)
        task_timeout, exec_timeout = self._task_timeouts(dataset)
        args = [
            sys.executable,
            "-m",
            "cobras.scripts.run_baseline",
            "--dataset",
            dataset,
            "--split",
            split,
            "--limit",
            str(limit),
            "--workers",
            str(task.workers),
            "--model",
            self.config.target.model,
            "--provider",
            self.config.target.provider,
            "--base-url",
            self.target_url,
            "--api-key-env",
            "TARGET_API_KEY",
            "--out-root",
            str(out_root),
            "--max-completion-tokens",
            str(task.max_completion_tokens),
            "--task-timeout",
            str(task_timeout),
            "--exec-timeout",
            str(exec_timeout),
            "--reasoning-effort",
            task.reasoning_effort,
            "--mode",
            task.mode,
            "--seed",
            str(seed),
            "--image-detail",
            task.image_detail,
            "--prompt-version",
            task.prompt_version,
        ]
        if task.max_turns > 0:
            args.extend(["--max-turns", str(task.max_turns)])
        return args

    def baseline(self, dataset: str) -> Path:
        self._validate_credentials(target=True, teacher=False, embedding=False)
        cfg = self.config
        task = cfg.dataset(dataset)
        root = cfg.baseline_root(dataset)
        identity = {
            "stage": "baseline",
            "dataset": dataset,
            "target": cfg.identity(dataset)["target"],
            "harness": cfg.harness.backend,
            "task": task.__dict__,
        }
        self._ensure_manifest(root / "request.json", identity)
        complete = _evaluation_artifacts_complete(
            root,
            task.results_file,
            expected=task.train_size,
        )
        if self._resume_or_raise(complete, root, f"baseline {dataset}"):
            return root
        command = self._eval_args(dataset, split="train", limit=task.train_size, out_root=root, seed=cfg.seed)
        self._run(command, env=self._stage_env(dataset, root, round_name="baseline"), label=f"baseline dataset={dataset}")
        if not self.dry_run:
            write_usage_reports(root)
        return root

    def init(self, dataset: str, trial: int) -> Path:
        self._validate_credentials(target=False, teacher=True, embedding=False)
        cfg = self.config
        task = cfg.dataset(dataset)
        baseline_root = cfg.baseline_root(dataset)
        results_path = baseline_root / task.results_file
        if not self.dry_run and not _evaluation_artifacts_complete(
            baseline_root,
            task.results_file,
            expected=task.train_size,
        ):
            raise RuntimeError(
                f"Baseline is missing, incomplete, or contains runtime failures: "
                f"{baseline_root}. Run `cobras baseline` again."
            )
        root = cfg.init_root(dataset, trial)
        identity = {
            "stage": "init_skills",
            "dataset": dataset,
            "trial": trial,
            "target": cfg.identity(dataset)["target"],
            "teacher": cfg.identity(dataset)["teacher"],
            "harness": cfg.harness.backend,
            "init_skills": cfg.init_skills.__dict__,
            "baseline": str(baseline_root),
        }
        self._ensure_manifest(root / "request.json", identity)
        skills = [root / f"seed_{index:02d}" / "skill.md" for index in range(1, cfg.init_skills.count + 1)]
        if self._resume_or_raise(all(path.exists() for path in skills), root, f"init skills {dataset} {trial_name(trial)}"):
            return root
        sample_seed = cfg.init_skills.sample_seed + trial - 1
        generation_root = root
        if cfg.init_skills.count == 1:
            generation_root = root / "seed_01"
        command = [
            sys.executable,
            "-m",
            "cobras.scripts.generate_skill_from_results",
            "--dataset",
            dataset,
            "--run-root",
            str(baseline_root),
            "--results-path",
            str(results_path),
            "--out-dir",
            str(generation_root),
            "--sample-size",
            str(cfg.init_skills.sample_size),
            "--sample-seed",
            str(sample_seed),
            "--num-skills",
            str(cfg.init_skills.count),
            "--parallel",
            str(cfg.init_skills.parallel),
            "--model",
            cfg.teacher.model,
            "--provider",
            cfg.teacher.provider,
            "--base-url",
            self.teacher_url,
            "--api-key-env",
            "TEACHER_API_KEY",
            "--reasoning-effort",
            cfg.init_skills.reasoning_effort,
        ]
        if cfg.resume:
            command.append("--resume")
        self._run(command, env=self._stage_env(dataset, root, round_name="init_skill_generation"), label=f"init dataset={dataset} trial={trial_name(trial)}")
        if not self.dry_run:
            write_usage_reports(root, initial_pool_root=root)
        return root

    def optimize(self, dataset: str, trial: int) -> Path:
        self._validate_credentials(target=True, teacher=True, embedding=True)
        cfg = self.config
        task = cfg.dataset(dataset)
        task_timeout, exec_timeout = self._task_timeouts(dataset)
        baseline_root = cfg.baseline_root(dataset)
        init_root = cfg.init_root(dataset, trial)
        if not self.dry_run:
            if not _evaluation_artifacts_complete(
                baseline_root,
                task.results_file,
                expected=task.train_size,
            ):
                raise RuntimeError(
                    f"Baseline is missing, incomplete, or contains runtime failures: "
                    f"{baseline_root}. Run `cobras baseline` again."
                )
            init_count = len(list(init_root.glob("seed_*/skill.md")))
            if init_count < cfg.algorithm.pool_size:
                raise FileNotFoundError(
                    f"Need {cfg.algorithm.pool_size} init skills under {init_root}, found {init_count}"
                )
        root = cfg.run_root(dataset, trial)
        identity = {"stage": "optimize", **cfg.identity(dataset, trial)}
        self._ensure_manifest(root / "request.json", identity)
        if not self.dry_run:
            write_json(root / "resolved_config.json", cfg.public_dict())
        if self._resume_or_raise((root / "optimized_skill" / "skill.md").exists(), root, f"optimization {dataset} {trial_name(trial)}"):
            return root
        algorithm = cfg.algorithm
        seed = cfg.seed + trial - 1
        command = [
            sys.executable,
            "-m",
            "cobras.scripts.run_cobras",
            "--dataset",
            dataset,
            "--rounds",
            str(algorithm.rounds),
            "--pool-size",
            str(algorithm.pool_size),
            "--prune-count",
            str(algorithm.prune_count),
            "--prune-schedule",
            algorithm.prune_schedule,
            "--prune-log-threshold",
            str(algorithm.prune_log_threshold),
            "--prune-interval",
            str(algorithm.prune_interval),
            "--regen-sample-size",
            str(algorithm.regen_sample_size),
            "--rollout-sample-size",
            str(algorithm.rollout_sample_size),
            "--eval-limit",
            str(min(algorithm.eval_limit, task.train_size)),
            "--train-limit",
            str(task.train_size),
            "--eval-workers",
            str(task.workers),
            "--generate-parallel",
            str(algorithm.generate_parallel),
            "--reward-field",
            task.reward_field,
            "--nu",
            str(algorithm.nu),
            "--lambda_",
            str(algorithm.lambda_),
            "--seed",
            str(seed),
            "--model",
            cfg.target.model,
            "--provider",
            cfg.target.provider,
            "--base-url",
            self.target_url,
            "--api-key-env",
            "TARGET_API_KEY",
            "--teacher-model",
            cfg.teacher.model,
            "--teacher-provider",
            cfg.teacher.provider,
            "--teacher-base-url",
            self.teacher_url,
            "--teacher-api-key-env",
            "TEACHER_API_KEY",
            "--teacher-reasoning-effort",
            cfg.teacher.reasoning_effort,
            "--embedding-backend",
            cfg.embedding.backend,
            "--embedding-model",
            cfg.embedding.model,
            "--embedding-base-url",
            self.embedding_url,
            "--embedding-api-key-env",
            "EMBEDDING_API_KEY",
            "--embedding-dim",
            str(cfg.embedding.dimension),
            "--initial-pool-root",
            str(init_root),
            "--regenerate-baseline-root",
            str(baseline_root),
            "--out-root",
            str(root),
            "--max-completion-tokens",
            str(task.max_completion_tokens),
            "--task-timeout",
            str(task_timeout),
            "--exec-timeout",
            str(exec_timeout),
            "--reasoning-effort",
            task.reasoning_effort,
            "--mode",
            task.mode,
            "--prompt-version",
            task.prompt_version,
        ]
        if task.max_turns > 0:
            command.extend(["--max-turns", str(task.max_turns)])
        if task.prune_cooldown_rounds > 0:
            command.extend(["--prune-cooldown-rounds", str(task.prune_cooldown_rounds)])
        if task.prune_max_interval > 0:
            command.extend(["--prune-max-interval", str(task.prune_max_interval)])
        should_eval = cfg.initial_eval == "all" or (cfg.initial_eval == "first_trial" and trial == 1)
        if not should_eval:
            command.append("--skip-initial-eval")
        if cfg.resume:
            command.append("--resume")
        self._run(command, env=self._stage_env(dataset, root, round_name="setup"), label=f"optimize dataset={dataset} trial={trial_name(trial)}")
        return root

    def _test_one(self, dataset: str, trial: int, *, label: str, skill_path: Path | None) -> Path:
        cfg = self.config
        task = cfg.dataset(dataset)
        run_root = cfg.run_root(dataset, trial)
        root = run_root / "test_eval" / label
        identity = {
            "stage": "test",
            "kind": label,
            "dataset": dataset,
            "trial": trial,
            "target": cfg.identity(dataset)["target"],
            "harness": cfg.harness.backend,
            "task": task.__dict__,
            "skill_path": str(skill_path) if skill_path else "",
        }
        self._ensure_manifest(root / "request.json", identity)
        complete = _evaluation_artifacts_complete(
            root,
            task.results_file,
            expected=task.test_size,
        )
        if self._resume_or_raise(
            complete,
            root,
            f"test {dataset} {trial_name(trial)} {label}",
        ):
            return root
        command = self._eval_args(
            dataset,
            split="test",
            limit=task.test_size,
            out_root=root,
            seed=cfg.seed + trial - 1,
        )
        if skill_path is not None:
            if not self.dry_run and not skill_path.exists():
                raise FileNotFoundError(f"Optimized skill is missing: {skill_path}")
            command.extend(["--skill-path", str(skill_path)])
        self._run(command, env=self._stage_env(dataset, root, round_name=f"test_{label}"), label=f"test dataset={dataset} trial={trial_name(trial)} kind={label}")
        if not self.dry_run:
            write_usage_reports(root)
        return root

    def test(self, dataset: str, trial: int, target: str = "both") -> Path:
        self._validate_credentials(target=True, teacher=False, embedding=False)
        if target not in {"baseline", "optimized", "both"}:
            raise ValueError("test target must be baseline, optimized, or both")
        run_root = self.config.run_root(dataset, trial)
        roots: dict[str, Path] = {}
        if target in {"baseline", "both"}:
            roots["baseline"] = self._test_one(dataset, trial, label="baseline", skill_path=None)
        if target in {"optimized", "both"}:
            roots["optimized"] = self._test_one(
                dataset,
                trial,
                label="optimized",
                skill_path=run_root / "optimized_skill" / "skill.md",
            )
        if not self.dry_run:
            payload: dict[str, Any] = {"dataset": dataset, "trial": trial}
            for label, root in roots.items():
                hard, soft = _score(root / "summary.json")
                payload[label] = {"hard": hard, "soft": soft, "summary": str(root / "summary.json")}
                print(
                    f"[cobra-skills] result dataset={dataset} trial={trial_name(trial)} "
                    f"kind={label} hard={hard} soft={soft}",
                    flush=True,
                )
            write_json(run_root / "test_eval" / "summary.json", payload)
        return run_root / "test_eval"

    def run_all(self, datasets: Iterable[str], trials: Iterable[int]) -> None:
        for dataset in datasets:
            self.baseline(dataset)
            for trial in trials:
                self.init(dataset, trial)
                self.optimize(dataset, trial)
                self.test(dataset, trial)


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    # Validation lives in _parse_datasets. argparse applies ``choices`` to the
    # empty list produced by nargs="*", which breaks the documented default of
    # reading all selected datasets from config when no positional is supplied.
    parser.add_argument("dataset", nargs="*", metavar="DATASET")
    parser.add_argument("--datasets", action="append", help="Comma-separated datasets; defaults to config.")
    parser.add_argument("--trials", nargs="+", help="Trial numbers, separated by spaces or commas.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--run-name")
    parser.add_argument("--output-root")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--initial-eval", choices=["all", "first_trial", "none"])
    parser.add_argument("--target-model")
    parser.add_argument("--target-provider")
    parser.add_argument("--target-reasoning-effort")
    parser.add_argument("--target-enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-provider")
    parser.add_argument("--teacher-reasoning-effort")
    parser.add_argument("--target-max-tokens", type=int)
    parser.add_argument("--teacher-max-tokens", type=int)
    parser.add_argument("--embedding-backend", choices=["openrouter", "hash"])
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dimension", type=int)
    parser.add_argument("--harness", choices=HARNESS_BACKENDS)
    parser.add_argument("--harness-timeout", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--prune-count", type=int)
    parser.add_argument("--nu", type=float)
    parser.add_argument("--lambda", dest="lambda_", type=float)
    parser.add_argument("--init-count", type=int)
    parser.add_argument("--init-sample-size", type=int)
    parser.add_argument("--init-parallel", type=int)
    parser.add_argument("--set", dest="set_values", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cobras", description="Unified COBRA-Skills experiment runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()
    for command, help_text in (
        ("baseline", "Run target-model training baselines."),
        ("init", "Generate initial skill pools from baseline trajectories."),
        ("optimize", "Run NN + LinearUCB skill optimization."),
        ("run", "Run baseline, init, optimize, and test end to end."),
    ):
        subparsers.add_parser(command, parents=[common], help=help_text)
    test_parser = subparsers.add_parser("test", parents=[common], help="Evaluate baseline and optimized skills on test data.")
    test_parser.add_argument("--test-target", choices=["baseline", "optimized", "both"], default="both")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        overrides = _config_overrides(args)
        config = resolve_config(args.config, args.env_file, overrides=overrides)
        datasets = list(config.datasets)
        trials = list(config.trials)
        pipeline = Pipeline(config, dry_run=args.dry_run)
        print(
            f"[cobra-skills] command={args.command} datasets={','.join(datasets)} "
            f"trials={','.join(str(value) for value in trials)} target={config.target.model} "
            f"teacher={config.teacher.model} harness={config.harness.backend} "
            f"output={config.output_root}",
            flush=True,
        )
        if args.command == "baseline":
            for dataset in datasets:
                pipeline.baseline(dataset)
        elif args.command == "init":
            for dataset in datasets:
                for trial in trials:
                    pipeline.init(dataset, trial)
        elif args.command == "optimize":
            for dataset in datasets:
                for trial in trials:
                    pipeline.optimize(dataset, trial)
        elif args.command == "test":
            for dataset in datasets:
                for trial in trials:
                    pipeline.test(dataset, trial, target=args.test_target)
        else:
            pipeline.run_all(datasets, trials)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"cobra-skills: {exc}") from exc


if __name__ == "__main__":
    main()
