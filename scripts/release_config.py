from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from cobras.cobras_core.paths import COBRAS_ROOT


DATASETS = (
    "docvqa",
    "livemath",
    "searchqa",
    "socialmaze_hard",
    "spreadsheetbench",
    "alfworld",
)
HARNESS_BACKENDS = ("native", "codex", "claude_code")
PROVIDERS = ("openrouter", "yunwu", "custom", "local")


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): deepcopy(item) for key, item in value.items()}


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if resolved < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {resolved}")
    return resolved


def _number(value: Any, *, name: str, minimum: float = 0.0) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if resolved < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {resolved}")
    return resolved


def _boolean(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _string_list(value: Any, *, name: str) -> list[str]:
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        values = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise ValueError(f"{name} must be a list or comma-separated string")
    return values


def _trial_list(value: Any) -> list[int]:
    trials = [_integer(item, name="experiment.trials", minimum=1) for item in _string_list(value, name="experiment.trials")]
    if len(set(trials)) != len(trials):
        raise ValueError("experiment.trials contains duplicate values")
    return trials


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def set_dotted(payload: dict[str, Any], dotted: str, value: Any) -> None:
    cursor = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = cursor.get(part)
        if not isinstance(current, dict):
            current = {}
            cursor[part] = current
        cursor = current
    cursor[parts[-1]] = value


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(f"Invalid env assignment at {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid env key at {path}:{line_number}: {key!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_runtime_env(path: Path) -> dict[str, str]:
    merged = parse_env_file(path)
    merged.update({key: value for key, value in os.environ.items()})
    return merged


def slug(value: str) -> str:
    normalized = str(value or "model").strip().split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", normalized).strip("-") or "model"


def trial_name(trial: int) -> str:
    return f"trial_{trial:03d}"


@dataclass(frozen=True)
class RoleConfig:
    model: str
    provider: str
    reasoning_effort: str
    temperature: float
    enable_thinking: bool
    max_completion_tokens: int


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str
    model: str
    dimension: int


@dataclass(frozen=True)
class HarnessConfig:
    backend: str
    timeout_seconds: int
    context_window: int
    empty_response_retries: int
    codex: dict[str, Any]
    claude_code: dict[str, Any]


@dataclass(frozen=True)
class AlgorithmConfig:
    rounds: int
    pool_size: int
    prune_count: int
    prune_schedule: str
    prune_log_threshold: float
    prune_interval: int
    regen_sample_size: int
    rollout_sample_size: int
    eval_limit: int
    generate_parallel: int
    nu: float
    lambda_: float


@dataclass(frozen=True)
class InitSkillsConfig:
    count: int
    sample_size: int
    sample_seed: int
    parallel: int
    reasoning_effort: str


@dataclass(frozen=True)
class DatasetRunConfig:
    name: str
    train_size: int
    test_size: int
    workers: int
    max_completion_tokens: int
    task_timeout: int
    exec_timeout: int
    reasoning_effort: str
    mode: str
    max_turns: int
    prompt_version: str
    image_detail: str
    results_file: str
    reward_field: str
    prune_cooldown_rounds: int
    prune_max_interval: int


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path
    env_path: Path
    name: str
    output_root: Path
    seed: int
    trials: tuple[int, ...]
    datasets: tuple[str, ...]
    resume: bool
    initial_eval: str
    target: RoleConfig
    teacher: RoleConfig
    embedding: EmbeddingConfig
    harness: HarnessConfig
    algorithm: AlgorithmConfig
    init_skills: InitSkillsConfig
    task_settings: dict[str, DatasetRunConfig]
    raw: dict[str, Any]

    def dataset(self, name: str) -> DatasetRunConfig:
        try:
            return self.task_settings[name]
        except KeyError as exc:
            raise ValueError(f"Dataset {name!r} is not configured") from exc

    @property
    def target_tag(self) -> str:
        return slug(self.target.model)

    @property
    def teacher_tag(self) -> str:
        return slug(self.teacher.model)

    @property
    def model_pair_tag(self) -> str:
        return f"target_{self.target_tag}__teacher_{self.teacher_tag}"

    def baseline_root(self, dataset: str) -> Path:
        task = self.dataset(dataset)
        return (
            self.output_root
            / "baselines"
            / dataset
            / f"target_{self.target_tag}"
            / f"harness_{self.harness.backend}"
            / f"train_{task.train_size}"
        )

    def init_root(self, dataset: str, trial: int) -> Path:
        return (
            self.output_root
            / "init_skills"
            / dataset
            / self.model_pair_tag
            / f"harness_{self.harness.backend}"
            / trial_name(trial)
        )

    def run_root(self, dataset: str, trial: int) -> Path:
        return (
            self.output_root
            / "runs"
            / dataset
            / self.model_pair_tag
            / f"harness_{self.harness.backend}"
            / slug(self.name)
            / trial_name(trial)
        )

    def public_dict(self) -> dict[str, Any]:
        return deepcopy(self.raw)

    def identity(self, dataset: str, trial: int | None = None) -> dict[str, Any]:
        task = self.dataset(dataset)
        payload: dict[str, Any] = {
            "dataset": dataset,
            "target": {
                "model": self.target.model,
                "provider": self.target.provider,
                "reasoning_effort": self.target.reasoning_effort,
                "temperature": self.target.temperature,
                "enable_thinking": self.target.enable_thinking,
                "max_completion_tokens": self.target.max_completion_tokens,
            },
            "teacher": {
                "model": self.teacher.model,
                "provider": self.teacher.provider,
                "reasoning_effort": self.teacher.reasoning_effort,
                "temperature": self.teacher.temperature,
                "enable_thinking": self.teacher.enable_thinking,
                "max_completion_tokens": self.teacher.max_completion_tokens,
            },
            "embedding": {
                "backend": self.embedding.backend,
                "model": self.embedding.model,
                "dimension": self.embedding.dimension,
            },
            "harness": self.harness.backend,
            "task": task.__dict__,
            "algorithm": self.algorithm.__dict__,
            "init_skills": self.init_skills.__dict__,
            "experiment": self.name,
        }
        if trial is not None:
            payload["trial"] = trial
            payload["seed"] = self.seed + trial - 1
        return payload


def _role(raw: Mapping[str, Any], *, name: str, default_reasoning: str) -> RoleConfig:
    provider = str(raw.get("provider", "openrouter")).strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"{name}.provider must be one of {', '.join(PROVIDERS)}")
    model = str(raw.get("model", "")).strip()
    if not model:
        raise ValueError(f"{name}.model is required")
    return RoleConfig(
        model=model,
        provider=provider,
        reasoning_effort=str(raw.get("reasoning_effort", default_reasoning)).strip(),
        temperature=_number(raw.get("temperature", 0.0), name=f"{name}.temperature"),
        enable_thinking=_boolean(raw.get("enable_thinking", False), name=f"{name}.enable_thinking"),
        max_completion_tokens=_integer(
            raw.get("max_completion_tokens", 16384),
            name=f"{name}.max_completion_tokens",
            minimum=1,
        ),
    )


def resolve_config(
    config_path: Path,
    env_path: Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    config_path = config_path.expanduser().resolve()
    env_path = env_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw = _mapping(loaded, name="configuration")
    for dotted, value in (overrides or {}).items():
        if value is not None:
            set_dotted(raw, dotted, value)

    experiment = _mapping(raw.get("experiment", {}), name="experiment")
    target_raw = _mapping(raw.get("target", {}), name="target")
    teacher_raw = _mapping(raw.get("teacher", {}), name="teacher")
    embedding_raw = _mapping(raw.get("embedding", {}), name="embedding")
    harness_raw = _mapping(raw.get("harness", {}), name="harness")
    algorithm_raw = _mapping(raw.get("algorithm", {}), name="algorithm")
    init_raw = _mapping(raw.get("init_skills", {}), name="init_skills")
    datasets_raw = _mapping(raw.get("datasets", {}), name="datasets")

    selected_datasets = _string_list(
        experiment.get("datasets", DATASETS),
        name="experiment.datasets",
    )
    unknown = sorted(set(selected_datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unsupported datasets: {', '.join(unknown)}")
    trials = _trial_list(experiment.get("trials", [1, 2, 3]))
    initial_eval = str(experiment.get("initial_eval", "first_trial")).strip().lower()
    if initial_eval not in {"all", "first_trial", "none"}:
        raise ValueError("experiment.initial_eval must be all, first_trial, or none")

    output_root = Path(str(experiment.get("output_root", "outputs"))).expanduser()
    if not output_root.is_absolute():
        output_root = COBRAS_ROOT / output_root
    output_root = output_root.resolve()

    harness_backend = str(harness_raw.get("backend", "native")).strip().lower()
    if harness_backend not in HARNESS_BACKENDS:
        raise ValueError(f"harness.backend must be one of {', '.join(HARNESS_BACKENDS)}")
    harness = HarnessConfig(
        backend=harness_backend,
        timeout_seconds=_integer(harness_raw.get("timeout_seconds", 600), name="harness.timeout_seconds", minimum=1),
        context_window=_integer(harness_raw.get("context_window", 131072), name="harness.context_window", minimum=1),
        empty_response_retries=_integer(harness_raw.get("empty_response_retries", 0), name="harness.empty_response_retries"),
        codex=_mapping(harness_raw.get("codex", {}), name="harness.codex"),
        claude_code=_mapping(harness_raw.get("claude_code", {}), name="harness.claude_code"),
    )

    embedding_backend = str(embedding_raw.get("backend", "openrouter")).strip().lower()
    if embedding_backend not in {"openrouter", "hash"}:
        raise ValueError("embedding.backend must be openrouter or hash")
    embedding = EmbeddingConfig(
        backend=embedding_backend,
        model=str(embedding_raw.get("model", "qwen/qwen3-embedding-4b")).strip(),
        dimension=_integer(embedding_raw.get("dimension", 2560), name="embedding.dimension", minimum=1),
    )

    algorithm = AlgorithmConfig(
        rounds=_integer(algorithm_raw.get("rounds", 30), name="algorithm.rounds", minimum=1),
        pool_size=_integer(algorithm_raw.get("pool_size", 10), name="algorithm.pool_size", minimum=1),
        prune_count=_integer(algorithm_raw.get("prune_count", 3), name="algorithm.prune_count"),
        prune_schedule=str(algorithm_raw.get("prune_schedule", "log")).strip().lower(),
        prune_log_threshold=_number(algorithm_raw.get("prune_log_threshold", 0.35), name="algorithm.prune_log_threshold", minimum=0.000001),
        prune_interval=_integer(algorithm_raw.get("prune_interval", 3), name="algorithm.prune_interval", minimum=1),
        regen_sample_size=_integer(algorithm_raw.get("regen_sample_size", 8), name="algorithm.regen_sample_size", minimum=1),
        rollout_sample_size=_integer(algorithm_raw.get("rollout_sample_size", 8), name="algorithm.rollout_sample_size", minimum=1),
        eval_limit=_integer(algorithm_raw.get("eval_limit", 50), name="algorithm.eval_limit", minimum=1),
        generate_parallel=_integer(algorithm_raw.get("generate_parallel", 3), name="algorithm.generate_parallel", minimum=1),
        nu=_number(algorithm_raw.get("nu", 0.1), name="algorithm.nu"),
        lambda_=_number(algorithm_raw.get("lambda", 0.03), name="algorithm.lambda", minimum=0.000001),
    )
    if algorithm.prune_schedule not in {"fixed", "log"}:
        raise ValueError("algorithm.prune_schedule must be fixed or log")
    if algorithm.pool_size <= algorithm.prune_count:
        raise ValueError("algorithm.pool_size must be greater than algorithm.prune_count")

    init_skills = InitSkillsConfig(
        count=_integer(init_raw.get("count", 10), name="init_skills.count", minimum=1),
        sample_size=_integer(init_raw.get("sample_size", 8), name="init_skills.sample_size", minimum=1),
        sample_seed=_integer(init_raw.get("sample_seed", 42), name="init_skills.sample_seed"),
        parallel=_integer(init_raw.get("parallel", 3), name="init_skills.parallel", minimum=1),
        reasoning_effort=str(init_raw.get("reasoning_effort", teacher_raw.get("reasoning_effort", "medium"))).strip(),
    )
    if init_skills.count < algorithm.pool_size:
        raise ValueError("init_skills.count must be at least algorithm.pool_size")

    task_settings: dict[str, DatasetRunConfig] = {}
    for dataset in selected_datasets:
        task_raw = _mapping(datasets_raw.get(dataset, {}), name=f"datasets.{dataset}")
        mode = str(task_raw.get("mode", "multi")).strip().lower()
        if mode not in {"single", "multi"}:
            raise ValueError(f"datasets.{dataset}.mode must be single or multi")
        reward_field = str(task_raw.get("reward_field", "hard")).strip().lower()
        if reward_field not in {"hard", "soft"}:
            raise ValueError(f"datasets.{dataset}.reward_field must be hard or soft")
        task_settings[dataset] = DatasetRunConfig(
            name=dataset,
            train_size=_integer(task_raw.get("train_size", 50), name=f"datasets.{dataset}.train_size", minimum=1),
            test_size=_integer(task_raw.get("test_size", 100), name=f"datasets.{dataset}.test_size", minimum=1),
            workers=_integer(task_raw.get("workers", 15), name=f"datasets.{dataset}.workers", minimum=1),
            max_completion_tokens=_integer(
                task_raw.get("max_completion_tokens", target_raw.get("max_completion_tokens", 8192)),
                name=f"datasets.{dataset}.max_completion_tokens",
                minimum=1,
            ),
            task_timeout=_integer(task_raw.get("task_timeout", 600), name=f"datasets.{dataset}.task_timeout", minimum=1),
            exec_timeout=_integer(task_raw.get("exec_timeout", harness.timeout_seconds), name=f"datasets.{dataset}.exec_timeout", minimum=1),
            reasoning_effort=str(task_raw.get("reasoning_effort", target_raw.get("reasoning_effort", "none"))).strip(),
            mode=mode,
            max_turns=_integer(task_raw.get("max_turns", 1), name=f"datasets.{dataset}.max_turns"),
            prompt_version=str(task_raw.get("prompt_version", "direct")).strip().lower(),
            image_detail=str(task_raw.get("image_detail", "auto")).strip().lower(),
            results_file=str(task_raw.get("results_file", "results.jsonl")).strip(),
            reward_field=reward_field,
            prune_cooldown_rounds=_integer(task_raw.get("prune_cooldown_rounds", 0), name=f"datasets.{dataset}.prune_cooldown_rounds"),
            prune_max_interval=_integer(task_raw.get("prune_max_interval", 0), name=f"datasets.{dataset}.prune_max_interval"),
        )

    return ExperimentConfig(
        source_path=config_path,
        env_path=env_path,
        name=str(experiment.get("name", "default")).strip() or "default",
        output_root=output_root,
        seed=_integer(experiment.get("seed", 42), name="experiment.seed"),
        trials=tuple(trials),
        datasets=tuple(selected_datasets),
        resume=_boolean(experiment.get("resume", True), name="experiment.resume"),
        initial_eval=initial_eval,
        target=_role(target_raw, name="target", default_reasoning="none"),
        teacher=_role(teacher_raw, name="teacher", default_reasoning="medium"),
        embedding=embedding,
        harness=harness,
        algorithm=algorithm,
        init_skills=init_skills,
        task_settings=task_settings,
        raw=raw,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
