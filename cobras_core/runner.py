from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from cobras.cobras_core.bandit_runtime import BanditRuntimeMixin
from cobras.cobras_core.operators import CobrasOperatorsMixin
from cobras.cobras_core.run_loop import RunnerLoopMixin
from cobras.cobras_core.run_state import (
    PROJECT_ROOT,
    ArmState,
    arm_serial as _arm_serial,
    load_initial_arms as _load_initial_arms,
)
from cobras.cobras_core.update_schedule import PopulationUpdateSchedule
from cobras.scripts.embeddings import DEFAULT_OPENROUTER_BASE_URL, DEFAULT_OPENROUTER_MODEL
from cobras.task_envs.cobras_adapter import PromptTaskCobrasAdapter, load_cobras_adapter



class COBRASRunner(BanditRuntimeMixin, CobrasOperatorsMixin, RunnerLoopMixin):
    def __init__(
        self,
        *,
        initial_pool_root: Path,
        initial_eval_summary: Path,
        regenerate_baseline_root: Path,
        out_root: Path,
        rounds: int,
        pool_size: int = 10,
        prune_count: int = 3,
        crossover_threshold: int = 8,
        train_rollout_tasks: int = 8,
        train_rollout_parallel: int = 24,
        eval_parallel: int = 24,
        max_turns: int = 0,
        selected_eval_limit: int = 0,
        selected_eval_split: str = "val",
        selected_eval_split_json: Path | None = None,
        generate_parallel: int = 3,
        prune_schedule: str = "log",
        prune_interval: int = 3,
        prune_log_threshold: float = 0.35,
        prune_cooldown_rounds: int | None = None,
        prune_max_interval: int | None = None,
        crossover_num_children: int = 1,
        crossover_parallel: int = 1,
        crossover_k: int = 2,
        crossover_top_pool_size: int = 4,
        crossover_bottom_pool_size: int = 4,
        crossover_score_field: str = "avg_soft_reward",
        regen_sample_size: int = 6,
        regen_min_success: int = 1,
        regen_min_fail: int = 1,
        rollout_sample_size: int = 6,
        rollout_sample_min_success: int = 1,
        rollout_sample_min_fail: int = 1,
        nu: float = 0.1,
        lambda_: float = 0.03,
        mlp_l2: float = 1e-4,
        embedding_backend: str = "openrouter",
        embedding_model: str = DEFAULT_OPENROUTER_MODEL,
        embedding_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        embedding_api_key_env: str = "OPENROUTER_API_KEY",
        embedding_dim: int = 0,
        model: str = "gpt-5.5",
        provider: str = "yunwu",
        teacher_model: str = "",
        teacher_provider: str = "",
        teacher_reasoning_effort: str = "medium",
        base_url: str = "",
        api_key_env: str = "OPENAI_API_KEY",
        api_key: str = "",
        teacher_base_url: str = "",
        teacher_api_key_env: str = "",
        teacher_api_key: str = "",
        openai_base_url_env: str = "OPENAI_BASE_URL",
        task_package: str = "scripts",
        dataset: str = "",
        reward_field: str = "soft",
        prompt_version: str = "direct",
        test_mode: bool = False,
        resume: bool = False,
        seed: int = 42,
    ) -> None:
        self.initial_pool_root = initial_pool_root.resolve()
        self.initial_eval_summary = initial_eval_summary.resolve()
        self.regenerate_baseline_root = regenerate_baseline_root.resolve()
        self.out_root = out_root.resolve()
        self.rounds = rounds
        self.pool_size = pool_size
        self.prune_count = prune_count
        self.crossover_threshold = crossover_threshold
        self.train_rollout_tasks = train_rollout_tasks
        self.train_rollout_parallel = train_rollout_parallel
        self.eval_parallel = eval_parallel
        if max_turns < 0:
            raise ValueError(f"max_turns must be non-negative, got {max_turns}")
        self.max_turns = max_turns
        self.selected_eval_limit = selected_eval_limit
        self.selected_eval_split = selected_eval_split
        self.selected_eval_split_json = selected_eval_split_json.resolve() if selected_eval_split_json is not None else None
        self.generate_parallel = generate_parallel
        self.prune_interval = max(1, prune_interval)
        self.update_schedule = PopulationUpdateSchedule(
            mode=prune_schedule,
            total_rounds=rounds,
            min_interval=self.prune_interval,
            log_threshold=prune_log_threshold,
            cooldown_rounds=prune_cooldown_rounds,
            max_interval=prune_max_interval,
        )
        self.crossover_num_children = crossover_num_children
        self.crossover_parallel = crossover_parallel
        self.crossover_k = crossover_k
        self.crossover_top_pool_size = crossover_top_pool_size
        self.crossover_bottom_pool_size = crossover_bottom_pool_size
        self.crossover_score_field = crossover_score_field
        self.regen_sample_size = regen_sample_size
        self.regen_min_success = regen_min_success
        self.regen_min_fail = regen_min_fail
        self.rollout_sample_size = rollout_sample_size
        self.rollout_sample_min_success = rollout_sample_min_success
        self.rollout_sample_min_fail = rollout_sample_min_fail
        self.nu = nu
        self.lambda_ = lambda_
        self.mlp_l2 = mlp_l2
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url
        self.embedding_api_key_env = embedding_api_key_env
        self.embedding_dim = embedding_dim
        self.model = model
        self.provider = provider
        self.teacher_model = teacher_model or model
        self.teacher_provider = teacher_provider or provider
        self.teacher_reasoning_effort = teacher_reasoning_effort
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.api_key = api_key
        self.teacher_base_url = teacher_base_url or base_url
        self.teacher_api_key_env = teacher_api_key_env or api_key_env
        self.teacher_api_key = teacher_api_key or api_key
        self.openai_base_url_env = openai_base_url_env
        self.task_package = task_package.rstrip(".") or "scripts"
        self.dataset = dataset.strip().lower()
        self.dataset_adapter: PromptTaskCobrasAdapter | None = (
            load_cobras_adapter(self.dataset) if self.dataset else None
        )
        if self.dataset and self.dataset_adapter is None:
            raise ValueError(f"Dataset {self.dataset!r} does not provide a COBRAS adapter")
        if reward_field not in {"hard", "soft"}:
            raise ValueError(f"reward_field must be hard or soft, got {reward_field!r}")
        self.reward_field = reward_field
        self.prompt_version = prompt_version
        self.test_mode = test_mode
        self.resume = resume
        self.seed = seed
        self.rng = random.Random(seed)
        self.rounds_dir = self.out_root / "rounds"
        self.embedding_cache_root = self.out_root / "embedding_cache"
        self.initial_arms = _load_initial_arms(self.initial_eval_summary)
        if len(self.initial_arms) < self.pool_size:
            raise ValueError(f"Need at least {self.pool_size} initial arms, found {len(self.initial_arms)}")
        self._next_arm_serial = max(_arm_serial(row["arm_id"]) for row in self.initial_arms) + 1
        self.arms = []
        for row in self.initial_arms[: self.pool_size]:
            self.arms.append(
                ArmState(
                    arm_id=str(row["arm_id"]),
                    skill_name=str(row["skill_name"]),
                    skill_root=str(row["skill_root"]),
                    workspace_dir=str(row["workspace_dir"]),
                    latest_reward=None,
                    latest_soft_reward=None,
                    latest_hard_reward=None,
                    latest_eval_summary=None,
                    latest_eval_root=None,
                    latest_train_rollout_root=None,
                    origin=str(row.get("origin", "init")),
                    parent_ids=list(row.get("parent_ids") or []),
                    generation=int(row.get("generation", 0)),
                )
            )
        self.history: list[dict[str, Any]] = []
        self.dynamic_unique_skill_roots: set[str] = set()
        self.embeddings: dict[str, np.ndarray] = {}
        self.prev_scored: dict[str, dict[str, Any]] = {}
        self._selected_eval_split_json_cache: Path | None = None

    def _module(self, name: str) -> str:
        return f"{self.task_package}.{name}"

    def _selected_eval_split_json_path(self) -> Path | None:
        if self.selected_eval_split_json is not None:
            return self.selected_eval_split_json
        if self.selected_eval_split != "trainval":
            return None
        if self._selected_eval_split_json_cache is not None:
            return self._selected_eval_split_json_cache
        train_path = PROJECT_ROOT / "data" / "legacy_id_split" / "train" / "items.json"
        val_path = PROJECT_ROOT / "data" / "legacy_id_split" / "val" / "items.json"
        train_items = json.loads(train_path.read_text(encoding="utf-8"))
        val_items = json.loads(val_path.read_text(encoding="utf-8"))
        merged_path = self.out_root / "inputs" / "selected_eval_trainval_items.json"
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_path.write_text(json.dumps(list(train_items) + list(val_items), ensure_ascii=False, indent=2), encoding="utf-8")
        self._selected_eval_split_json_cache = merged_path
        return merged_path

    def _selected_eval_limit_value(self) -> int:
        if self.selected_eval_split == "trainval" and self.selected_eval_limit <= 0:
            return 50
        return self.selected_eval_limit

    def _load_history_from_disk(self) -> list[dict[str, Any]]:
        history_path = self.out_root / "history.jsonl"
        if not history_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(history_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {history_path}:{line_no}") from exc
            selected_eval = payload.get("selected_eval", {})
            selected_arm = payload.get("selected_arm", {})
            soft_reward = selected_eval.get("avg_soft_reward")
            hard_reward = selected_eval.get("avg_hard_reward")
            reward = hard_reward if self.reward_field == "hard" else soft_reward
            arm_id = selected_arm.get("arm_id")
            skill_root = selected_arm.get("skill_root")
            if arm_id is None or skill_root is None or reward is None:
                continue
            rows.append(
                {
                    "round": int(payload.get("round", len(rows))),
                    "arm_id": str(arm_id),
                    "skill_root": str(skill_root),
                    "reward": float(reward),
                    "soft_reward": float(soft_reward or 0.0),
                    "hard_reward": float(hard_reward or 0.0),
                    "summary_path": str(selected_eval.get("summary_path", "")),
                    "source": "selected_eval",
                    "resume": True,
                }
            )
        return rows

    def _resume_state_if_available(self) -> int:
        if not self.resume:
            return 0
        completed_rounds: list[tuple[int, Path]] = []
        if self.rounds_dir.exists():
            for path in self.rounds_dir.glob("round_*/pool_state.json"):
                try:
                    round_idx = int(path.parent.name.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                completed_rounds.append((round_idx, path))
        if not completed_rounds:
            print("[cobras] resume requested but no completed rounds found; starting from round=000", flush=True)
            return 0
        last_round, pool_state_path = max(completed_rounds, key=lambda item: item[0])
        payload = json.loads(pool_state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"Invalid pool state in {pool_state_path}")
        self.arms = []
        for row in payload:
            self.arms.append(
                ArmState(
                    arm_id=str(row["arm_id"]),
                    skill_name=str(row.get("skill_name") or Path(str(row["workspace_dir"])).name),
                    skill_root=str(Path(str(row["skill_root"])).resolve()),
                    workspace_dir=str(Path(str(row["workspace_dir"])).resolve()),
                    latest_reward=None if row.get("latest_reward") is None else float(row["latest_reward"]),
                    latest_soft_reward=None
                    if row.get("latest_soft_reward") is None
                    else float(row["latest_soft_reward"]),
                    latest_hard_reward=None
                    if row.get("latest_hard_reward") is None
                    else float(row["latest_hard_reward"]),
                    latest_eval_summary=row.get("latest_eval_summary"),
                    latest_eval_root=row.get("latest_eval_root"),
                    latest_train_rollout_root=row.get("latest_train_rollout_root"),
                    origin=str(row.get("origin", "resume")),
                    parent_ids=list(row.get("parent_ids") or []),
                    generation=int(row.get("generation", 0)),
                )
            )
        self.history = self._load_history_from_disk()
        self.dynamic_unique_skill_roots = {str(row["skill_root"]) for row in self.history}
        self._next_arm_serial = max(_arm_serial(arm.arm_id) for arm in self.arms) + 1
        print(
            f"[cobras] resume from round={last_round + 1:03d} "
            f"pool_state={pool_state_path} history={len(self.history)} next_arm={self._next_arm_serial}",
            flush=True,
        )
        return last_round + 1

    def _allocate_arm_id(self) -> str:
        arm_id = f"arm_{self._next_arm_serial:03d}"
        self._next_arm_serial += 1
        return arm_id
