from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Mapping

from cobras.cobras_core.config import get_dataset
from cobras.task_envs.eval_runtime import safe_name


@dataclass(frozen=True)
class PromptTaskCobrasAdapter:
    """Task-owned details needed by the shared COBRAS optimization loop."""

    dataset: str
    generation_results_filename: str = "results.jsonl"
    eval_results_filename: str = "results.jsonl"
    target_reasoning_effort: str = ""
    eval_mode: str = "multi"
    eval_max_turns: int = 0
    eval_max_completion_tokens: int = 0
    eval_task_timeout: int = 0
    include_prompt_version: bool = False
    use_eval_feedback: bool = False
    rollout_sampling: str = "task"
    include_unevaluated_pool_arms: bool = True
    pool_summary_sort_field: str = "ucb_score"
    regenerate_env: Mapping[str, str] = field(default_factory=dict)
    mutation_env: Mapping[str, str] = field(default_factory=dict)

    def generation_results_path(self, run_root: Path) -> Path:
        return run_root if run_root.is_file() else run_root / self.generation_results_filename

    def eval_results_path(self, eval_root: Path) -> Path:
        return eval_root / self.eval_results_filename

    def normalize_eval_summary(
        self,
        *,
        summary_path: Path,
        skill_name: str,
        skill_root: Path,
        workspace_dir: Path,
    ) -> dict:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        count = int(payload.get("count", payload.get("n", 0)) or 0)
        hard = payload.get("hard_acc")
        if hard is None:
            hard_count = float(payload.get("hard", payload.get("hard_correct", 0.0)) or 0.0)
            hard = hard_count / max(count, 1)
        soft = float(payload.get("avg_soft", payload.get("avg_soft_reward", hard)) or 0.0)
        return {
            "skill_name": skill_name,
            "workspace_dir": str(workspace_dir.resolve()),
            "skill_root": str(skill_root.resolve()),
            "avg_soft_reward": soft,
            "avg_hard_reward": float(hard or 0.0),
            "success_tasks": payload.get("hard", payload.get("hard_correct")),
            "count": count,
            "summary_path": str(summary_path.resolve()),
            "results_path": str(self.eval_results_path(summary_path.parent).resolve()),
            "predictions_dir": str((summary_path.parent / "predictions").resolve()),
        }

    def prepare_mutation_inputs(
        self,
        *,
        skill_path: Path,
        eval_root: Path,
        out_root: Path,
    ) -> dict[str, str]:
        """Stage the one-skill sweep layout consumed by the existing mutators."""

        skill_name = safe_name(out_root.name)
        staging_root = out_root.parent / f".{skill_name}_mutation_inputs"
        if staging_root.exists():
            shutil.rmtree(staging_root)
        skills_root = staging_root / "skills"
        staged_skill = skills_root / skill_name / "skill.md"
        staged_skill.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(skill_path, staged_skill)

        staged_eval_root = staging_root / "eval"
        staged_run_root = staged_eval_root / skill_name
        staged_run_root.mkdir(parents=True, exist_ok=True)
        results_path = self.eval_results_path(eval_root)
        if not results_path.exists():
            raise FileNotFoundError(f"Missing mutation results: {results_path}")
        shutil.copy2(results_path, staged_run_root / "results.jsonl")
        predictions_dir = eval_root / "predictions"
        if predictions_dir.exists():
            shutil.copytree(predictions_dir, staged_run_root / "predictions")

        env = {
            "SKILLS_ROOT": str(skills_root),
            "EVAL_ROOT": str(staged_eval_root),
            "OUT_ROOT": str(out_root.parent),
            **{str(key): str(value) for key, value in self.mutation_env.items()},
        }
        return env


def load_cobras_adapter(dataset: str) -> PromptTaskCobrasAdapter | None:
    cfg = get_dataset(dataset)
    if not cfg.cobras_adapter_module:
        return None
    module = import_module(cfg.cobras_adapter_module)
    adapter = getattr(module, "ADAPTER", None)
    if not isinstance(adapter, PromptTaskCobrasAdapter):
        raise RuntimeError(
            f"{cfg.cobras_adapter_module} must export ADAPTER as PromptTaskCobrasAdapter"
        )
    if adapter.dataset != cfg.name:
        raise RuntimeError(
            f"Adapter dataset mismatch: config={cfg.name!r}, adapter={adapter.dataset!r}"
        )
    return adapter
