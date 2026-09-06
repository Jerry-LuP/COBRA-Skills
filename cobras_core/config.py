from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cobras.cobras_core.paths import COBRAS_ROOT


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    reward_field: str
    train_limit: int
    test_limit: int
    data_root: Path
    init_skills_root: Path
    evaluator_module: str
    skill_generator_module: str
    skill_mutator_module: str
    crossover_module: str | None
    cobras_driver_module: str
    cobras_adapter_module: str | None = None


DATASETS: dict[str, DatasetConfig] = {
    "alfworld": DatasetConfig(
        name="alfworld",
        reward_field="hard",
        train_limit=50,
        test_limit=100,
        data_root=COBRAS_ROOT / "data" / "alfworld",
        init_skills_root=COBRAS_ROOT / "datasets" / "alfworld" / "init_skills",
        evaluator_module="cobras.task_envs.alfworld.eval",
        skill_generator_module="cobras.task_envs.alfworld.scripts.generate_alfworld_skill_from_results",
        skill_mutator_module="cobras.task_envs.alfworld.scripts.mutate_alfworld_skill_from_results",
        crossover_module="cobras.task_envs.alfworld.scripts.crossover_skill",
        cobras_driver_module="cobras.task_envs.alfworld.cobras_driver",
        cobras_adapter_module="cobras.task_envs.alfworld.cobras_adapter",
    ),
    "docvqa": DatasetConfig(
        name="docvqa",
        reward_field="hard",
        train_limit=50,
        test_limit=100,
        data_root=COBRAS_ROOT / "data" / "docvqa",
        init_skills_root=COBRAS_ROOT / "datasets" / "docvqa" / "init_skills",
        evaluator_module="cobras.task_envs.docvqa.eval",
        skill_generator_module="cobras.task_envs.docvqa.scripts.generate_docvqa_skill_from_results",
        skill_mutator_module="cobras.task_envs.docvqa.scripts.mutate_docvqa_skill_from_results",
        crossover_module="cobras.task_envs.docvqa.scripts.crossover_skill",
        cobras_driver_module="cobras.task_envs.docvqa.cobras_driver",
        cobras_adapter_module="cobras.task_envs.docvqa.cobras_adapter",
    ),
    "livemath": DatasetConfig(
        name="livemath",
        reward_field="hard",
        train_limit=50,
        test_limit=100,
        data_root=COBRAS_ROOT / "data" / "livemath",
        init_skills_root=COBRAS_ROOT / "datasets" / "livemath" / "init_skills",
        evaluator_module="cobras.task_envs.livemath.eval",
        skill_generator_module="cobras.task_envs.livemath.scripts.generate_livemath_skill_from_results",
        skill_mutator_module="cobras.task_envs.livemath.scripts.mutate_livemath_skill_from_results",
        crossover_module="cobras.task_envs.livemath.scripts.crossover_skill",
        cobras_driver_module="cobras.task_envs.livemath.cobras_driver",
        cobras_adapter_module="cobras.task_envs.livemath.cobras_adapter",
    ),
    "socialmaze_hard": DatasetConfig(
        name="socialmaze_hard",
        reward_field="soft",
        train_limit=50,
        test_limit=100,
        data_root=COBRAS_ROOT / "data" / "socialmaze_hard",
        init_skills_root=COBRAS_ROOT / "datasets" / "socialmaze_hard" / "init_skills",
        evaluator_module="cobras.task_envs.socialmaze_hard.eval",
        skill_generator_module="cobras.task_envs.socialmaze_hard.scripts.generate_socialmaze_hidden_role_skill_from_results",
        skill_mutator_module="cobras.task_envs.socialmaze_hard.scripts.mutate_socialmaze_hidden_role_skill_from_results",
        crossover_module="cobras.task_envs.socialmaze_hard.scripts.crossover_skill",
        cobras_driver_module="cobras.task_envs.socialmaze_hard.cobras_driver",
        cobras_adapter_module="cobras.task_envs.socialmaze_hard.cobras_adapter",
    ),
    "spreadsheetbench": DatasetConfig(
        name="spreadsheetbench",
        reward_field="hard",
        train_limit=50,
        test_limit=100,
        data_root=COBRAS_ROOT / "data" / "spreadsheetbench",
        init_skills_root=COBRAS_ROOT / "datasets" / "spreadsheetbench" / "init_skills",
        evaluator_module="cobras.task_envs.spreadsheetbench.eval",
        skill_generator_module="cobras.task_envs.spreadsheetbench.scripts.generate_spreadsheetbench_skill_from_results",
        skill_mutator_module="cobras.task_envs.spreadsheetbench.scripts.mutate_spreadsheetbench_skill_from_results",
        crossover_module="cobras.task_envs.spreadsheetbench.scripts.crossover_skill",
        cobras_driver_module="cobras.task_envs.spreadsheetbench.cobras_driver",
        cobras_adapter_module="cobras.task_envs.spreadsheetbench.cobras_adapter",
    ),
    "searchqa": DatasetConfig(
        name="searchqa",
        reward_field="hard",
        train_limit=50,
        test_limit=100,
        data_root=COBRAS_ROOT / "data" / "searchqa",
        init_skills_root=COBRAS_ROOT / "datasets" / "searchqa" / "init_skills",
        evaluator_module="cobras.task_envs.searchqa.eval",
        skill_generator_module="cobras.task_envs.searchqa.scripts.generate_searchqa_skill_from_results",
        skill_mutator_module="cobras.task_envs.searchqa.scripts.mutate_searchqa_skill_from_results",
        crossover_module="cobras.task_envs.searchqa.scripts.crossover_skill",
        cobras_driver_module="cobras.task_envs.searchqa.cobras_driver",
        cobras_adapter_module="cobras.task_envs.searchqa.cobras_adapter",
    ),
}


def get_dataset(name: str) -> DatasetConfig:
    key = name.strip().lower()
    if key not in DATASETS:
        raise SystemExit(f"Unknown dataset={name!r}. Available: {', '.join(sorted(DATASETS))}")
    return DATASETS[key]
