from __future__ import annotations

from pathlib import Path
from typing import Any

from cobras.cobras_core.run_state import (
    copy_skill_tree as _copy_skill_tree,
    deterministic_rng as _deterministic_rng,
    load_generated_child as _load_generated_child,
    load_json as _load_json,
    run_module as _run_module,
    skill_entrypoint_path as _skill_entrypoint_path,
    write_json,
)


class SkillOpsMixin:
    """Regenerate, rollout-mutate, and crossover skill operators."""

    def _generate_regenerate(self, round_root: Path, child_index: int) -> dict[str, Any]:
        child_root = round_root / "generated" / "regenerate" / f"child_{child_index:03d}"
        if self.test_mode:
            rng = _deterministic_rng(self.seed, round_root.name, child_index, "regenerate")
            source = rng.choice(self.initial_arms)
            workspace_dir = child_root / "workspace_001"
            skill_dir = workspace_dir / "skill"
            _copy_skill_tree(Path(str(source["skill_root"])), skill_dir)
            summary_path = child_root / f"sample{self.regen_sample_size}_skills1" / "summary.json"
            row = {
                "skill_name": str(source["skill_name"]),
                "workspace_dir": str(workspace_dir),
                "skill_root": str(skill_dir),
                "avg_soft_reward": rng.random(),
                "avg_hard_reward": rng.random(),
                "count": 1,
                "summary_path": str(summary_path),
                "origin": "regenerate",
                "test_mode": True,
            }
            write_json(
                summary_path,
                {
                    "generated": [row],
                    "test_mode": True,
                },
            )
            print(
                f"[cobras-generate] child={child_index} mode=regenerate done skill={row.get('skill_root')} test_mode=1",
                flush=True,
            )
            return row
        print(
            f"[cobras-generate] child={child_index} mode=regenerate start baseline={self.regenerate_baseline_root}",
            flush=True,
        )
        last_error: Exception | None = None
        for attempt_index in range(1, 4):
            attempt_root = child_root / f"attempt_{attempt_index:02d}"
            try:
                if attempt_index > 1:
                    print(
                        f"[cobras-generate] child={child_index} mode=regenerate retry={attempt_index}",
                        flush=True,
                    )
                if self.dataset_adapter is not None:
                    workspace_dir = attempt_root / f"sample{self.regen_sample_size}_skills1" / "workspace_001"
                    skill_root = workspace_dir / "skill"
                    results_path = self.dataset_adapter.generation_results_path(
                        self.regenerate_baseline_root
                    )
                    _run_module(
                        "cobras.scripts.generate_skill_from_results",
                        [
                            "--dataset",
                            self.dataset,
                            "--run-root",
                            str(self.regenerate_baseline_root),
                            "--results-path",
                            str(results_path),
                            "--out-dir",
                            str(skill_root),
                            "--sample-size",
                            str(self.regen_sample_size),
                            "--sample-seed",
                            str(self.seed + child_index + attempt_index * 1000),
                            "--num-skills",
                            "1",
                            "--parallel",
                            "1",
                            "--model",
                            self.teacher_model,
                            "--provider",
                            self.teacher_provider,
                            "--base-url",
                            self.teacher_base_url,
                            "--api-key-env",
                            self.teacher_api_key_env,
                            "--api-key",
                            self.teacher_api_key,
                            "--reasoning-effort",
                            self.teacher_reasoning_effort,
                        ],
                        stream=True,
                    )
                    row = {
                        "skill_name": f"regenerate_{child_index:03d}",
                        "workspace_dir": str(workspace_dir),
                        "skill_root": str(skill_root),
                        "origin": "regenerate",
                    }
                    print(
                        f"[cobras-generate] child={child_index} mode=regenerate done skill={skill_root}",
                        flush=True,
                    )
                    return row

                _run_module(
                    self._module("generate_initial_skill"),
                    [
                        "--baseline-root",
                        str(self.regenerate_baseline_root),
                        "--out-root",
                        str(attempt_root),
                        "--provider",
                        self.teacher_provider,
                        "--sample-size",
                        str(self.regen_sample_size),
                        "--min-success",
                        str(self.regen_min_success),
                        "--min-fail",
                        str(self.regen_min_fail),
                        "--num-skills",
                        "1",
                        "--parallel-skills",
                        "1",
                        "--seed",
                        str(self.seed + child_index + attempt_index * 1000),
                        "--model",
                        self.teacher_model,
                        "--reasoning-effort",
                        self.teacher_reasoning_effort,
                        "--base-url",
                        self.teacher_base_url,
                        "--api-key-env",
                        self.teacher_api_key_env,
                        "--api-key",
                        self.teacher_api_key,
                        "--openai-base-url-env",
                        self.openai_base_url_env,
                    ],
                    stream=True,
                )
                summary_path = attempt_root / f"sample{self.regen_sample_size}_skills1" / "summary.json"
                row = _load_generated_child(summary_path)
                row["origin"] = "regenerate"
                print(
                    f"[cobras-generate] child={child_index} mode=regenerate done skill={row.get('skill_root')}",
                    flush=True,
                )
                return row
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(
                    f"[cobras-generate] child={child_index} mode=regenerate attempt={attempt_index} "
                    f"failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        raise RuntimeError(f"regenerate produced no usable skill after 3 attempts for child={child_index}") from last_error

    def _generate_rollout_mutate(
        self,
        round_root: Path,
        rollout_root: Path,
        child_index: int,
        parent_skill_root: Path | None = None,
    ) -> dict[str, Any]:
        out_root = round_root / "generated" / "rollout_mutate" / f"child_{child_index:03d}"
        if self.test_mode:
            rng = _deterministic_rng(self.seed, round_root.name, child_index, rollout_root.name, "rollout_mutate")
            source = rng.choice(self.initial_arms)
            workspace_dir = out_root / "workspace_001"
            skill_dir = workspace_dir / "skill"
            _copy_skill_tree(Path(str(source["skill_root"])), skill_dir)
            summary_path = out_root / f"{rollout_root.name}_direct_model_{self.teacher_model.replace('/', '_')}" / "summary.json"
            row = {
                "skill_name": str(source["skill_name"]),
                "workspace_dir": str(workspace_dir),
                "skill_root": str(skill_dir),
                "avg_soft_reward": rng.random(),
                "avg_hard_reward": rng.random(),
                "count": 1,
                "summary_path": str(summary_path),
                "origin": "rollout_mutate",
                "test_mode": True,
            }
            write_json(
                summary_path,
                {
                    "generated": [row],
                    "test_mode": True,
                },
            )
            print(
                f"[cobras-generate] child={child_index} mode=rollout_mutate done skill={row.get('skill_root')} test_mode=1",
                flush=True,
            )
            return row
        print(
            f"[cobras-generate] child={child_index} mode=rollout_mutate start pool={rollout_root}",
            flush=True,
        )
        try:
            if self.dataset_adapter is not None:
                if parent_skill_root is None:
                    raise ValueError("Dataset mutation requires the selected parent skill")
                workspace_dir = out_root / "workspace_001"
                skill_root = workspace_dir / "skill"
                _run_module(
                    "cobras.scripts.mutate_skill_from_results",
                    [
                        "--dataset",
                        self.dataset,
                        "--skill-path",
                        str(_skill_entrypoint_path(parent_skill_root)),
                        "--eval-root",
                        str(rollout_root),
                        "--out-root",
                        str(skill_root),
                        "--sample-size",
                        str(self.rollout_sample_size),
                        "--sample-seed",
                        str(self.seed + child_index),
                        "--model",
                        self.teacher_model,
                        "--provider",
                        self.teacher_provider,
                        "--base-url",
                        self.teacher_base_url,
                        "--api-key-env",
                        self.teacher_api_key_env,
                        "--api-key",
                        self.teacher_api_key,
                        "--reasoning-effort",
                        self.teacher_reasoning_effort,
                    ],
                    stream=True,
                )
                row = {
                    "skill_name": f"rollout_mutate_{child_index:03d}",
                    "workspace_dir": str(workspace_dir),
                    "skill_root": str(skill_root),
                    "origin": "rollout_mutate",
                }
                print(
                    f"[cobras-generate] child={child_index} mode=rollout_mutate done skill={skill_root}",
                    flush=True,
                )
                return row

            _run_module(
                self._module("evolve_skill_direct_from_rollouts"),
                [
                    "--rollouts-root",
                    str(rollout_root),
                    "--out-root",
                    str(out_root),
                    "--parallel",
                    "1",
                    "--provider",
                    self.teacher_provider,
                    "--model",
                    self.teacher_model,
                    "--reasoning-effort",
                    self.teacher_reasoning_effort,
                    "--base-url",
                    self.teacher_base_url,
                    "--api-key-env",
                    self.teacher_api_key_env,
                    "--api-key",
                    self.teacher_api_key,
                    "--openai-base-url-env",
                    self.openai_base_url_env,
                ],
                stream=True,
            )
            summary_path = out_root / f"{rollout_root.name}_direct_model_{self.teacher_model.replace('/', '_')}" / "summary.json"
            row = _load_generated_child(summary_path)
            row["origin"] = "rollout_mutate"
            print(
                f"[cobras-generate] child={child_index} mode=rollout_mutate done skill={row.get('skill_root')}",
                flush=True,
            )
            return row
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"rollout_mutate failed for child={child_index} rollout_root={rollout_root}") from exc

    def _generate_crossover(self, round_root: Path, pool_summary: Path, child_index: int) -> dict[str, Any]:
        out_root = round_root / "generated" / "crossover"
        summary_dir_name = f"children{self.crossover_num_children}_k{self.crossover_k}"
        model_summary_dir_name = f"{summary_dir_name}_model_{self.teacher_model.replace('/', '_')}"
        if self.test_mode:
            rng = _deterministic_rng(self.seed, round_root.name, child_index, "crossover")
            source = rng.choice(self.initial_arms)
            workspace_dir = out_root / "workspace_001"
            skill_dir = workspace_dir / "skill"
            _copy_skill_tree(Path(str(source["skill_root"])), skill_dir)
            summary_path = out_root / summary_dir_name / "summary.json"
            row = {
                "skill_name": str(source["skill_name"]),
                "workspace_dir": str(workspace_dir),
                "skill_root": str(skill_dir),
                "avg_soft_reward": rng.random(),
                "avg_hard_reward": rng.random(),
                "count": 1,
                "summary_path": str(summary_path),
                "origin": "crossover",
                "test_mode": True,
            }
            write_json(
                summary_path,
                {
                    "generated": [row],
                    "test_mode": True,
                },
            )
            print(
                f"[cobras-generate] child={child_index} mode=crossover done skill={row.get('skill_root')} test_mode=1",
                flush=True,
            )
            return row
        print(
            f"[cobras-generate] child={child_index} mode=crossover start pool_summary={pool_summary}",
            flush=True,
        )
        try:
            if self.dataset_adapter is not None:
                _run_module(
                    "cobras.scripts.crossover_skill",
                    [
                        "--dataset",
                        self.dataset,
                        "--eval-summary",
                        str(pool_summary),
                        "--out-root",
                        str(out_root),
                        "--num-children",
                        str(self.crossover_num_children),
                        "--parallel",
                        str(self.crossover_parallel),
                        "--k",
                        str(self.crossover_k),
                        "--top-pool-size",
                        str(self.crossover_top_pool_size),
                        "--bottom-pool-size",
                        str(self.crossover_bottom_pool_size),
                        "--score-field",
                        self.crossover_score_field,
                        "--seed",
                        str(self.seed + child_index),
                        "--provider",
                        self.teacher_provider,
                        "--model",
                        self.teacher_model,
                        "--reasoning-effort",
                        self.teacher_reasoning_effort,
                        "--base-url",
                        self.teacher_base_url,
                        "--api-key-env",
                        self.teacher_api_key_env,
                        "--api-key",
                        self.teacher_api_key,
                        "--openai-base-url-env",
                        self.openai_base_url_env,
                    ],
                    stream=True,
                )
                summary_path = out_root / model_summary_dir_name / "summary.json"
                row = _load_generated_child(summary_path)
                row["origin"] = "crossover"
                print(
                    f"[cobras-generate] child={child_index} mode=crossover done skill={row.get('skill_root')}",
                    flush=True,
                )
                return row
            _run_module(
                self._module("crossover_skill"),
                [
                    "--eval-summary",
                    str(pool_summary),
                    "--out-root",
                    str(out_root),
                    "--num-children",
                    str(self.crossover_num_children),
                    "--parallel",
                    str(self.crossover_parallel),
                    "--k",
                    str(self.crossover_k),
                    "--top-pool-size",
                    str(self.crossover_top_pool_size),
                    "--bottom-pool-size",
                    str(self.crossover_bottom_pool_size),
                    "--score-field",
                    self.crossover_score_field,
                    "--seed",
                    str(self.seed + child_index),
                    "--provider",
                    self.teacher_provider,
                    "--model",
                    self.teacher_model,
                    "--reasoning-effort",
                    self.teacher_reasoning_effort,
                    "--base-url",
                    self.teacher_base_url,
                    "--api-key-env",
                    self.teacher_api_key_env,
                    "--api-key",
                    self.teacher_api_key,
                    "--openai-base-url-env",
                    self.openai_base_url_env,
                ],
                stream=True,
            )
            summary_path = out_root / model_summary_dir_name / "summary.json"
            row = _load_generated_child(summary_path)
            row["origin"] = "crossover"
            print(
                f"[cobras-generate] child={child_index} mode=crossover done skill={row.get('skill_root')}",
                flush=True,
            )
            return row
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"crossover failed for child={child_index} pool_summary={pool_summary}") from exc
