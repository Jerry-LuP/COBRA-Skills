from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from cobras.cobras_core.run_state import (
    ArmState,
    deterministic_rng as _deterministic_rng,
    load_json as _load_json,
    write_json,
    write_jsonl,
)
from cobras.scripts.result_sampling import load_trial_rows, sample_trials


class PopulationOpsMixin:
    """Build rollout evidence, prune arms, and materialize new arm state."""

    def _make_pool_summary(
        self,
        round_root: Path,
        scored: list[dict[str, Any]] | None = None,
    ) -> Path:
        rows: list[dict[str, Any]] = []
        scored_by_id = {str(row["arm_id"]): row for row in (scored or [])}
        for arm in self.arms:
            include_unevaluated = bool(
                self.dataset_adapter is not None
                and self.dataset_adapter.include_unevaluated_pool_arms
            )
            if arm.latest_reward is None and not include_unevaluated:
                continue
            score_row = scored_by_id.get(arm.arm_id, {})
            pred = float(score_row.get("pred", 0.5))
            bonus = float(score_row.get("bonus", 0.0))
            rows.append(
                {
                    "skill_name": arm.skill_name,
                    "workspace_dir": arm.workspace_dir,
                    "skill_root": arm.skill_root,
                    "avg_soft_reward": (
                        arm.latest_soft_reward
                        if arm.latest_soft_reward is not None
                        else pred
                    ),
                    "avg_hard_reward": arm.latest_hard_reward if arm.latest_hard_reward is not None else 0.0,
                    "bandit_reward": arm.latest_reward if arm.latest_reward is not None else pred,
                    "pred": pred,
                    "bonus": bonus,
                    "ucb_score": float(score_row.get("ucb_score", pred + bonus)),
                    "success_tasks": None,
                    "count": None,
                    "summary_path": arm.latest_eval_summary or "",
                    "origin": arm.origin,
                }
            )
        sort_field = (
            self.dataset_adapter.pool_summary_sort_field
            if self.dataset_adapter is not None
            else "avg_soft_reward"
        )
        rows.sort(key=lambda row: float(row.get(sort_field, 0.0)), reverse=True)
        pool_summary = {
            "skills_root": str(self.initial_pool_root),
            "split": self.selected_eval_split,
            "limit": 0,
            "model": self.model,
            "provider": self.provider,
            "parallel": self.eval_parallel,
            "skill_parallel": 1,
            "results": rows,
        }
        path = round_root / "pool_summary.json"
        write_json(path, pool_summary)
        return path

    def _count_crossover_parents(self, pool_summary: Path) -> int:
        payload = _load_json(pool_summary)
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            return 0
        count = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            skill_root = Path(str(row.get("skill_root", "")))
            if skill_root.exists():
                count += 1
        return count

    def _build_rollout_pool_from_selected_eval(self, round_root: Path, arm: ArmState, round_idx: int) -> Path:
        if self.dataset_adapter is not None:
            if not arm.latest_eval_root:
                raise ValueError(f"Selected arm {arm.arm_id} does not have an eval root")
            eval_root = Path(arm.latest_eval_root)
            results_path = self.dataset_adapter.eval_results_path(eval_root)
            if not results_path.exists():
                raise FileNotFoundError(f"Expected selected-eval results under {results_path}")
            if self.dataset_adapter.rollout_sampling == "task":
                print(
                    f"[cobras-rollout] dataset={self.dataset} arm={arm.arm_id} "
                    f"source={eval_root} policy=task_mutator_sample",
                    flush=True,
                )
                return eval_root
            trial_results_path = results_path
        elif not arm.latest_eval_summary:
            raise ValueError(f"Selected arm {arm.arm_id} does not have a selected eval summary")
        else:
            eval_summary_path = Path(arm.latest_eval_summary)
            trial_results_path = eval_summary_path.parent / "workspace_001" / "trial_results.jsonl"
        if not trial_results_path.exists():
            raise FileNotFoundError(f"Expected trial_results.jsonl under {trial_results_path}")
        sampled_rows = self._sample_rollout_examples(
            trial_results_path,
            seed_parts=(self.seed, round_root.name, arm.arm_id, round_idx, "rollout_pool"),
        )
        pool_root = round_root / "rollout_pool" / f"{arm.arm_id}_sample{len(sampled_rows)}"
        pool_root.mkdir(parents=True, exist_ok=True)
        enriched_rows: list[dict[str, Any]] = []
        for row in sampled_rows:
            enriched = dict(row)
            enriched["skill_root"] = arm.skill_root
            enriched["skill_name"] = arm.skill_name
            enriched["workspace_dir"] = arm.workspace_dir
            enriched["summary_path"] = str(pool_root / "summary.json")
            enriched_rows.append(enriched)
        sampled_ids = [str(row.get("uid") or row.get("id") or "") for row in enriched_rows]
        payload = {
            "rollouts_root": str(trial_results_path),
            "split": self.selected_eval_split,
            "limit": self._selected_eval_limit_value(),
            "sample_size": len(sampled_rows),
            "sampled_uids": sampled_ids,
            "sampled_ids": sampled_ids,
            "results": enriched_rows,
        }
        write_json(pool_root / "summary.json", payload)
        write_jsonl(pool_root / "trial_results.jsonl", enriched_rows)
        print(
            f"[cobras-rollout] arm={arm.arm_id} pool={pool_root} sampled_uids={','.join(sampled_ids)}",
            flush=True,
        )
        return pool_root

    def _sample_rollout_examples(self, eval_summary_path: Path, *, seed_parts: tuple[Any, ...]) -> list[dict[str, Any]]:
        rows = load_trial_rows(eval_summary_path)
        usable = [
            row
            for row in rows
            if (row.get("uid") or row.get("id")) and (
                "question" in row
                or "hard_reward" in row
                or "soft_reward" in row
                or "hard" in row
                or "soft" in row
            )
        ]
        if not usable:
            raise ValueError(f"No usable rollout examples found in {eval_summary_path}")
        success_rows = [row for row in usable if float(row.get("hard_reward", row.get("hard", 0.0))) >= 1.0]
        fail_rows = [row for row in usable if float(row.get("hard_reward", row.get("hard", 0.0))) < 1.0]
        print(
            f"[cobras-rollout] sample source={eval_summary_path} total={len(rows)} usable={len(usable)} "
            f"success={len(success_rows)} fail={len(fail_rows)} target={self.rollout_sample_size} "
            f"min_success={self.rollout_sample_min_success} min_fail={self.rollout_sample_min_fail}",
            flush=True,
        )
        rng = random.Random(_deterministic_rng(*seed_parts).randint(0, 2**31 - 1))
        return sample_trials(
            rng=rng,
            rows=usable,
            sample_size=min(self.rollout_sample_size, len(usable)),
            min_success=min(self.rollout_sample_min_success, len(success_rows)),
            min_fail=min(self.rollout_sample_min_fail, len(fail_rows)),
            success_field="hard_reward",
            success_threshold=1.0,
        )

    def _prune_pool(self, scored: list[dict[str, Any]]) -> tuple[list[ArmState], list[dict[str, Any]]]:
        ranked_ids = [row["arm_id"] for row in sorted(scored, key=lambda row: row["ucb_score"])]
        prune_ids = set(ranked_ids[: self.prune_count])
        kept = [arm for arm in self.arms if arm.arm_id not in prune_ids]
        pruned_rows = [row for row in scored if row["arm_id"] in prune_ids]
        pruned_rows.sort(key=lambda row: row["ucb_score"])
        return kept, pruned_rows

    def _arm_by_id(self, arm_id: str) -> ArmState:
        for arm in self.arms:
            if arm.arm_id == arm_id:
                return arm
        raise KeyError(arm_id)

    def _row_to_arm(self, row: dict[str, Any], *, generation: int, origin: str, parent_ids: list[str]) -> ArmState:
        arm_id = self._allocate_arm_id()
        return ArmState(
            arm_id=arm_id,
            skill_name=str(row.get("skill_name") or Path(str(row["workspace_dir"])).name),
            skill_root=str(Path(str(row["skill_root"])).resolve()),
            workspace_dir=str(Path(str(row["workspace_dir"])).resolve()),
            latest_reward=None,
            latest_soft_reward=None,
            latest_hard_reward=None,
            latest_eval_summary=None,
            latest_eval_root=None,
            latest_train_rollout_root=None,
            origin=origin,
            parent_ids=parent_ids,
            generation=generation,
        )
