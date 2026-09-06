from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cobras.cobras_core.run_state import ArmState, append_jsonl, format_score as _fmt_score, write_json
from cobras.cobras_core.token_usage import write_usage_reports
from cobras.scripts.models import LinearUCB, NumpyMLPRegressor


class RunnerLoopMixin:
    """The round-by-round COBRAS control flow."""

    def run(self) -> dict[str, Any]:
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.out_root / "update_schedule.json", self.update_schedule.to_json())
        start_round = self._resume_state_if_available()
        print(
            f"[cobras] start rounds={self.rounds} pool={self.pool_size} prune={self.prune_count} "
            f"dataset={self.dataset or self.task_package} reward={self.reward_field} "
            f"crossover_threshold={self.crossover_threshold} student={self.provider}:{self.model} "
            f"teacher={self.teacher_provider}:{self.teacher_model} "
            f"max_turns={self.max_turns or 'adapter-default'} "
            f"test_mode={int(self.test_mode)} resume={int(self.resume)} start_round={start_round} "
            f"update_schedule={self.update_schedule.tag} "
            f"update_rounds={list(self.update_schedule.update_rounds)}",
            flush=True,
        )

        model: NumpyMLPRegressor | None = None
        ucb: LinearUCB | None = None
        fit_diag: dict[str, Any] = {"n_samples": 0, "trained": False, "reason": "no_initial_train", "prior": 0.5}
        if self.resume and self.history:
            print(
                f"[cobras] resume warm-start train from history n={len(self.history)}",
                flush=True,
            )
            model, fit_diag = self._train_model()
            print(
                f"[cobras] resume warm-start done n={fit_diag.get('n_samples')} trained={fit_diag.get('trained')} "
                f"reason={fit_diag.get('reason')} prior={float(fit_diag.get('prior', 0.0)):.4f}",
                flush=True,
            )
            ucb = self._build_ucb()

        for round_idx in range(start_round, self.rounds):
            os.environ["COBRAS_ROUND"] = str(round_idx)
            round_root = self.rounds_dir / f"round_{round_idx:03d}"
            round_root.mkdir(parents=True, exist_ok=True)
            if model is None:
                print(f"[cobras] round={round_idx:03d} no carried model", flush=True)
            else:
                print(f"[cobras] round={round_idx:03d} use carried model", flush=True)
            self._refresh_embeddings()
            if ucb is None:
                ucb = self._build_ucb()
            scored_before = self._score_pool(model, ucb)
            print(f"[cobras] round={round_idx:03d} top candidates before eval:", flush=True)
            for rank, row in enumerate(scored_before[: min(10, len(scored_before))], start=1):
                print(f"  {rank:02d}. {_fmt_score(row)}", flush=True)
            print(f"[cobras] round={round_idx:03d} bottom candidates before eval:", flush=True)
            for rank, row in enumerate(scored_before[-min(5, len(scored_before)) :], start=1):
                print(f"  {rank:02d}. {_fmt_score(row)}", flush=True)
            selected_row = self._selected_arm(scored_before)
            selected_arm = self._arm_by_id(selected_row["arm_id"])
            print(
                f"[cobras] round={round_idx:03d} selected={selected_row['arm_id']} "
                f"pred={selected_row['pred']:.4f} bonus={selected_row['bonus']:.4f} "
                f"score={selected_row['ucb_score']:.4f}",
                flush=True,
            )

            print(f"[cobras] round={round_idx:03d} eval selected arm", flush=True)
            selected_arm, eval_row = self._evaluate_selected_arm(round_root, selected_arm)

            print(f"[cobras] round={round_idx:03d} retrain after eval", flush=True)
            model, post_eval_fit_diag = self._train_model()
            print(
                f"[cobras] round={round_idx:03d} retrain_done "
                f"n={post_eval_fit_diag.get('n_samples')} trained={post_eval_fit_diag.get('trained')} "
                f"reason={post_eval_fit_diag.get('reason')} prior={float(post_eval_fit_diag.get('prior', 0.0)):.4f}",
                flush=True,
            )

            ucb = self._build_ucb()
            scored_after = self._score_pool(model, ucb)
            scored_before_by_id = {row["arm_id"]: row for row in scored_before}
            print(f"[cobras] round={round_idx:03d} top candidates after eval+retrain:", flush=True)
            for rank, row in enumerate(scored_after[: min(10, len(scored_after))], start=1):
                prev = scored_before_by_id.get(row["arm_id"], {})
                delta_bonus = float(row["bonus"]) - float(prev.get("bonus", row["bonus"]))
                delta_score = float(row["ucb_score"]) - float(prev.get("ucb_score", row["ucb_score"]))
                print(
                    f"  {rank:02d}. {_fmt_score(row)} "
                    f"delta_bonus={delta_bonus:+.4f} delta_score={delta_score:+.4f}",
                    flush=True,
                )
            pool_summary_path = self._make_pool_summary(round_root, scored_after)
            iteration = round_idx + 1
            do_mutation_round = self.update_schedule.should_update(iteration)
            kept_arms = list(self.arms)
            pruned_rows: list[dict[str, Any]] = []
            rollout_root: Path | None = None
            generated: list[dict[str, Any]] = []
            generation_modes: list[str] = []
            if do_mutation_round:
                kept_arms, pruned_rows = self._prune_pool(scored_after)
                print(f"[cobras] round={round_idx:03d} pruned arms:", flush=True)
                for rank, row in enumerate(pruned_rows, start=1):
                    print(f"  {rank:02d}. {_fmt_score(row)}", flush=True)

                print(f"[cobras] round={round_idx:03d} build rollout pool from selected eval", flush=True)
                rollout_root = self._build_rollout_pool_from_selected_eval(round_root, selected_arm, round_idx)
                selected_arm.latest_train_rollout_root = str(rollout_root)

                crossover_parent_count = self._count_crossover_parents(pool_summary_path)
                required_crossover_parents = max(self.crossover_top_pool_size, self.crossover_bottom_pool_size)
                need_crossover = (
                    len(self.dynamic_unique_skill_roots) > self.crossover_threshold
                    and crossover_parent_count >= required_crossover_parents
                )
                generation_modes = ["regenerate", "regenerate", "rollout_mutate"]
                if need_crossover:
                    generation_modes[1] = "crossover"
                print(
                    f"[cobras] round={round_idx:03d} generation_modes={generation_modes} "
                    f"crossover_parents={crossover_parent_count}/{required_crossover_parents} "
                    f"evaluated_unique={len(self.dynamic_unique_skill_roots)} threshold={self.crossover_threshold}",
                    flush=True,
                )

                with ThreadPoolExecutor(max_workers=max(1, self.generate_parallel)) as executor:
                    futures = {}
                    for child_index, mode in enumerate(generation_modes):
                        print(
                            f"[cobras] round={round_idx:03d} submit child={child_index} mode={mode} "
                            f"parent={selected_arm.arm_id}",
                            flush=True,
                        )
                        if mode == "regenerate":
                            fut = executor.submit(self._generate_regenerate, round_root, child_index)
                        elif mode == "rollout_mutate":
                            fut = executor.submit(
                                self._generate_rollout_mutate,
                                round_root,
                                rollout_root,
                                child_index,
                                Path(selected_arm.skill_root),
                            )
                        elif mode == "crossover":
                            fut = executor.submit(self._generate_crossover, round_root, pool_summary_path, child_index)
                        else:
                            raise ValueError(mode)
                        futures[fut] = mode
                    for fut in as_completed(futures):
                        mode = futures[fut]
                        row = fut.result()
                        generated.append(row)
                        print(
                            f"[cobras] round={round_idx:03d} child completed mode={mode} "
                            f"skill={row.get('skill_root')}",
                            flush=True,
                        )

                generated.sort(key=lambda row: row["origin"])
                new_arms: list[ArmState] = []
                for idx, row in enumerate(generated, start=1):
                    origin = str(row.get("origin", "generated"))
                    parent_ids = [selected_arm.arm_id]
                    new_arms.append(
                        self._row_to_arm(
                            row,
                            generation=round_idx + 1,
                            origin=origin,
                            parent_ids=parent_ids,
                        )
                    )

                self.arms = kept_arms + new_arms
                self._refresh_embeddings()
                print(f"[cobras] round={round_idx:03d} new arms:", flush=True)
                for arm in new_arms:
                    print(
                        f"  {arm.arm_id} origin={arm.origin} parent={','.join(arm.parent_ids or [])} "
                        f"skill={arm.skill_root}",
                        flush=True,
                    )
            else:
                print(
                    f"[cobras] round={round_idx:03d} mutation skipped "
                    f"schedule={self.update_schedule.tag} "
                    f"next={self.update_schedule.next_update_after(iteration)}",
                    flush=True,
                )

            print(f"[cobras] round={round_idx:03d} pool after update:", flush=True)
            for arm in self.arms:
                latest = "NA" if arm.latest_reward is None else f"{arm.latest_reward:.4f}"
                print(f"  {arm.arm_id} latest={latest} origin={arm.origin} skill={arm.skill_name}", flush=True)

            round_payload = {
                "round": round_idx,
                "iteration": iteration,
                "population_update": do_mutation_round,
                "update_schedule": self.update_schedule.tag,
                "selected_arm": selected_row,
                "selected_eval": eval_row,
                "fit_diagnostics": post_eval_fit_diag,
                "selected_train_rollout_root": str(rollout_root) if rollout_root is not None else "",
                "pool_summary_path": str(pool_summary_path),
                "pruned": pruned_rows,
                "generation_modes": generation_modes,
                "generated": generated,
                "top_candidates": scored_after[:5],
            }
            write_json(round_root / "selection.json", round_payload)
            write_json(round_root / "pool_state.json", [arm.to_json() for arm in self.arms])
            append_jsonl(self.out_root / "history.jsonl", round_payload)
            write_usage_reports(self.out_root, initial_pool_root=self.initial_pool_root)
            print(f"[cobras] round={round_idx:03d} done", flush=True)

        summary = {
            "initial_pool_root": str(self.initial_pool_root),
            "initial_eval_summary": str(self.initial_eval_summary),
            "out_root": str(self.out_root),
            "rounds": self.rounds,
            "pool_size": self.pool_size,
            "prune_count": self.prune_count,
            "update_schedule": self.update_schedule.to_json(),
            "crossover_threshold": self.crossover_threshold,
            "train_rollout_tasks": self.train_rollout_tasks,
            "max_turns": self.max_turns,
            "model": self.model,
            "provider": self.provider,
            "dataset": self.dataset,
            "reward_field": self.reward_field,
            "teacher_model": self.teacher_model,
            "teacher_provider": self.teacher_provider,
            "history": self.history,
            "final_pool": [arm.to_json() for arm in self.arms],
        }
        write_json(self.out_root / "summary.json", summary)
        print(f"[cobras] finished out_root={self.out_root}", flush=True)
        return summary
