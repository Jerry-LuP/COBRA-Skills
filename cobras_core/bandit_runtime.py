from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from cobras.cobras_core.run_state import ArmState, deterministic_rng
from cobras.scripts.embeddings import hash_embed_skill, openrouter_embed_skill
from cobras.scripts.models import LinearUCB, MLPConfig, NumpyMLPRegressor


class BanditRuntimeMixin:
    """Embedding, reward-model, and LinearUCB operations used by the runner."""

    def _embed(self, skill_root: str) -> np.ndarray:
        path = Path(skill_root)
        if self.test_mode:
            return hash_embed_skill(path, dim=self.embedding_dim or 2560)
        if self.embedding_backend == "openrouter":
            return openrouter_embed_skill(
                path,
                model=self.embedding_model,
                base_url=self.embedding_base_url,
                api_key_env=self.embedding_api_key_env,
                cache_dir=self.embedding_cache_root,
            )
        if self.embedding_backend == "hash":
            return hash_embed_skill(path, dim=self.embedding_dim or 256)
        raise ValueError(f"Unsupported embedding backend: {self.embedding_backend}")

    def _ensure_embedding(self, arm: ArmState) -> np.ndarray:
        if arm.arm_id not in self.embeddings:
            print(
                f"[cobras-embed] arm={arm.arm_id} backend={self.embedding_backend} "
                f"model={self.embedding_model} start skill={arm.skill_root}",
                flush=True,
            )
            self.embeddings[arm.arm_id] = self._embed(arm.skill_root)
            print(
                f"[cobras-embed] arm={arm.arm_id} done dim={len(self.embeddings[arm.arm_id])}",
                flush=True,
            )
        return self.embeddings[arm.arm_id]

    def _ensure_history_embedding(self, row: dict[str, Any]) -> np.ndarray:
        arm_id = str(row["arm_id"])
        if arm_id not in self.embeddings:
            print(
                f"[cobras-embed] history_arm={arm_id} backend={self.embedding_backend} "
                f"model={self.embedding_model} start skill={row['skill_root']}",
                flush=True,
            )
            self.embeddings[arm_id] = self._embed(str(row["skill_root"]))
            print(
                f"[cobras-embed] history_arm={arm_id} done dim={len(self.embeddings[arm_id])}",
                flush=True,
            )
        return self.embeddings[arm_id]

    def _refresh_embeddings(self) -> None:
        started = time.perf_counter()
        start_count = len(self.embeddings)
        for arm in self.arms:
            self._ensure_embedding(arm)
        elapsed = time.perf_counter() - started
        print(
            f"[cobras-embed] pool embeddings ready cached_before={start_count} cached_after={len(self.embeddings)}",
            flush=True,
        )
        print(
            f"[cobras-time] round={os.environ.get('COBRAS_ROUND', 'setup')} stage=embeddings "
            f"arms={len(self.arms)} new={len(self.embeddings) - start_count} seconds={elapsed:.3f}",
            flush=True,
        )

    def _train_model(self) -> tuple[NumpyMLPRegressor, dict[str, Any]]:
        started = time.perf_counter()
        self._refresh_embeddings()
        embeddings_ready = time.perf_counter()
        for row in self.history:
            self._ensure_history_embedding(row)
        dim = len(next(iter(self.embeddings.values())))
        if self.history:
            x = np.stack([self.embeddings[row["arm_id"]] for row in self.history])
            y = np.asarray([float(row["reward"]) for row in self.history], dtype=np.float64)
        else:
            x = np.empty((0, dim), dtype=np.float64)
            y = np.empty((0,), dtype=np.float64)
        model = NumpyMLPRegressor(
            dim,
            MLPConfig(
                seed=self.seed,
                lr=1e-3,
                l2=self.mlp_l2,
                epochs=50,
                hidden_dim=64,
                min_samples=1,
                verbose=True,
                log_prefix="[cobras-train]",
            ),
        )
        fit_diag = model.fit(x, y)
        finished = time.perf_counter()
        print(
            f"[cobras-time] round={os.environ.get('COBRAS_ROUND', 'setup')} stage=nn_train "
            f"samples={len(self.history)} dim={dim} "
            f"embedding_s={embeddings_ready - started:.3f} fit_s={finished - embeddings_ready:.3f} "
            f"total_s={finished - started:.3f}",
            flush=True,
        )
        return model, fit_diag

    def _build_ucb(self) -> LinearUCB:
        started = time.perf_counter()
        dim = len(next(iter(self.embeddings.values())))
        ucb = LinearUCB(dim, nu=self.nu, lambda_=self.lambda_)
        history_ready = time.perf_counter()
        inverse_seconds = 0.0
        if self.history:
            history_x = np.stack([self._ensure_history_embedding(row) for row in self.history])
            history_ready = time.perf_counter()
            ucb.fit(history_x)
            inverse_seconds = time.perf_counter() - history_ready
        finished = time.perf_counter()
        print(
            f"[cobras-time] round={os.environ.get('COBRAS_ROUND', 'setup')} stage=ucb_build "
            f"history={len(self.history)} dim={dim} inverse_dim={len(self.history)} "
            f"history_matrix_s={history_ready - started:.3f} inverse_s={inverse_seconds:.3f} "
            f"total_s={finished - started:.3f}",
            flush=True,
        )
        return ucb

    def _score_pool(
        self,
        model: NumpyMLPRegressor | None,
        ucb: LinearUCB,
    ) -> list[dict[str, Any]]:
        if not self.arms:
            return []
        started = time.perf_counter()
        pool_x = np.stack([self._ensure_embedding(arm) for arm in self.arms])
        embeddings_ready = time.perf_counter()
        predictions = np.full(len(self.arms), 0.5, dtype=np.float64) if model is None else model.predict(pool_x)
        predictions_ready = time.perf_counter()
        bonuses = ucb.bonus_many(pool_x)
        bonuses_ready = time.perf_counter()
        scored: list[dict[str, Any]] = []
        for arm, pred, bonus in zip(self.arms, predictions, bonuses):
            scored.append(
                {
                    "arm_id": arm.arm_id,
                    "skill_name": arm.skill_name,
                    "skill_root": arm.skill_root,
                    "workspace_dir": arm.workspace_dir,
                    "pred": pred,
                    "bonus": bonus,
                    "ucb_score": pred + bonus,
                    "latest_reward": arm.latest_reward,
                    "latest_soft_reward": arm.latest_soft_reward,
                    "latest_hard_reward": arm.latest_hard_reward,
                    "origin": arm.origin,
                }
            )
        if os.environ.get("COBRAS_SEEDED_TIE_BREAK", "").strip().lower() in {"1", "true", "yes", "on"}:
            for row in scored:
                row["tie_break"] = deterministic_rng(
                    self.seed,
                    "pool_score_tie",
                    row["arm_id"],
                ).random()
            scored.sort(key=lambda row: (row["ucb_score"], row["tie_break"]), reverse=True)
        else:
            scored.sort(key=lambda row: row["ucb_score"], reverse=True)
        finished = time.perf_counter()
        print(
            f"[cobras-time] round={os.environ.get('COBRAS_ROUND', 'setup')} stage=pool_score "
            f"arms={len(self.arms)} dim={pool_x.shape[1]} "
            f"matrix_s={embeddings_ready - started:.3f} "
            f"predict_s={predictions_ready - embeddings_ready:.3f} "
            f"bonus_s={bonuses_ready - predictions_ready:.3f} "
            f"format_s={finished - bonuses_ready:.3f} total_s={finished - started:.3f}",
            flush=True,
        )
        return scored

    @staticmethod
    def _selected_arm(scored: list[dict[str, Any]]) -> dict[str, Any]:
        if not scored:
            raise RuntimeError("No arms to score")
        return scored[0]
