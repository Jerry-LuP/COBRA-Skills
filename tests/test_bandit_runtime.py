from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from cobras.cobras_core.bandit_runtime import BanditRuntimeMixin
from cobras.cobras_core.run_state import ArmState
from cobras.scripts.models import LinearUCB


class BanditRuntimeTests(unittest.TestCase):
    @staticmethod
    def _runtime(seed: int) -> BanditRuntimeMixin:
        runtime = BanditRuntimeMixin()
        runtime.seed = seed
        runtime.arms = [
            ArmState(
                arm_id=f"arm_{index:03d}",
                skill_name=f"skill_{index:03d}",
                skill_root=f"/tmp/skill_{index:03d}",
                workspace_dir=f"/tmp/skill_{index:03d}",
            )
            for index in range(1, 11)
        ]
        runtime.embeddings = {
            arm.arm_id: np.eye(10, dtype=np.float64)[index]
            for index, arm in enumerate(runtime.arms)
        }
        return runtime

    @staticmethod
    def _empty_ucb() -> LinearUCB:
        return LinearUCB(10, nu=0.1, lambda_=0.03)

    def test_seeded_tie_break_is_reproducible_and_seed_dependent(self) -> None:
        with patch.dict("os.environ", {"COBRAS_SEEDED_TIE_BREAK": "1"}):
            seed_42_first = self._runtime(42)._score_pool(None, self._empty_ucb())[0]["arm_id"]
            seed_42_repeat = self._runtime(42)._score_pool(None, self._empty_ucb())[0]["arm_id"]
            seed_43_first = self._runtime(43)._score_pool(None, self._empty_ucb())[0]["arm_id"]

        self.assertEqual(seed_42_first, seed_42_repeat)
        self.assertNotEqual(seed_42_first, seed_43_first)

    def test_default_tie_break_preserves_initial_arm_order(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            first = self._runtime(42)._score_pool(None, self._empty_ucb())[0]["arm_id"]

        self.assertEqual(first, "arm_001")


if __name__ == "__main__":
    unittest.main()
