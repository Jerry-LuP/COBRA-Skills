from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from cobras.scripts.models import LinearUCB


class LinearUCBTests(unittest.TestCase):
    def test_bonus_many_matches_diagonal_formula(self) -> None:
        history = np.asarray([[1.0, 0.5, -0.5], [0.25, 1.5, 0.75]])
        candidates = np.asarray([[0.5, 0.0, 1.0], [1.0, -1.0, 0.25]])
        lambda_ = 0.3
        nu = 0.2

        ucb = LinearUCB(3, nu=nu, lambda_=lambda_)
        ucb.fit(history)

        a_inv = np.linalg.inv(lambda_ * np.eye(3) + history.T @ history)
        expected = nu * np.sqrt(np.diag(candidates @ a_inv @ candidates.T))
        np.testing.assert_allclose(ucb.bonus_many(candidates), expected)

    def test_update_defers_inverse_until_bonus_is_requested(self) -> None:
        ucb = LinearUCB(3, nu=0.2, lambda_=0.3)
        with patch("cobras.scripts.models.np.linalg.inv", wraps=np.linalg.inv) as inverse:
            ucb.update(np.asarray([1.0, 0.0, 0.0]))
            ucb.update(np.asarray([0.0, 1.0, 0.0]))
            self.assertEqual(inverse.call_count, 0)

            ucb.bonus_many(np.eye(3))
            self.assertEqual(inverse.call_count, 1)

            ucb.bonus(np.asarray([1.0, 1.0, 0.0]))
            self.assertEqual(inverse.call_count, 1)

            ucb.update(np.asarray([0.0, 0.0, 1.0]))
            ucb.bonus(np.asarray([1.0, 1.0, 1.0]))
            self.assertEqual(inverse.call_count, 2)

    def test_high_dimensional_empty_history_uses_analytic_bonus(self) -> None:
        dim = 2560
        nu = 0.1
        lambda_ = 0.03
        candidates = np.eye(10, dim, dtype=np.float64)

        ucb = LinearUCB(dim, nu=nu, lambda_=lambda_)

        self.assertEqual(ucb._history.shape, (0, dim))
        self.assertEqual(ucb._dual_inv.shape, (0, 0))
        np.testing.assert_allclose(
            ucb.bonus_many(candidates),
            np.full(10, nu / np.sqrt(lambda_)),
        )

    def test_dual_inverse_is_history_sized(self) -> None:
        history = np.asarray([[1.0, 0.5, -0.5], [0.25, 1.5, 0.75]])
        ucb = LinearUCB(3, nu=0.2, lambda_=0.3)

        ucb.fit(history)

        self.assertEqual(ucb._dual_inv.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
