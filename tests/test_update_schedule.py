from __future__ import annotations

import unittest

from cobras.cobras_core.update_schedule import PopulationUpdateSchedule


class PopulationUpdateScheduleTests(unittest.TestCase):
    def test_log_schedule_for_thirty_rounds(self) -> None:
        schedule = PopulationUpdateSchedule(mode="log", total_rounds=30)
        self.assertEqual(schedule.resolved_cooldown_rounds, 4)
        self.assertEqual(schedule.resolved_max_interval, 7)
        self.assertEqual(schedule.active_horizon, 26)
        self.assertEqual(schedule.update_rounds, (3, 6, 9, 13, 19, 26))
        self.assertEqual(schedule.tag, "prune_log")
        self.assertFalse(any(schedule.should_update(t) for t in range(27, 31)))

    def test_fixed_schedule_preserves_periodic_updates(self) -> None:
        schedule = PopulationUpdateSchedule(mode="fixed", total_rounds=30, min_interval=3)
        self.assertEqual(schedule.update_rounds, tuple(range(3, 31, 3)))
        self.assertEqual(schedule.tag, "prune_3")

    def test_explicit_bounds_override_horizon_defaults(self) -> None:
        schedule = PopulationUpdateSchedule(
            mode="log",
            total_rounds=30,
            cooldown_rounds=5,
            max_interval=6,
        )
        self.assertEqual(schedule.active_horizon, 25)
        self.assertEqual(schedule.resolved_max_interval, 6)
        self.assertTrue(all(t <= 25 for t in schedule.update_rounds))


if __name__ == "__main__":
    unittest.main()
