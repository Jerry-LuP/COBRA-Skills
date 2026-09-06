from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class PopulationUpdateSchedule:
    """Deterministic 1-based schedule for prune and skill generation updates."""

    mode: str
    total_rounds: int
    min_interval: int = 3
    log_threshold: float = 0.35
    cooldown_rounds: int | None = None
    max_interval: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "log"}:
            raise ValueError(f"Unsupported update schedule: {self.mode}")
        if self.total_rounds < 1:
            raise ValueError("total_rounds must be positive")
        if self.min_interval < 1:
            raise ValueError("min_interval must be positive")
        if self.log_threshold <= 0:
            raise ValueError("log_threshold must be positive")
        if self.cooldown_rounds is not None and self.cooldown_rounds < 0:
            raise ValueError("cooldown_rounds must be non-negative")
        if self.max_interval is not None and self.max_interval < self.min_interval:
            raise ValueError("max_interval must be at least min_interval")

    @property
    def resolved_cooldown_rounds(self) -> int:
        if self.mode == "fixed":
            return 0
        if self.cooldown_rounds is not None:
            return self.cooldown_rounds
        return math.ceil(math.log1p(self.total_rounds))

    @property
    def resolved_max_interval(self) -> int:
        if self.max_interval is not None:
            return self.max_interval
        return self.min_interval + self.resolved_cooldown_rounds

    @property
    def active_horizon(self) -> int:
        if self.mode == "fixed":
            return self.total_rounds
        return max(self.min_interval, self.total_rounds - self.resolved_cooldown_rounds)

    @property
    def update_rounds(self) -> tuple[int, ...]:
        if self.mode == "fixed":
            return tuple(range(self.min_interval, self.total_rounds + 1, self.min_interval))

        horizon = min(self.total_rounds, self.active_horizon)
        if self.min_interval > horizon:
            return ()
        updates = [self.min_interval]
        while True:
            last = updates[-1]
            log_candidate = math.floor(math.exp(self.log_threshold) * last) + 1
            candidate = min(
                last + self.resolved_max_interval,
                max(last + self.min_interval, log_candidate),
            )
            if candidate > horizon:
                break
            updates.append(candidate)
        return tuple(updates)

    @property
    def tag(self) -> str:
        return "prune_log" if self.mode == "log" else f"prune_{self.min_interval}"

    def should_update(self, iteration: int) -> bool:
        return iteration in self.update_rounds

    def next_update_after(self, iteration: int) -> int | None:
        return next((value for value in self.update_rounds if value > iteration), None)

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "total_rounds": self.total_rounds,
            "min_interval": self.min_interval,
            "log_threshold": self.log_threshold,
            "cooldown_rounds": self.resolved_cooldown_rounds,
            "max_interval": self.resolved_max_interval,
            "active_horizon": self.active_horizon,
            "update_rounds": list(self.update_rounds),
            "tag": self.tag,
        }


def schedule_from_args(args: argparse.Namespace) -> PopulationUpdateSchedule:
    return PopulationUpdateSchedule(
        mode=args.prune_schedule,
        total_rounds=args.rounds,
        min_interval=args.prune_interval,
        log_threshold=args.prune_log_threshold,
        cooldown_rounds=args.prune_cooldown_rounds,
        max_interval=args.prune_max_interval,
    )


def add_schedule_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prune-schedule", choices=["fixed", "log"], default="log")
    parser.add_argument("--prune-interval", type=int, default=3, help="Fixed interval or minimum log-schedule dwell time.")
    parser.add_argument("--prune-log-threshold", type=float, default=0.35)
    parser.add_argument("--prune-cooldown-rounds", type=int, default=None)
    parser.add_argument("--prune-max-interval", type=int, default=None)
