from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from cobras.cobras_core.config import DatasetConfig
from cobras.cobras_core.llm import ProviderConfig


@dataclass(frozen=True)
class EvalConfig:
    """Dataset-neutral inputs for one baseline or skill-injected evaluation."""

    dataset: DatasetConfig
    provider: ProviderConfig
    split: str
    limit: int
    workers: int
    out_root: Path
    skill_path: Path | None = None
    api_key: str = ""
    max_turns: int = 0
    max_completion_tokens: int = 0
    task_timeout: int = 0
    reasoning_effort: str = ""
    mode: str = "multi"
    seed: int = 42
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalArtifacts:
    """Common paths returned by every evaluator without normalizing task rewards."""

    out_root: Path
    summary_path: Path
    results_path: Path
    brief_result_path: Path | None = None

    @property
    def summary(self) -> dict[str, Any]:
        return json.loads(self.summary_path.read_text(encoding="utf-8"))

    def validate(self) -> None:
        required = [self.out_root, self.summary_path, self.results_path]
        missing = [str(path) for path in required if not path.exists()]
        if self.brief_result_path is not None and not self.brief_result_path.exists():
            missing.append(str(self.brief_result_path))
        if missing:
            raise RuntimeError("Evaluator did not produce required artifacts: " + ", ".join(missing))


class TaskEvaluator(Protocol):
    def run_eval(self, config: EvalConfig) -> EvalArtifacts:
        """Run one evaluation while preserving task-specific behavior and output."""

