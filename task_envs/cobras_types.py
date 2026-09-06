from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from cobras.cobras_core.config import DatasetConfig
from cobras.cobras_core.llm import ProviderConfig


@dataclass(frozen=True)
class CobrasDriverContext:
    args: Namespace
    dataset: DatasetConfig
    provider: ProviderConfig
    teacher_provider: ProviderConfig
    common_args: tuple[str, ...]
    out_root: Path
    baseline_root: Path
    initial_eval_summary: Path
    initial_pool_root: Path
