from __future__ import annotations

from importlib import import_module
from typing import Callable

from cobras.cobras_core.config import get_dataset
from cobras.task_envs.eval_types import EvalArtifacts, EvalConfig


EvalFunction = Callable[[EvalConfig], EvalArtifacts]


def load_evaluator(dataset: str) -> EvalFunction:
    """Load a dataset evaluator without adding dataset branches to orchestration."""

    cfg = get_dataset(dataset)
    module = import_module(cfg.evaluator_module)
    run_eval = getattr(module, "run_eval", None)
    if not callable(run_eval):
        raise RuntimeError(f"{cfg.evaluator_module} must export run_eval(config)")
    return run_eval

