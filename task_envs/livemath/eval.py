from __future__ import annotations

import argparse
from pathlib import Path

from cobras.task_envs.eval_runtime import api_key_for
from cobras.task_envs.eval_types import EvalArtifacts, EvalConfig
from cobras.task_envs.livemath.scripts.run_baseline_single import run


def run_eval(config: EvalConfig) -> EvalArtifacts:
    provider = config.provider.provider
    runner_provider = provider if provider in {"openrouter", "yunwu", "custom", "local"} else "custom"
    args = argparse.Namespace(
        provider=runner_provider,
        split=config.split,
        split_root=config.dataset.data_root,
        limit=config.limit,
        trials=int(config.options.get("trials", 1)),
        parallel=config.workers,
        seed=config.seed,
        model=config.provider.model,
        max_completion_tokens=config.max_completion_tokens or 16384,
        use_theorem=bool(config.options.get("use_theorem", False)),
        use_sketch=bool(config.options.get("use_sketch", False)),
        prompt_version=str(config.options.get("prompt_version", "direct")),
        skill_path=config.skill_path or Path(""),
        base_url=config.provider.base_url,
        api_key_env=config.provider.api_key_env,
        api_key=api_key_for(config),
        openai_base_url_env="OPENAI_BASE_URL",
        out_root=config.out_root.parent,
        run_name=config.out_root.name,
    )
    run_root = run(args)
    artifacts = EvalArtifacts(
        out_root=run_root,
        summary_path=run_root / "summary.json",
        results_path=run_root / "trial_results.jsonl",
    )
    artifacts.validate()
    return artifacts
