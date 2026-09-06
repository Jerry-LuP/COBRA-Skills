from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from cobras.cobras_core.config import DATASETS, get_dataset
from cobras.cobras_core.llm import child_env_for_provider, resolve_provider
from cobras.cobras_core.paths import COBRAS_ROOT, ensure_project_paths
from cobras.task_envs.cobras_adapter import load_cobras_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mutate a dataset skill from rollout results.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--skill-path", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--model", default="openai/gpt-5.4")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--reasoning-effort", default="medium")
    return parser.parse_args()


def main() -> None:
    ensure_project_paths()
    args = parse_args()
    dataset_cfg = get_dataset(args.dataset)
    dataset_adapter = load_cobras_adapter(args.dataset)
    provider_cfg = resolve_provider(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    env = os.environ.copy()
    env.update(child_env_for_provider(provider_cfg, api_key=args.api_key, role="teacher"))
    env["PYTHONPATH"] = f"{COBRAS_ROOT.parent}:{COBRAS_ROOT}:{env.get('PYTHONPATH', '')}"
    env["SKILL_PATH"] = str(args.skill_path.resolve())
    env["EVAL_ROOT"] = str(args.eval_root.resolve())
    env["OUT_ROOT"] = str(args.out_root.resolve())
    env["PARENT_SKILL_PATH"] = str(args.skill_path.resolve())
    env["RUN_ROOT"] = str(args.eval_root.resolve())
    env["OUT_DIR"] = str(args.out_root.resolve())
    env["MUTATION_SAMPLE_SIZE"] = str(args.sample_size)
    env["SAMPLE_SIZE"] = str(args.sample_size)
    env["SAMPLE_SEED"] = str(args.sample_seed)
    env["SKILL_MUTATE_MODEL"] = provider_cfg.model
    env["SKILL_MUTATE_REASONING_EFFORT"] = args.reasoning_effort
    env["MUTATION_MODEL"] = provider_cfg.model
    env["MUTATION_REASONING_EFFORT"] = args.reasoning_effort
    env["MUTATION_SAMPLE_SEED"] = str(args.sample_seed)
    if dataset_adapter is not None:
        env.update(
            dataset_adapter.prepare_mutation_inputs(
                skill_path=args.skill_path.resolve(),
                eval_root=args.eval_root.resolve(),
                out_root=args.out_root.resolve(),
            )
        )
    completed = subprocess.run(
        [sys.executable, "-m", dataset_cfg.skill_mutator_module],
        cwd=COBRAS_ROOT,
        env=env,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
