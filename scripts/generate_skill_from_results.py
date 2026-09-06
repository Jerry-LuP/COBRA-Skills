from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
from pathlib import Path

from cobras.cobras_core.config import DATASETS, get_dataset
from cobras.cobras_core.llm import child_env_for_provider, resolve_provider
from cobras.cobras_core.paths import COBRAS_ROOT, ensure_project_paths
from cobras.task_envs.cobras_adapter import load_cobras_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a dataset skill from rollout results.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--results-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--num-skills", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default="openai/gpt-5.4")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--reasoning-effort", default="medium")
    return parser.parse_args()


def pool_skill_dir(out_root: Path, index: int) -> Path:
    return out_root / f"seed_{index:02d}"


def pool_sample_seed(base_seed: int, index: int) -> int:
    return base_seed + (index - 1) * 1009


def run_generator(
    *,
    args: argparse.Namespace,
    dataset_module: str,
    base_env: dict[str, str],
    out_dir: Path,
    sample_seed: int,
) -> dict:
    skill_path = out_dir / "skill.md"
    if args.resume and skill_path.exists():
        print(f"[skill-gen] {out_dir.name} resume skip", flush=True)
        return {
            "out_dir": str(out_dir),
            "sample_seed": sample_seed,
            "status": "skipped",
        }
    env = base_env.copy()
    env["OUT_DIR"] = str(out_dir)
    env["SAMPLE_SEED"] = str(sample_seed)
    print(f"[skill-gen] {out_dir.name} start sample_seed={sample_seed}", flush=True)
    completed = subprocess.run(
        [sys.executable, "-m", dataset_module],
        cwd=COBRAS_ROOT,
        env=env,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"generator exited with code {completed.returncode}")
    if not skill_path.exists():
        raise FileNotFoundError(f"Generator did not create {skill_path}")
    print(f"[skill-gen] {out_dir.name} completed", flush=True)
    return {
        "out_dir": str(out_dir),
        "sample_seed": sample_seed,
        "status": "completed",
    }


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
    # COBRAS_ROOT.parent is the import root containing the standalone `cobras` package.
    env["PYTHONPATH"] = f"{COBRAS_ROOT.parent}:{COBRAS_ROOT}"
    env["RUN_ROOT"] = str(args.run_root.resolve())
    env["RESULTS_PATH"] = str((args.results_path or (args.run_root / "results.jsonl")).resolve())
    env["SAMPLE_SIZE"] = str(args.sample_size)
    env["SKILL_GEN_MODEL"] = provider_cfg.model
    env["SKILL_GEN_REASONING_EFFORT"] = args.reasoning_effort
    if dataset_adapter is not None:
        env.update(
            {str(key): str(value) for key, value in dataset_adapter.regenerate_env.items()}
        )
    out_root = args.out_dir.resolve()

    if args.num_skills < 1:
        raise SystemExit("--num-skills must be at least 1")
    if args.parallel < 1:
        raise SystemExit("--parallel must be at least 1")
    if args.num_skills == 1:
        try:
            run_generator(
                args=args,
                dataset_module=dataset_cfg.skill_generator_module,
                base_env=env,
                out_dir=out_root,
                sample_seed=args.sample_seed,
            )
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(str(exc)) from exc
        return

    jobs = [
        (index, pool_sample_seed(args.sample_seed, index), pool_skill_dir(out_root, index))
        for index in range(1, args.num_skills + 1)
    ]
    for _, _, out_dir in jobs:
        skill_path = out_dir / "skill.md"
        if skill_path.exists() and not args.resume:
            raise SystemExit(f"Skill already exists: {skill_path}. Use --resume to keep it.")
        if out_dir.exists() and any(out_dir.iterdir()) and not skill_path.exists():
            raise SystemExit(f"Refusing to overwrite non-empty directory: {out_dir}")

    generated: list[dict] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(args.parallel, args.num_skills)) as executor:
        futures = {
            executor.submit(
                run_generator,
                args=args,
                dataset_module=dataset_cfg.skill_generator_module,
                base_env=env,
                out_dir=out_dir,
                sample_seed=sample_seed,
            ): (index, sample_seed, out_dir)
            for index, sample_seed, out_dir in jobs
        }
        for future in as_completed(futures):
            index, sample_seed, out_dir = futures[future]
            try:
                row = future.result()
                row["index"] = index
                generated.append(row)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "index": index,
                        "out_dir": str(out_dir),
                        "sample_seed": sample_seed,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"[skill-gen] {out_dir.name} failed: {type(exc).__name__}: {exc}", flush=True)

    generated.sort(key=lambda row: row["index"])
    failures.sort(key=lambda row: row["index"])
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "generation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "run_root": str(args.run_root.resolve()),
                "teaching_model": provider_cfg.model,
                "provider": provider_cfg.provider,
                "sample_size": args.sample_size,
                "base_sample_seed": args.sample_seed,
                "num_skills": args.num_skills,
                "parallel": args.parallel,
                "generated": generated,
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[skill-gen] summary={summary_path}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
