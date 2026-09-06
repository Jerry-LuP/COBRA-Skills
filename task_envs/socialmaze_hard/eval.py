from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from cobras.task_envs.eval_runtime import configure_chat_target, load_skill, safe_name
from cobras.task_envs.eval_types import EvalArtifacts, EvalConfig
from cobras.task_envs.socialmaze_hard.scripts.run_socialmaze_baseline import (
    load_items,
    run_one,
    write_outputs,
)


def run_eval(config: EvalConfig) -> EvalArtifacts:
    _, reasoning, max_tokens, timeout = configure_chat_target(
        config,
        default_reasoning_effort="",
        default_max_completion_tokens=2048,
        default_timeout=240,
    )
    skill_path, skill_content = load_skill(config.skill_path)
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    default_data_path = config.dataset.data_root / f"{config.split}.jsonl"
    configured_data_path = str(config.options.get("data_path", "")).strip()
    data_path = configured_data_path or (str(default_data_path) if default_data_path.exists() else "")
    sample_seed = config.options.get("sample_seed")
    seed = int(sample_seed) if sample_seed is not None else None

    print(
        f"[socialmaze] provider={config.provider.provider} model={config.provider.model} "
        f"split={config.split} data_path={data_path or 'none'} limit={config.limit} "
        f"workers={config.workers} reasoning={reasoning or 'off'} "
        f"skill={skill_path or 'none'} out={out_root}",
        flush=True,
    )
    items = load_items(config.split, config.limit, seed, data_path)
    print(f"[socialmaze] loaded_items={len(items)}", flush=True)

    results: list[dict] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                row,
                out_root=out_root,
                skill_content=skill_content,
                max_completion_tokens=max_tokens,
                timeout=timeout,
            ): row
            for row in items
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done = len(results)
            hard = sum(int(item["hard"]) for item in results)
            soft = sum(float(item["soft"]) for item in results) / max(done, 1)
            print(
                f"[socialmaze] {done}/{len(items)} id={result['id']} hard={result['hard']} "
                f"soft={result['soft']:.2f} reward_soft={soft:.3f} hard_acc={hard / max(done, 1):.3f}",
                flush=True,
            )

    order = {safe_name(row.get("id")): index for index, row in enumerate(items)}
    results.sort(key=lambda row: order.get(row["id"], 10**9))
    run_id = str(config.options.get("run_id", out_root.name))
    output_config = {
        "dataset": "MBZUAI/SocialMaze",
        "split": config.split,
        "data_path": data_path,
        "limit": config.limit,
        "workers": config.workers,
        "provider": config.provider.provider,
        "model": config.provider.model,
        "reasoning_effort": reasoning,
        "skill_path": skill_path,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "failures": failures,
    }
    write_outputs(out_root, results, output_config)
    artifacts = EvalArtifacts(
        out_root=out_root,
        summary_path=out_root / "summary.json",
        results_path=out_root / "results.jsonl",
        brief_result_path=out_root / "brief_result.jsonl",
    )
    artifacts.validate()
    return artifacts
