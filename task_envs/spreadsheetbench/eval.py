from __future__ import annotations

import json
from pathlib import Path

from cobras.cobras_core.paths import COBRAS_ROOT
from cobras.task_envs.eval_runtime import configure_chat_target, load_skill
from cobras.task_envs.eval_types import EvalArtifacts, EvalConfig
from cobras.task_envs.spreadsheetbench.rollout import run_spreadsheet_batch_codegen


def _load_items(config: EvalConfig) -> tuple[list[dict], Path]:
    split_path = COBRAS_ROOT / "data" / "spreadsheetbench_id_split" / config.split / "items.json"
    split_items = json.loads(split_path.read_text(encoding="utf-8"))[: config.limit]
    data_root = config.dataset.data_root
    dataset_path = data_root / "dataset.json"
    if not dataset_path.exists():
        dataset_path = data_root / "spreadsheetbench_verified_400" / "dataset.json"
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    by_id = {str(row["id"]): row for row in rows}
    items: list[dict] = []
    for entry in split_items:
        task_id = str(entry["id"])
        merged = dict(by_id.get(task_id, {}))
        merged.update(entry)
        if not merged.get("instruction"):
            prompt_path = data_root / str(merged.get("spreadsheet_path", f"spreadsheet/{task_id}")) / "prompt.txt"
            if prompt_path.exists():
                merged["instruction"] = prompt_path.read_text(encoding="utf-8").strip()
        if not merged.get("instruction"):
            raise RuntimeError(f"Missing instruction for task {task_id}")
        items.append(merged)
    return items, data_root


def run_eval(config: EvalConfig) -> EvalArtifacts:
    _, reasoning, max_tokens, task_timeout = configure_chat_target(
        config,
        default_reasoning_effort="",
        default_max_completion_tokens=8192,
        default_timeout=600,
    )
    mode = config.mode.strip().lower() or "multi"
    if mode not in {"single", "multi"}:
        raise ValueError(f"SpreadsheetBench mode must be single or multi, got {mode!r}")
    max_turns = config.max_turns or 30
    use_eval_feedback = bool(config.options.get("use_eval_feedback", False))
    skill_path, skill_content = load_skill(config.skill_path)
    out_root = config.out_root.resolve()
    items, data_root = _load_items(config)

    print(
        f"[spreadsheet-eval] provider={config.provider.provider} model={config.provider.model} "
        f"split={config.split} limit={config.limit} workers={config.workers} mode={mode} "
        f"turns={max_turns} timeout={task_timeout} reasoning={reasoning or 'off'} "
        f"eval_feedback={use_eval_feedback} skill={skill_path or 'none'} out={out_root}",
        flush=True,
    )
    results = run_spreadsheet_batch_codegen(
        items=items,
        data_root=str(data_root),
        out_root=str(out_root),
        skill_content=skill_content,
        mode=mode,
        max_turns=max_turns,
        max_completion_tokens=max_tokens,
        max_api_workers=config.workers,
        task_timeout=task_timeout,
        use_eval_feedback=use_eval_feedback,
    )
    if len(results) != len(items):
        raise RuntimeError(
            f"SpreadsheetBench evaluation incomplete: {len(results)}/{len(items)}"
        )

    results_path = out_root / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    brief_path = out_root / "brief_result.jsonl"
    with brief_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "instruction_type": row.get("instruction_type", ""),
                        "task_type": row.get("task_type", ""),
                        "hard": int(row.get("hard") or 0),
                        "soft": float(row.get("soft") or 0.0),
                        "n_pass": int(row.get("n_pass") or 0),
                        "n_cases": int(row.get("n_cases") or 0),
                        "n_turns": int(row.get("n_turns") or 0),
                        "llm_ok": bool(row.get("llm_ok")),
                        "code_ok": bool(row.get("code_ok")),
                        "exec_ok": bool(row.get("exec_ok")),
                        "fail_reason": row.get("fail_reason", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    count = len(results)
    hard = sum(int(row.get("hard") or 0) for row in results)
    avg_soft = sum(float(row.get("soft") or 0.0) for row in results) / max(count, 1)
    llm_ok = sum(1 for row in results if row.get("llm_ok"))
    code_ok = sum(1 for row in results if row.get("code_ok"))
    exec_ok = sum(1 for row in results if row.get("exec_ok"))
    summary = {
        "n": count,
        "hard": hard,
        "hard_acc": hard / max(count, 1),
        "avg_soft": avg_soft,
        "llm_ok": llm_ok,
        "code_ok": code_ok,
        "exec_ok": exec_ok,
        "mode": mode,
        "split": config.split,
        "limit": config.limit,
        "workers": config.workers,
        "max_turns": max_turns,
        "task_timeout": task_timeout,
        "reasoning_effort": reasoning or "off",
        "out_root": str(out_root),
        "brief_result": str(brief_path),
        "results_jsonl": str(results_path),
        "skill_path": skill_path,
        "provider": config.provider.provider,
        "model": config.provider.model,
        "base_url": config.provider.base_url,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"brief_result={brief_path}", flush=True)
    print(
        f"count={count} llm_ok={llm_ok} code_ok={code_ok} exec_ok={exec_ok} "
        f"hard_acc={hard}/{count}={hard / max(count, 1):.4f} avg_soft={avg_soft:.4f}",
        flush=True,
    )
    artifacts = EvalArtifacts(out_root, summary_path, results_path, brief_path)
    artifacts.validate()
    return artifacts
