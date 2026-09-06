from __future__ import annotations

import json
from pathlib import Path

from cobras.task_envs.eval_runtime import configure_chat_target, load_skill
from cobras.task_envs.searchqa.dataloader import SearchQADataLoader
from cobras.task_envs.eval_types import EvalArtifacts, EvalConfig
from cobras.task_envs.searchqa.rollout import run_batch


def run_eval(config: EvalConfig) -> EvalArtifacts:
    _, reasoning, max_tokens, task_timeout = configure_chat_target(
        config,
        default_reasoning_effort="",
        default_max_completion_tokens=2048,
        default_timeout=600,
    )
    max_turns = config.max_turns or 1
    exec_timeout = int(config.options.get("exec_timeout") or 120)
    skill_path, skill_content = load_skill(config.skill_path)
    out_root = config.out_root.resolve()

    split_path = config.dataset.data_root / config.split
    items = SearchQADataLoader().load_split_items(str(split_path))[: config.limit]
    print(
        f"[searchqa-openrouter-baseline] model={config.provider.model} split={config.split} "
        f"limit={config.limit} workers={config.workers} turns={max_turns} "
        f"reasoning={reasoning or 'off'} out={out_root}",
        flush=True,
    )
    if skill_content.strip():
        print(f"[searchqa-openrouter-baseline] skill={skill_path} chars={len(skill_content)}", flush=True)

    results = run_batch(
        items=items,
        out_root=str(out_root),
        skill_content=skill_content,
        max_turns=max_turns,
        exec_timeout=exec_timeout,
        workers=config.workers,
        max_completion_tokens=max_tokens,
        task_timeout=task_timeout,
    )
    if len(results) != len(items):
        raise RuntimeError(f"SearchQA evaluation incomplete: {len(results)}/{len(items)}")
    brief_path = out_root / "brief_result.jsonl"
    with brief_path.open("w", encoding="utf-8") as handle:
        for row in results:
            hard = int(row.get("hard") or 0)
            soft = float(row.get("soft") or 0.0)
            handle.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "question": row.get("question", ""),
                        "ground_truth": row.get("gold_answers", row.get("gold_answer", [])),
                        "llm_answer": row.get("predicted_answer", ""),
                        "reward": hard,
                        "hard": hard,
                        "soft": soft,
                        "agent_ok": bool(row.get("agent_ok")),
                        "fail_reason": row.get("fail_reason", ""),
                        "n_turns": row.get("n_turns"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    count = len(results)
    hard = sum(int(row.get("hard") or 0) for row in results)
    avg_soft = sum(float(row.get("soft") or 0.0) for row in results) / max(count, 1)
    agent_ok = sum(1 for row in results if row.get("agent_ok"))
    summary = {
        "skill": "baseline" if not skill_path else Path(skill_path).parent.name,
        "skill_file": skill_path,
        "n": count,
        "hard": hard,
        "hard_acc": hard / max(count, 1),
        "avg_soft": avg_soft,
        "agent_ok": agent_ok,
        "out_root": str(out_root),
        "brief_result": str(brief_path),
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"brief_result={brief_path}", flush=True)
    print(
        f"count={count} agent_ok={agent_ok} hard_acc={hard}/{count}={hard / max(count, 1):.4f} "
        f"avg_soft={avg_soft:.4f}",
        flush=True,
    )
    artifacts = EvalArtifacts(out_root, summary_path, out_root / "results.jsonl", brief_path)
    artifacts.validate()
    return artifacts
