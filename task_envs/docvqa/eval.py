from __future__ import annotations

import json
from pathlib import Path

from cobras.task_envs.docvqa.dataloader import DocVQADataLoader
from cobras.task_envs.docvqa.rollout import run_batch
from cobras.task_envs.eval_runtime import configure_chat_target, load_skill
from cobras.task_envs.eval_types import EvalArtifacts, EvalConfig


def run_eval(config: EvalConfig) -> EvalArtifacts:
    _, reasoning, max_tokens, task_timeout = configure_chat_target(
        config,
        default_reasoning_effort="",
        default_max_completion_tokens=4096,
        default_timeout=600,
    )
    max_turns = config.max_turns or 1
    exec_timeout = int(config.options.get("exec_timeout") or 300)
    image_detail = str(config.options.get("image_detail", "auto"))
    skill_path, skill_content = load_skill(config.skill_path)
    out_root = config.out_root.resolve()

    split_path = config.dataset.data_root / "splits" / config.split
    items = DocVQADataLoader().load_split_items(str(split_path))[: config.limit]
    print(
        f"[docvqa-eval] model={config.provider.model} split={config.split} "
        f"limit={config.limit} workers={config.workers} turns={max_turns} "
        f"reasoning={reasoning or 'off'} skill={skill_path or 'none'} out={out_root}",
        flush=True,
    )
    results = run_batch(
        items,
        str(out_root),
        skill_content,
        max_turns=max_turns,
        exec_timeout=exec_timeout,
        workers=config.workers,
        image_detail=image_detail,
        max_completion_tokens=max_tokens,
        task_timeout=task_timeout,
    )
    if len(results) != len(items):
        raise RuntimeError(f"DocVQA evaluation incomplete: {len(results)}/{len(items)}")

    brief_path = out_root / "brief_result.jsonl"
    with brief_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "question": row.get("question"),
                        "ground_truth": row.get("gold_answer"),
                        "llm_answer": row.get("predicted_answer"),
                        "hard": row.get("hard"),
                        "soft": row.get("soft"),
                        "agent_ok": row.get("agent_ok"),
                        "fail_reason": row.get("fail_reason"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    count = len(results)
    hard = sum(int(row.get("hard", 0)) for row in results)
    avg_soft = sum(float(row.get("soft", 0.0)) for row in results) / count if count else 0.0
    agent_ok = sum(1 for row in results if row.get("agent_ok"))
    summary = {
        "count": count,
        "agent_ok": agent_ok,
        "hard_correct": hard,
        "hard_acc": hard / count if count else 0.0,
        "avg_soft": avg_soft,
        "model": config.provider.model,
        "reasoning_effort": reasoning,
        "skill_path": skill_path,
        "out_root": str(out_root),
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"brief_result={brief_path}", flush=True)
    print(
        f"count={count} agent_ok={agent_ok} hard_acc={hard}/{count}={summary['hard_acc']:.4f} "
        f"avg_soft={avg_soft:.4f}",
        flush=True,
    )
    artifacts = EvalArtifacts(out_root, summary_path, out_root / "results.jsonl", brief_path)
    artifacts.validate()
    return artifacts
