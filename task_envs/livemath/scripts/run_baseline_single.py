#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cobras.cobras_core.model import (
    chat_target,
    configure_azure_openai,
    get_target_deployment,
    is_target_exec_backend,
    set_target_backend,
    set_target_deployment,
)
from cobras.cobras_core.model.exec_harness import prepare_workspace, render_skill_md, run_target_exec
from cobras.scripts.env_utils import load_env_files, resolve_chat_provider
from cobras.task_envs.eval_errors import is_timeout_error, raise_evaluation_error
from cobras.task_envs.livemath.common import ensure_clean_dir, load_normalized_split_items, write_json
from cobras.task_envs.livemath.reward import evaluate


LOADED_ENV_FILES = load_env_files(
    [
        _PROJECT_ROOT / ".env.livemath.local",
        _PROJECT_ROOT / ".env",
    ]
)

DEFAULT_MODEL = os.environ.get("LIVEMATH_MODEL", "gpt-5.5")
DEFAULT_BASE_URL = os.environ.get("LIVEMATH_BASE_URL") or os.environ.get(
    "OPENAI_BASE_URL", "https://yunwu.ai/v1"
)
DEFAULT_PROVIDER = os.environ.get("LIVEMATH_PROVIDER", "openrouter").strip().lower() or "openrouter"
DEFAULT_SPLIT_ROOT = Path(
    os.environ.get("LIVEMATH_SPLIT_ROOT", str(_PROJECT_ROOT / "data" / "livemath"))
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct-answer LiveMath baseline or skill evaluation.")
    parser.add_argument("--provider", choices=["openrouter", "yunwu", "custom", "local"], default=DEFAULT_PROVIDER)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--limit", type=int, default=0, help="0 means use the full split.")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="")
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--use-theorem", action="store_true")
    parser.add_argument("--use-sketch", action="store_true")
    parser.add_argument("--prompt-version", choices=["direct", "analysis"], default="direct")
    parser.add_argument("--skill-path", type=Path, default=Path(""))
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="TARGET_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--openai-base-url-env", default="OPENAI_BASE_URL")
    parser.add_argument("--out-root", type=Path, default=_PROJECT_ROOT / "output" / "livemath")
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def _set_target_env(base_url: str, api_key_env: str, api_key: str, openai_base_url_env: str) -> None:
    if base_url:
        os.environ[openai_base_url_env] = base_url
    if api_key:
        os.environ[api_key_env] = api_key


def _configure_target(args: argparse.Namespace) -> str:
    requested_backend = os.environ.get("TARGET_BACKEND", "").strip().lower()
    if requested_backend in {"codex", "codex_exec", "claude_code_exec"}:
        backend = "codex_exec" if requested_backend == "codex" else requested_backend
        set_target_backend(backend)
        set_target_deployment(args.model)
        return backend

    set_target_backend("openai_chat")
    set_target_deployment(args.model)
    configure_azure_openai(
        endpoint=args.base_url,
        api_key=args.api_key,
        auth_mode="openai_compatible",
        target_endpoint=args.base_url,
        target_api_key=args.api_key,
        target_auth_mode="openai_compatible",
    )
    return "openai_chat"


def _resolve_provider(args: argparse.Namespace) -> None:
    if args.provider == "openrouter":
        args.model = (
            args.model
            or os.environ.get("OPENROUTER_LIVEMATH_MODEL", "")
            or os.environ.get("LIVEMATH_MODEL", "")
            or "openai/gpt-5.4-nano"
        )
        args.base_url = (
            args.base_url
            or os.environ.get("TARGET_BASE_URL", "")
            or "https://openrouter.ai/api/v1"
        )
        args.api_key = (
            args.api_key
            or os.environ.get("TARGET_API_KEY", "")

        )
        return

    provider_cfg = resolve_chat_provider(args.provider)
    args.provider = provider_cfg["provider"]
    args.model = args.model or provider_cfg["model"] or DEFAULT_MODEL
    args.base_url = args.base_url or provider_cfg["base_url"] or DEFAULT_BASE_URL
    args.api_key = args.api_key or provider_cfg["api_key"]


def _prompt_rule_lines(prompt_version: str) -> list[str]:
    if prompt_version == "analysis":
        return [
            "- Analyze the question carefully before giving the final answer.",
            "- End your response with exactly one final choice label inside <answer>...</answer>.",
            "- The final answer tag must contain only the label, such as <answer>A</answer>.",
            "- Compare all choices carefully before deciding.",
        ]
    return [
        "- Output only the final choice label inside <answer>...</answer>.",
        "- Do not add explanations.",
        "- Compare all choices carefully before deciding.",
    ]


def _build_system(*, prompt_version: str, skill_content: str) -> str:
    analysis_hint = (
        "Think through the question carefully before answering."
        if prompt_version == "analysis"
        else "Work carefully before answering."
    )
    parts = [
        "You are a careful LiveMath solver.",
        analysis_hint,
    ]
    if skill_content.strip():
        parts.extend(["", "## Skill", skill_content.strip()])
    parts.extend(["", "Always return the final choice label inside <answer>...</answer>."])
    return "\n".join(parts)


def _build_user(
    item: dict[str, Any],
    *,
    use_theorem: bool,
    use_sketch: bool,
    prompt_version: str,
) -> str:
    lines = [
        "You are answering one LiveMathematicianBench multiple-choice question.",
        "",
        "Rules:",
        *_prompt_rule_lines(prompt_version),
        "",
    ]
    if use_theorem and item.get("theorem"):
        lines.extend([f"## Theorem\n{item['theorem']}", ""])
    if use_sketch and item.get("sketch"):
        lines.extend([f"## Proof Sketch\n{item['sketch']}", ""])
    lines.extend([f"## Question\n{item['question']}", "", "## Choices"])
    for choice in item["choices"]:
        lines.append(f"{choice['label']}. {choice['text']}")
    return "\n".join(lines)


def _build_exec_skill(skill_content: str) -> str:
    return render_skill_md(
        skill_content,
        description="Dynamic ReflACT skill for the current LiveMath multiple-choice question.",
        preamble=(
            "Use this skill when solving the current LiveMath question. "
            "Return only the required choice label inside <answer>...</answer>."
        ),
    )


def _run_exec_trial(
    *,
    workspace: Path,
    system: str,
    user: str,
    skill_content: str,
) -> tuple[str, str]:
    work_dir = workspace / "exec_backend"
    task_text = (
        "## System Contract\n"
        f"{system}\n\n"
        "## Question\n"
        f"{user}"
    )
    prepare_workspace(
        work_dir=str(work_dir),
        skill_md=_build_exec_skill(skill_content),
        task_text=task_text,
    )
    return run_target_exec(
        work_dir=str(work_dir),
        prompt=(
            "Read task.md and .agents/skills/cobras-target/SKILL.md. "
            "Solve the multiple-choice question and return only the final label "
            "inside <answer>...</answer>."
        ),
        model=get_target_deployment(),
        timeout=max(1, int(os.environ.get("LIVEMATH_EXEC_TIMEOUT", "300"))),
        stage="livemath_rollout",
    )


def _artifact_paths(trial_root: Path) -> dict[str, str]:
    workspace = trial_root / "workspace"
    return {
        "trial_root": str(trial_root),
        "workspace_root": str(workspace),
        "conversation_json": str(workspace / "conversation.json"),
        "target_system_prompt": str(workspace / "target_system_prompt.txt"),
        "target_user_prompt": str(workspace / "target_user_prompt.txt"),
    }


def _run_trial(
    *,
    item: dict[str, Any],
    trial_index: int,
    run_root: Path,
    max_completion_tokens: int,
    prompt_version: str,
    skill_content: str,
    use_theorem: bool,
    use_sketch: bool,
) -> dict[str, Any]:
    trial_root = run_root / "tasks" / item["id"] / f"trial_{trial_index:02d}"
    workspace = trial_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    system = _build_system(prompt_version=prompt_version, skill_content=skill_content)
    user = _build_user(
        item,
        use_theorem=use_theorem,
        use_sketch=use_sketch,
        prompt_version=prompt_version,
    )
    (workspace / "target_system_prompt.txt").write_text(system, encoding="utf-8")
    (workspace / "target_user_prompt.txt").write_text(user, encoding="utf-8")

    if is_target_exec_backend():
        response, raw = _run_exec_trial(
            workspace=workspace,
            system=system,
            user=user,
            skill_content=skill_content,
        )
        _usage: dict[str, Any] = {}
        (workspace / "exec_raw.txt").write_text(raw, encoding="utf-8")
    else:
        response, _usage = chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=5,
            stage="rollout",
        )
    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": response},
    ]
    write_json(workspace / "conversation.json", conversation)

    eval_result = evaluate(response, item["correct_choice"], item["choices"])
    hard = int(eval_result["em"])
    payload = {
        "id": item["id"],
        "question": item["question"],
        "choices": item["choices"],
        "task_type": item.get("theorem_type", ["math_mcq"])[0]
        if item.get("theorem_type")
        else "math_mcq",
        "hard": hard,
        "soft": float(eval_result["f1"]),
        "predicted_answer": eval_result["predicted_answer"],
        "predicted_label": eval_result["predicted_label"],
        "predicted_text": eval_result["predicted_text"],
        "correct_label": eval_result["correct_label"],
        "correct_text": eval_result["correct_text"],
        "response": response,
        "fail_reason": ""
        if hard
        else (
            f"predicted '{eval_result['predicted_label'] or eval_result['predicted_answer']}' "
            f"but expected '{eval_result['correct_label']}'"
        ),
        "agent_ok": True,
        "n_turns": 1,
        "trial_index": trial_index,
        "hard_reward": float(hard),
        "soft_reward": float(eval_result["f1"]),
        "artifact_paths": _artifact_paths(trial_root),
        "skill_used": bool(skill_content.strip()),
        "use_theorem": use_theorem,
        "use_sketch": use_sketch,
        "prompt_version": prompt_version,
    }
    write_json(trial_root / "result.json", payload)
    return payload


def _timeout_error_result(
    *,
    item: dict[str, Any],
    trial_index: int,
    run_root: Path,
    error: Exception,
    skill_used: bool,
    use_theorem: bool,
    use_sketch: bool,
    prompt_version: str,
) -> dict[str, Any]:
    trial_root = run_root / "tasks" / item["id"] / f"trial_{trial_index:02d}"
    trial_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": item["id"],
        "question": item["question"],
        "choices": item["choices"],
        "task_type": item.get("theorem_type", ["math_mcq"])[0]
        if item.get("theorem_type")
        else "math_mcq",
        "hard": 0,
        "soft": 0.0,
        "predicted_answer": "",
        "predicted_label": "",
        "predicted_text": "",
        "correct_label": item["correct_choice"]["label"],
        "correct_text": item["correct_choice"]["text"],
        "response": "",
        "fail_reason": f"timeout: {type(error).__name__}: {error}",
        "agent_ok": False,
        "phase": "timeout",
        "n_turns": 0,
        "trial_index": trial_index,
        "hard_reward": 0.0,
        "soft_reward": 0.0,
        "artifact_paths": _artifact_paths(trial_root),
        "skill_used": skill_used,
        "use_theorem": use_theorem,
        "use_sketch": use_sketch,
        "prompt_version": prompt_version,
    }
    write_json(trial_root / "result.json", payload)
    return payload


def run(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    _resolve_provider(args)
    _set_target_env(args.base_url, args.api_key_env, args.api_key, args.openai_base_url_env)
    target_backend = _configure_target(args)

    items = load_normalized_split_items(args.split_root.resolve(), args.split, limit=args.limit)
    has_skill_path = str(args.skill_path) not in {"", "."}
    if has_skill_path and not args.skill_path.exists():
        raise FileNotFoundError(f"Skill file not found: {args.skill_path}")
    skill_content = args.skill_path.read_text(encoding="utf-8") if has_skill_path else ""

    prompt_suffix = "" if args.prompt_version == "direct" else f"_prompt_{args.prompt_version}"
    default_run_name = (
        f"limit{args.limit}_trials{args.trials}_model_{args.model.replace('/', '_')}_"
        f"{args.split}{prompt_suffix}_direct"
    )
    run_name = args.run_name.strip() or default_run_name
    run_root = args.out_root / run_name
    ensure_clean_dir(run_root)
    write_json(
        run_root / "config.json",
        {
            "split_root": str(args.split_root),
            "split": args.split,
            "provider": args.provider,
            "target_backend": target_backend,
            "limit": args.limit,
            "trials": args.trials,
            "parallel": args.parallel,
            "seed": args.seed,
            "model": args.model,
            "loaded_env_files": [str(path) for path in LOADED_ENV_FILES],
            "mode": "single",
            "eval_protocol": {"harness": "direct", "prompt_version": args.prompt_version},
            "max_completion_tokens": args.max_completion_tokens,
            "use_theorem": args.use_theorem,
            "use_sketch": args.use_sketch,
            "prompt_version": args.prompt_version,
            "skill_path": str(args.skill_path) if has_skill_path else "",
            "base_url": args.base_url,
            "api_key_env": args.api_key_env,
            "run_name": run_name,
        },
    )

    tasks = [
        (item, trial_index)
        for item in items
        for trial_index in range(1, args.trials + 1)
    ]
    print(
        f"[livemath direct] split={args.split} items={len(items)} trials={args.trials} "
        f"backend={target_backend}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_map = {
            executor.submit(
                _run_trial,
                item=item,
                trial_index=trial_index,
                run_root=run_root,
                max_completion_tokens=args.max_completion_tokens,
                prompt_version=args.prompt_version,
                skill_content=skill_content,
                use_theorem=args.use_theorem,
                use_sketch=args.use_sketch,
            ): (item, trial_index)
            for item, trial_index in tasks
        }
        for future in as_completed(future_map):
            item, trial_index = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                if not is_timeout_error(exc):
                    for pending_future in future_map:
                        pending_future.cancel()
                    raise_evaluation_error("LiveMath", item["id"], exc)
                result = _timeout_error_result(
                    item=item,
                    trial_index=trial_index,
                    run_root=run_root,
                    error=exc,
                    skill_used=bool(skill_content.strip()),
                    use_theorem=args.use_theorem,
                    use_sketch=args.use_sketch,
                    prompt_version=args.prompt_version,
                )
            results.append(result)
            print(
                f"[livemath direct] {item['id']} trial={trial_index} "
                f"hard={result['hard_reward']:.3f} soft={result['soft_reward']:.3f}",
                flush=True,
            )

    results.sort(key=lambda row: (str(row.get("id", "")), int(row.get("trial_index", 0))))
    results_path = run_root / "trial_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        run_root / "summary.json",
        {
            "count": len(results),
            "task_count": len(items),
            "trials_per_task": args.trials,
            "avg_hard_reward": sum(float(row["hard_reward"]) for row in results)
            / max(len(results), 1),
            "avg_soft_reward": sum(float(row["soft_reward"]) for row in results)
            / max(len(results), 1),
            "success_trials": sum(1 for row in results if float(row["hard_reward"]) >= 1.0),
            "results_path": str(results_path),
            "results": results,
        },
    )
    print(str(run_root), flush=True)
    return run_root


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
