from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any
from uuid import uuid4

from cobras.task_envs.alfworld.rollout import build_alfworld_env, run_alfworld_batch
from cobras.task_envs.eval_errors import (
    is_timeout_error,
    is_valid_result_row,
    raise_evaluation_error,
)
from cobras.task_envs.eval_runtime import configure_chat_target, load_skill, safe_name
from cobras.task_envs.eval_types import EvalArtifacts, EvalConfig


def _load_items(path: Path, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    rows = [dict(row) for row in payload if isinstance(row, dict)]
    for index, row in enumerate(rows):
        if not row.get("id") or not row.get("gamefile"):
            raise ValueError(f"ALFWorld item {index} is missing id or gamefile in {path}")
    return rows[:limit] if limit > 0 else rows


def _is_resumable_row(row: dict[str, Any]) -> bool:
    return is_valid_result_row(row)


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(row, dict)
            or not row.get("id")
            or not _is_resumable_row(row)
        ):
            continue
        artifacts = row.get("artifact_paths") or {}
        raw_conversation = str(artifacts.get("conversation_json") or "").strip()
        conversation = (
            path.parent / raw_conversation
            if raw_conversation
            else path.parent / "predictions" / safe_name(row["id"]) / "conversation.json"
        )
        if conversation.exists():
            rows[str(row["id"])] = row
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _accept_as_zero(row: dict[str, Any], reason: str) -> dict[str, Any]:
    accepted = dict(row)
    previous_reason = str(accepted.get("fail_reason") or "").strip()
    if reason and reason not in previous_reason:
        accepted["fail_reason"] = f"{previous_reason}; {reason}" if previous_reason else reason
    accepted.update(
        {
            "hard": 0,
            "soft": 0.0,
            "accepted_zero": True,
            "terminal_failure_reason": reason,
        }
    )
    return accepted


def _timeout_failure_zero(
    item: dict[str, Any],
    *,
    out_root: Path,
    reason: str,
) -> dict[str, Any]:
    item_id = str(item["id"])
    relative_conversation = Path("predictions") / safe_name(item_id) / "conversation.json"
    _write_json(
        out_root / relative_conversation,
        [
            {
                "type": "terminal_failure",
                "task_description": str(item.get("task_description") or ""),
                "reason": reason,
                "hard": 0,
                "soft": 0.0,
            }
        ],
    )
    return {
        "id": item_id,
        "hard": 0,
        "soft": 0.0,
        "n_turns": 0,
        "fail_reason": reason,
        "agent_ok": False,
        "accepted_zero": True,
        "terminal_failure_reason": reason,
        "task_timed_out": True,
        "task_type": str(item.get("task_type") or "other"),
        "gamefile": str(item.get("gamefile") or ""),
        "task_description": str(item.get("task_description") or ""),
        "instruction_type": str(item.get("task_type") or "other"),
        "artifact_paths": {"conversation_json": str(relative_conversation)},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid4().hex}.tmp"
    )
    try:
        temp_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _summary(rows: list[dict[str, Any]], *, expected: int) -> dict[str, Any]:
    hard = sum(int(row.get("hard") or 0) for row in rows)
    soft_sum = sum(float(row.get("soft") or 0.0) for row in rows)
    by_task = Counter(str(row.get("task_type") or "other") for row in rows)
    hard_by_task = Counter(
        str(row.get("task_type") or "other")
        for row in rows
        if int(row.get("hard") or 0)
    )
    return {
        "count": len(rows),
        "expected_count": expected,
        "hard": hard,
        "hard_acc": hard / max(len(rows), 1),
        "avg_soft": soft_sum / max(len(rows), 1),
        "task_counts": dict(sorted(by_task.items())),
        "task_hard": dict(sorted(hard_by_task.items())),
    }


def _persist(
    out_root: Path,
    rows_by_id: dict[str, dict[str, Any]],
    order: dict[str, int],
    *,
    expected: int,
) -> None:
    rows = sorted(rows_by_id.values(), key=lambda row: order.get(str(row.get("id")), 10**9))
    _write_jsonl(out_root / "results.jsonl", rows)
    _write_jsonl(
        out_root / "brief_result.jsonl",
        [
            {
                "id": row.get("id"),
                "hard": row.get("hard"),
                "soft": row.get("soft"),
                "n_turns": row.get("n_turns"),
                "task_type": row.get("task_type"),
                "fail_reason": row.get("fail_reason", ""),
            }
            for row in rows
        ],
    )
    _write_json(out_root / "summary.json", _summary(rows, expected=expected))


def _print_completion(
    item_id: str,
    result: dict[str, Any],
    rows_by_id: dict[str, dict[str, Any]],
    *,
    expected: int,
) -> None:
    current = len(rows_by_id)
    hard = sum(int(row.get("hard") or 0) for row in rows_by_id.values())
    reward = int(result.get("hard") or 0)
    agent_ok = str(bool(result.get("agent_ok", False))).lower()
    print(
        f"[alfworld] {current}/{expected} complete id={item_id} "
        f"agent_ok={agent_ok} reward={reward} turns={result.get('n_turns')} "
        f"hard_acc={hard / max(current, 1):.3f}",
        flush=True,
    )


def _split_kind(items: list[dict[str, Any]]) -> tuple[str, bool]:
    gamefile = str(items[0].get("gamefile") or "") if items else ""
    if "/valid_seen/" in f"/{gamefile}":
        return "eval_in_distribution", False
    if "/valid_unseen/" in f"/{gamefile}":
        return "eval_out_of_distribution", False
    return "train", True


def _run_item(
    item: dict[str, Any],
    *,
    item_seed: int,
    eval_dataset: str,
    is_train: bool,
    skill_content: str,
    max_steps: int,
    out_root: Path,
    max_completion_tokens: int,
    request_timeout: int,
    task_timeout: int,
) -> dict[str, Any]:
    """Run one complete episode so the executor can refill its worker slot."""
    for module_name in ("omegaconf", "alfworld"):
        try:
            importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                f"ALFWorld runtime dependency {module_name!r} is unavailable: {exc}. "
                "Install the ALFWorld extras with: python -m pip install -e '.[alfworld]'"
            ) from exc
    item_id = str(item["id"])
    env_manager = build_alfworld_env(
        env_num=1,
        eval_dataset=eval_dataset,
        seed=item_seed,
        is_train=is_train,
        specific_gamefiles=[str(item["gamefile"])],
    )
    try:
        results = run_alfworld_batch(
            env_manager,
            skill_content=skill_content,
            max_steps=max_steps,
            out_root=str(out_root),
            max_api_workers=1,
            max_completion_tokens=max_completion_tokens,
            request_timeout=request_timeout,
            task_timeout=task_timeout,
            result_ids=[item_id],
        )
    finally:
        env_manager.close()
    if len(results) != 1:
        raise RuntimeError(f"Expected one ALFWorld result for {item_id}, got {len(results)}")
    result = results[0]
    result["gamefile"] = str(item["gamefile"])
    result["task_type"] = str(item.get("task_type") or result.get("task_type") or "other")
    return result


def run_eval(config: EvalConfig) -> EvalArtifacts:
    os.environ.setdefault("OPENAI_TARGET_TEMPERATURE", "0.0")
    _, reasoning, max_tokens, task_timeout = configure_chat_target(
        config,
        default_reasoning_effort="",
        default_max_completion_tokens=16384,
        default_timeout=600,
    )
    request_timeout = max(1, int(os.environ.get("ALFWORLD_REQUEST_TIMEOUT", "300")))
    trajectory_retries = max(0, int(os.environ.get("ALFWORLD_TRAJECTORY_RETRIES", "1")))
    exec_session_mode = os.environ.get(
        "ALFWORLD_EXEC_SESSION_MODE",
        os.environ.get("ALFWORLD_CODEX_SESSION_MODE", "action"),
    )
    codex_session_mode = exec_session_mode
    skill_path, skill_content = load_skill(config.skill_path)
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    configured_path = str(config.options.get("data_path", "")).strip()
    items_path = (
        Path(configured_path).expanduser().resolve()
        if configured_path
        else config.dataset.data_root / config.split / "items.json"
    )
    items = _load_items(items_path, config.limit)
    runtime_root = config.dataset.data_root / "runtime"
    os.environ["ALFWORLD_DATA"] = str(runtime_root.resolve())
    missing = [str(runtime_root / str(row["gamefile"])) for row in items if not (runtime_root / str(row["gamefile"])).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} ALFWorld game files; first: {missing[0]}")

    results_path = out_root / "results.jsonl"
    order = {str(row["id"]): index for index, row in enumerate(items)}
    rows_by_id = {
        item_id: row
        for item_id, row in _load_existing(results_path).items()
        if item_id in order
    }
    pending = [row for row in items if str(row["id"]) not in rows_by_id]
    eval_dataset, is_train = _split_kind(items)
    workers = max(1, min(config.workers, len(items)))
    if config.max_turns <= 0:
        raise ValueError("ALFWorld evaluation requires max_turns > 0")
    max_steps = config.max_turns
    # Environment processes are launched from executor threads. Spawn avoids
    # forking a multithreaded process while model requests are in flight.
    os.environ.setdefault("ALFWORLD_WORKER_START_METHOD", "spawn")
    print(
        f"[alfworld] provider={config.provider.provider} model={config.provider.model} "
        f"split={config.split} items={len(items)} done={len(items) - len(pending)} "
        f"pending={len(pending)} workers={workers} scheduler=episode_queue max_steps={max_steps} "
        f"task_timeout={task_timeout}s trajectory_retries={trajectory_retries} "
        f"codex_session_mode={codex_session_mode} "
        f"reasoning={reasoning or 'off'} skill={skill_path or 'none'} out={out_root}",
        flush=True,
    )

    items_by_id = {str(item["id"]): item for item in items}
    attempt_items = pending
    for attempt_index in range(trajectory_retries + 1):
        if not attempt_items:
            break
        if attempt_index:
            print(
                f"[alfworld] retry_pass={attempt_index}/{trajectory_retries} "
                f"trajectories={len(attempt_items)}",
                flush=True,
            )

        invalid_ids: set[str] = set()
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="alfworld-episode")
        try:
            future_to_item = {
                executor.submit(
                    _run_item,
                    item,
                    item_seed=config.seed + order[str(item["id"])],
                    eval_dataset=eval_dataset,
                    is_train=is_train,
                    skill_content=skill_content,
                    max_steps=max_steps,
                    out_root=out_root,
                    max_completion_tokens=max_tokens,
                    request_timeout=request_timeout,
                    task_timeout=task_timeout,
                ): item
                for item in attempt_items
            }
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                item_id = str(item["id"])
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    if not is_timeout_error(exc):
                        for pending_future in future_to_item:
                            pending_future.cancel()
                        raise_evaluation_error("ALFWorld", item_id, exc)
                    reason = (
                        f"accepted after {task_timeout}s task timeout: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    result = _timeout_failure_zero(
                        item,
                        out_root=out_root,
                        reason=reason,
                    )
                if not result.get("agent_ok", True):
                    if result.get("task_timed_out", False):
                        if not result.get("accepted_zero", False):
                            result = _accept_as_zero(
                                result,
                                f"accepted after {task_timeout}s task timeout",
                            )
                        print(f"[alfworld] id={item_id} task_timeout accepted_zero", flush=True)
                    elif attempt_index < trajectory_retries:
                        invalid_ids.add(item_id)
                        print(
                            f"[alfworld] id={item_id} invalid trajectory; queued for retry",
                            flush=True,
                        )
                        # This attempt is not the task's final result. Do not
                        # persist it or emit a misleading "complete" line.
                        continue
                    else:
                        result = _accept_as_zero(
                            result,
                            f"accepted after {trajectory_retries} trajectory retries",
                        )
                        print(f"[alfworld] id={item_id} retries_exhausted accepted_zero", flush=True)

                rows_by_id[str(result["id"])] = result
                _persist(out_root, rows_by_id, order, expected=len(items))
                _print_completion(item_id, result, rows_by_id, expected=len(items))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        retry_ids = invalid_ids
        if retry_ids and attempt_index < trajectory_retries:
            for item_id in retry_ids:
                rows_by_id.pop(item_id, None)
            _persist(out_root, rows_by_id, order, expected=len(items))
            attempt_items = [items_by_id[item_id] for item_id in order if item_id in retry_ids]
            continue
        break

    _persist(out_root, rows_by_id, order, expected=len(items))
    _write_json(
        out_root / "config.json",
        {
            "dataset": "ALFWorld",
            "runtime_python": sys.executable,
            "split": config.split,
            "items_path": str(items_path),
            "runtime_root": str(runtime_root.resolve()),
            "limit": config.limit,
            "workers": workers,
            "provider": config.provider.provider,
            "model": config.provider.model,
            "reasoning_effort": reasoning,
            "temperature": float(os.environ.get("OPENAI_TARGET_TEMPERATURE", "0.0")),
            "max_steps": max_steps,
            "scheduler": "episode_queue",
            "max_completion_tokens": max_tokens,
            "request_timeout": request_timeout,
            "task_timeout": task_timeout,
            "trajectory_retries": trajectory_retries,
            "codex_session_mode": codex_session_mode,
            "skill_path": skill_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    if len(rows_by_id) != len(items):
        raise RuntimeError(f"ALFWorld evaluation incomplete: {len(rows_by_id)}/{len(items)}")
    artifacts = EvalArtifacts(
        out_root=out_root,
        summary_path=out_root / "summary.json",
        results_path=results_path,
        brief_result_path=out_root / "brief_result.jsonl",
    )
    artifacts.validate()
    return artifacts
