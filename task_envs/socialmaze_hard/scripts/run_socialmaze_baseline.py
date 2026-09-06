from __future__ import annotations

import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cobras.cobras_core.model import (
    chat_target,
    configure_azure_openai,
    get_target_deployment,
    is_target_exec_backend,
    set_reasoning_effort,
    set_target_backend,
    set_target_deployment,
)
from cobras.cobras_core.model.exec_harness import prepare_workspace, render_skill_md, run_target_exec
from cobras.task_envs.eval_errors import is_timeout_error, raise_evaluation_error


ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = ROOT
ROLES = ("Investigator", "Criminal", "Rumormonger", "Lunatic")


def load_env_files() -> None:
    for env_path in [
        PROJECT_ROOT / ".env.socialmaze.local",
        PROJECT_ROOT / ".env",
        ROOT / ".env.socialmaze.local",
        ROOT / ".env",
    ]:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("export "):
                s = s[len("export ") :].strip()
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def safe_name(value: object) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "item"


def clip(text: object, limit: int) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...[truncated]"


def load_skill(skill_path: str) -> tuple[str, str]:
    if not skill_path.strip():
        return "", ""
    path = Path(skill_path)
    if path.is_dir():
        path = path / "skill.md"
    if not path.exists():
        raise SystemExit(f"Skill file not found: {path}")
    return str(path), path.read_text(encoding="utf-8")


def parse_answer(text: str) -> dict[str, Any]:
    s = str(text or "")
    player_matches = list(
        re.finditer(
            r"(?:Final\s+)?Criminal\s*(?:Is|:)?\s*(?:Player\s*)?([0-9]+)",
            s,
            flags=re.IGNORECASE,
        )
    )
    if not player_matches:
        player_matches = list(
            re.finditer(
                r"(?:the\s+)?criminal\s*(?:player\s*)?(?:is|=|:)\s*(?:Player\s*)?([0-9]+)",
                s,
                flags=re.IGNORECASE,
            )
        )
    player_match = player_matches[-1] if player_matches else None

    role_matches = list(
        re.finditer(
            r"(?:My\s+Role|Player\s*1(?:'s)?\s+(?:true\s+)?role)\s*(?:Is|:|=)?\s*"
            r"(Investigator|Criminal|Rumormonger|Lunatic)",
            s,
            flags=re.IGNORECASE,
        )
    )
    if not role_matches:
        role_matches = list(
            re.finditer(
                r"\b(Investigator|Criminal|Rumormonger|Lunatic)\b",
                s,
                flags=re.IGNORECASE,
            )
        )
    role_match = role_matches[-1] if role_matches else None

    role = ""
    if role_match:
        role_raw = role_match.group(1).lower()
        role = next((r for r in ROLES if r.lower() == role_raw), "")
    return {
        "criminal_player": int(player_match.group(1)) if player_match else None,
        "role": role,
    }


def score_prediction(prediction: dict[str, Any], gold: dict[str, Any]) -> tuple[int, float, str]:
    criminal_ok = prediction.get("criminal_player") == gold.get("criminal_player")
    role_ok = prediction.get("role") == gold.get("role")
    soft = (int(criminal_ok) + int(role_ok)) / 2.0
    hard = int(criminal_ok and role_ok)
    reasons = []
    if not criminal_ok:
        reasons.append(f"criminal mismatch: pred={prediction.get('criminal_player')} gold={gold.get('criminal_player')}")
    if not role_ok:
        reasons.append(f"role mismatch: pred={prediction.get('role')!r} gold={gold.get('role')!r}")
    return hard, soft, "; ".join(reasons)


def build_prompt(row: dict[str, Any], skill_content: str) -> tuple[str, str]:
    system = str(row.get("system_prompt") or "").strip()
    if skill_content.strip():
        system += (
            "\n\n<skill>\n"
            "Use the following reusable reasoning guidance when relevant. It is not game evidence.\n\n"
            + skill_content.strip()
            + "\n</skill>"
        )
    user = str(row.get("prompt") or "").strip()
    user += (
        "\n\nAnswer with the final result in this exact final-line format:\n"
        "Final Criminal Is Player <number>. My Role Is <Investigator|Criminal|Rumormonger|Lunatic>."
    )
    return system, user


def _run_exec_target(
    *,
    pred_dir: Path,
    system: str,
    user: str,
    skill_content: str,
    timeout: int,
) -> tuple[str, str]:
    work_dir = pred_dir / "exec_backend"
    skill_md = render_skill_md(
        skill_content,
        description="Dynamic ReflACT skill for the current SocialMaze hidden-role question.",
        preamble=(
            "Use this skill as reusable reasoning guidance, not as game evidence. "
            "Return the required final criminal player and Player 1 role."
        ),
    )
    prepare_workspace(
        work_dir=str(work_dir),
        skill_md=skill_md,
        task_text=(
            "## System Contract\n"
            f"{system}\n\n"
            "## SocialMaze Instance\n"
            f"{user}"
        ),
    )
    return run_target_exec(
        work_dir=str(work_dir),
        prompt=(
            "Read task.md and .agents/skills/cobras-target/SKILL.md. "
            "Solve the hidden-role deduction task and end with exactly: "
            "Final Criminal Is Player <number>. My Role Is "
            "<Investigator|Criminal|Rumormonger|Lunatic>."
        ),
        model=get_target_deployment(),
        timeout=timeout,
        stage="socialmaze_rollout",
    )


def load_items(split: str, limit: int, seed: int | None, data_path: str = "") -> list[dict[str, Any]]:
    rows = []
    path = Path(data_path) if data_path.strip() else PROJECT_ROOT / "data" / "socialmaze_hard" / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"SocialMaze local split not found: {path}")
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        item.setdefault("id", f"{path.stem}_{idx}")
        rows.append(item)
        if len(rows) >= limit:
            break
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(rows)
    return rows


def run_one(
    row: dict[str, Any],
    *,
    out_root: Path,
    skill_content: str,
    max_completion_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    item_id = safe_name(row.get("id"))
    pred_dir = out_root / "predictions" / item_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    system, user = build_prompt(row, "" if is_target_exec_backend() else skill_content)
    gold = parse_answer(str(row.get("answer") or ""))
    started = time.time()
    response = ""
    usage: dict[str, Any] = {}
    ok = False
    error = ""
    try:
        if is_target_exec_backend():
            response, raw = _run_exec_target(
                pred_dir=pred_dir,
                system=system,
                user=user,
                skill_content=skill_content,
                timeout=timeout,
            )
            usage = {}
            (pred_dir / "exec_raw.txt").write_text(raw, encoding="utf-8")
        else:
            response, usage = chat_target(
                system=system,
                user=user,
                max_completion_tokens=max_completion_tokens,
                retries=3,
                stage="socialmaze_target",
                timeout=timeout,
            )
        ok = True
    except Exception as exc:  # noqa: BLE001
        if not is_timeout_error(exc):
            raise_evaluation_error("SocialMaze", item_id, exc)
        error = f"timeout: {type(exc).__name__}: {exc}"
        response = ""
    duration = time.time() - started
    prediction = parse_answer(response)
    hard, soft, fail_reason = score_prediction(prediction, gold) if ok else (0, 0.0, error)

    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": response},
    ]
    result = {
        "id": item_id,
        "task": row.get("task", ""),
        "split": item_id.split("_", 1)[0],
        "hard": hard,
        "soft": soft,
        "agent_ok": ok,
        "phase": "completed" if ok else "timeout",
        "gold_answer": row.get("answer", ""),
        "gold_parsed": gold,
        "predicted_answer": response,
        "predicted_parsed": prediction,
        "fail_reason": fail_reason,
        "duration": duration,
        "usage": usage,
        "n_turns": 1,
        "error": error,
    }
    reward = {
        "reward": hard,
        "hard": hard,
        "soft": soft,
        "criminal_correct": prediction.get("criminal_player") == gold.get("criminal_player"),
        "role_correct": prediction.get("role") == gold.get("role"),
        "gold": gold,
        "prediction": prediction,
        "fail_reason": fail_reason,
    }

    (pred_dir / "conversation.json").write_text(
        json.dumps(conversation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (pred_dir / "reward.json").write_text(
        json.dumps(reward, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (pred_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (pred_dir / "target_user_prompt.txt").write_text(user, encoding="utf-8")
    (pred_dir / "raw.txt").write_text(response, encoding="utf-8")
    return result


def write_outputs(out_root: Path, results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    results_path = out_root / "results.jsonl"
    brief_path = out_root / "brief_result.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with brief_path.open("w", encoding="utf-8") as f:
        for row in results:
            brief = {
                "id": row["id"],
                "hard": row["hard"],
                "soft": row["soft"],
                "agent_ok": row["agent_ok"],
                "ground_truth": row["gold_answer"],
                "llm_answer": clip(row["predicted_answer"], 600),
                "gold_parsed": row["gold_parsed"],
                "predicted_parsed": row["predicted_parsed"],
                "fail_reason": row["fail_reason"],
                "duration": row["duration"],
                "usage": row["usage"],
            }
            f.write(json.dumps(brief, ensure_ascii=False) + "\n")

    n = len(results)
    hard = sum(int(r["hard"]) for r in results)
    avg_soft = sum(float(r["soft"]) for r in results) / max(n, 1)
    criminal = sum(int(r["predicted_parsed"].get("criminal_player") == r["gold_parsed"].get("criminal_player")) for r in results)
    role = sum(int(r["predicted_parsed"].get("role") == r["gold_parsed"].get("role")) for r in results)
    agent_ok = sum(int(bool(r["agent_ok"])) for r in results)
    summary = {
        **config,
        "n": n,
        "hard": hard,
        "hard_acc": hard / max(n, 1),
        "avg_soft": avg_soft,
        "acc": avg_soft,
        "reward": avg_soft,
        "reward_field": "soft",
        "criminal_acc": criminal / max(n, 1),
        "role_acc": role / max(n, 1),
        "agent_ok": agent_ok,
        "out_root": str(out_root),
        "results_jsonl": str(results_path),
        "brief_result": str(brief_path),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"brief_result={brief_path}", flush=True)
    print(
        f"count={n} agent_ok={agent_ok} reward_soft={avg_soft:.4f} "
        f"hard_acc={hard}/{n}={hard / max(n, 1):.4f} "
        f"criminal_acc={summary['criminal_acc']:.4f} role_acc={summary['role_acc']:.4f}",
        flush=True,
    )


def main() -> None:
    load_env_files()
    provider = os.environ.get("PROVIDER", "openrouter").strip().lower()
    model = os.environ.get("MODEL", "openai/gpt-5.4-nano")
    if provider == "openrouter":
        base_url = os.environ.get("TARGET_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.environ.get("TARGET_API_KEY", "")
        target_model = os.environ.get("TARGET_MODEL", model)
    elif provider == "yunwu":
        base_url = os.environ.get("TARGET_BASE_URL", "https://yunwu.ai/v1")
        api_key = os.environ.get("TARGET_API_KEY", "")
        target_model = os.environ.get("TARGET_MODEL", model)
    else:
        raise SystemExit(f"Unsupported PROVIDER={provider}")
    if not api_key:
        raise SystemExit(f"Missing API key for provider={provider}")

    split = os.environ.get("SPLIT", "easy")
    data_path = os.environ.get("DATA_PATH", "").strip()
    limit = int(os.environ.get("LIMIT", os.environ.get("NUM_TASKS", "50")))
    workers = int(os.environ.get("WORKERS", "15"))
    timeout = int(os.environ.get("TASK_TIMEOUT", "240"))
    max_completion_tokens = int(os.environ.get("MAX_COMPLETION_TOKENS", "2048"))
    reasoning_effort = os.environ.get("COBRAS_TARGET_REASONING_EFFORT", "").strip() or None
    seed_raw = os.environ.get("SAMPLE_SEED", "").strip()
    seed = int(seed_raw) if seed_raw else None
    skill_path, skill_content = load_skill(os.environ.get("SKILL_PATH", ""))
    run_kind = "skill" if skill_content.strip() else "baseline"
    skill_name = safe_name(Path(skill_path).parent.name) if skill_path else "direct"
    reasoning_suffix = (reasoning_effort or "off").lower()
    default_run_id = (
        f"socialmaze_{provider}_{model.replace('/', '_')}_{run_kind}_{skill_name}_"
        f"{split}{limit}_parallel{workers}_reasoning{reasoning_suffix}"
    )
    run_id = os.environ.get("RUN_ID", default_run_id)
    out_root = Path(os.environ.get("OUT_ROOT", str(PROJECT_ROOT / "results/cobras/socialmaze_hard" / run_id)))
    out_root.mkdir(parents=True, exist_ok=True)

    requested_backend = os.environ.get("TARGET_BACKEND", "").strip().lower()
    if requested_backend in {"codex", "codex_exec", "claude_code_exec"}:
        backend = "codex_exec" if requested_backend == "codex" else requested_backend
        set_target_backend(backend)
        set_target_deployment(target_model)
    else:
        set_target_backend("openai_chat")
        set_target_deployment(target_model)
        set_reasoning_effort(reasoning_effort)
        configure_azure_openai(
            endpoint=base_url,
            api_key=api_key,
            auth_mode="openai_compatible",
            target_endpoint=base_url,
            target_api_key=api_key,
            target_auth_mode="openai_compatible",
        )

    print(
        "[socialmaze] "
        f"provider={provider} model={target_model} split={split} data_path={data_path or 'none'} "
        f"limit={limit} workers={workers} "
        f"reasoning={reasoning_effort or 'off'} skill={skill_path or 'none'} out={out_root}",
        flush=True,
    )
    items = load_items(split, limit, seed, data_path)
    print(f"[socialmaze] loaded_items={len(items)}", flush=True)

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_one,
                row,
                out_root=out_root,
                skill_content=skill_content,
                max_completion_tokens=max_completion_tokens,
                timeout=timeout,
            ): row
            for row in items
        }
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            done = len(results)
            hard = sum(int(r["hard"]) for r in results)
            print(
                f"[socialmaze] {done}/{len(items)} id={result['id']} hard={result['hard']} "
                f"soft={result['soft']:.2f} acc={hard / max(done, 1):.3f}",
                flush=True,
            )

    order = {safe_name(row.get("id")): idx for idx, row in enumerate(items)}
    results.sort(key=lambda r: order.get(r["id"], 10**9))
    config = {
        "dataset": "MBZUAI/SocialMaze",
        "split": split,
        "data_path": data_path,
        "limit": limit,
        "workers": workers,
        "provider": provider,
        "model": target_model,
        "reasoning_effort": reasoning_effort,
        "skill_path": skill_path,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "failures": failures,
    }
    write_outputs(out_root, results, config)


if __name__ == "__main__":
    main()
