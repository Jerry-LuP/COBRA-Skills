from __future__ import annotations

import json
import os
import random
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cobras.cobras_core.model import (
    chat_target,
    configure_azure_openai,
    set_reasoning_effort,
    set_target_backend,
    set_target_deployment,
)


ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = ROOT
SOCIALMAZE_ROOT = ROOT / "external/SocialMaze"
AGE_GROUPS = ("18-34", "35-54", "55+")
GENDERS = ("Male", "Female", "Non-binary")


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


def with_skill(system: str, skill_content: str) -> str:
    if not skill_content.strip():
        return system
    return (
        system.rstrip()
        + "\n\n<skill>\n"
        + "Use the following reusable reasoning guidance when relevant. It is not task evidence.\n\n"
        + skill_content.strip()
        + "\n</skill>"
    )


def parse_spy_prediction(response: str) -> str:
    for pattern in [
        r"Final\s+Spy\s*:\s*Player\s*([1-4])",
        r"Finial\s+Spy\s*:\s*Player\s*([1-4])",
    ]:
        match = re.search(pattern, response, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    match = re.search(r"(?:^|\D)([1-4])(?:\D|$)", response)
    return match.group(1) if match else ""


def parse_demographics(response: str) -> dict[str, str]:
    s = str(response or "")
    lower = s.lower()
    age = ""
    gender = ""

    age_match = re.search(r"age group\s*:\s*(18-34|35-54|55\+)", lower)
    if age_match:
        age = age_match.group(1)
    elif re.search(r"\b18-34\b|\byoung\b|\byouth\b", lower):
        age = "18-34"
    elif re.search(r"\b35-54\b|\bmiddle\b|\bmiddle-aged\b", lower):
        age = "35-54"
    elif re.search(r"\b55\+\b|\belderly\b|\bolder\b|\bsenior\b", lower):
        age = "55+"

    gender_match = re.search(r"gender\s*:\s*(male|female|non-binary)", lower)
    if gender_match:
        raw = gender_match.group(1)
        gender = next((g for g in GENDERS if g.lower() == raw), "")
    elif re.search(r"\bnon-binary\b|\bnonbinary\b|\bnon binary\b", lower):
        gender = "Non-binary"
    elif re.search(r"\bfemale\b|\bwoman\b|\bwomen\b", lower):
        gender = "Female"
    elif re.search(r"\b(?<!fe)male\b", lower):
        gender = "Male"

    return {"age_group": age, "gender": gender}


def build_find_the_spy(row: dict[str, Any], skill_content: str) -> tuple[str, str, dict[str, Any]]:
    system = (
        "You are a skilled player in a word description game. "
        "Your task is to identify which player is the 'spy' based on their descriptions.\n\n"
        "Game Rules:\n"
        "1. There are 4 players in the game.\n"
        "2. 3 players received the same word (normal players).\n"
        "3. 1 player received a different but related word (the spy).\n"
        "4. Each player describes their word without saying it directly.\n"
        "5. You need to determine who is the spy based on these descriptions.\n\n"
        f"You are Player 1, and your word is: \"{row['player_words']['1']}\".\n\n"
        "Analyze the descriptions carefully. Look for subtle differences that might reveal who has a different word."
    )
    user = ""
    statements = row.get("statements", [])
    for round_idx, round_data in enumerate(statements):
        user += f"Round {round_idx + 1}:\n"
        for statement in round_data.get("statements", []):
            user += f"{statement['statement']}\n"
        if round_idx < len(statements) - 1:
            user += "\n"
    user += (
        "\nBased on these descriptions, which player is the spy (Player 1, 2, 3, or 4)?\n"
        "Explain your reasoning step by step, then provide your final answer in the format: "
        "'Final Spy: Player X'"
    )
    gold = {"spy_player": str(row["spy_player"])}
    return with_skill(system, skill_content), user, gold


def build_upi_entity(row: dict[str, Any], skill_content: str) -> tuple[str, str, dict[str, Any]]:
    system = (
        "Your job is to analyze multiple reviews for a product and determine the most likely demographics "
        "of the primary user group who wrote these reviews.\n\n"
        "Focus on identifying the MAJORITY demographic group based on text patterns and content of the reviews."
    )
    user = "Reviews:\n"
    for comment_data in row["comments"]:
        user += f"\n\"{comment_data['comment']}\"\n"
    user += (
        "\nAnalyze these reviews carefully. What is the primary demographic group writing these reviews?\n\n"
        "First explain your reasoning, then provide your final demographic classification in exactly this format:\n"
        "Age Group: [18-34 OR 35-54 OR 55+]\n"
        "Gender: [Male OR Female OR Non-binary]\n"
    )
    gold = {
        "age_group": row["primary_user_group"]["primary_age_group"],
        "gender": row["primary_user_group"]["primary_gender"],
    }
    return with_skill(system, skill_content), user, gold


def build_upi_persona(row: dict[str, Any], skill_content: str) -> tuple[str, str, dict[str, Any]]:
    system = (
        "Your job is to analyze multiple reviews written by the same person and determine their likely age group "
        "and gender based on their writing style, interests, and perspectives.\n\n"
        "Focus on identifying the most likely demographic profile from the text patterns, interests, and perspectives "
        "in the comments."
    )
    user = "Reviews:\n"
    for comment_data in row["comments"]:
        user += f"\nOn {comment_data['product']}: \"{comment_data['comment']}\"\n"
    user += (
        "\nAnalyze these reviews carefully. What are the likely demographic characteristics of this user?\n\n"
        "First explain your reasoning, then provide your final demographic classification in exactly this format:\n"
        "Age Group: [18-34 OR 35-54 OR 55+]\n"
        "Gender: [Male OR Female OR Non-binary]\n"
    )
    gold = {
        "age_group": row["demographics"]["age_group"],
        "gender": row["demographics"]["gender"],
    }
    return with_skill(system, skill_content), user, gold


def load_items(task: str, limit: int, seed: int | None) -> list[dict[str, Any]]:
    if task == "find_the_spy":
        path = SOCIALMAZE_ROOT / "find_the_spy/data/fts_dataset_eval.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            row["id"] = row["scenario_id"]
    elif task == "upi_entity":
        path = SOCIALMAZE_ROOT / "user_profile_inference/data/user_entity.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for key, row in data["scenarios"].items():
            row["id"] = f"entity_{key}"
            rows.append(row)
    elif task == "upi_persona":
        path = SOCIALMAZE_ROOT / "user_profile_inference/data/user_persona.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for row in data["profile_groups"]:
            row["id"] = f"persona_{row['group_id']}"
            rows.append(row)
    else:
        raise SystemExit(f"Unsupported TASK={task}")

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(rows)
    return rows[:limit]


def build_prompt(task: str, row: dict[str, Any], skill_content: str) -> tuple[str, str, dict[str, Any]]:
    if task == "find_the_spy":
        return build_find_the_spy(row, skill_content)
    if task == "upi_entity":
        return build_upi_entity(row, skill_content)
    if task == "upi_persona":
        return build_upi_persona(row, skill_content)
    raise SystemExit(f"Unsupported TASK={task}")


def score(task: str, response: str, gold: dict[str, Any]) -> tuple[dict[str, Any], int, float, str]:
    if task == "find_the_spy":
        pred = {"spy_player": parse_spy_prediction(response)}
        hard = int(pred["spy_player"] == gold["spy_player"])
        reason = "" if hard else f"spy mismatch: pred={pred['spy_player']!r} gold={gold['spy_player']!r}"
        return pred, hard, float(hard), reason

    pred = parse_demographics(response)
    age_ok = pred.get("age_group") == gold.get("age_group")
    gender_ok = pred.get("gender") == gold.get("gender")
    soft = (int(age_ok) + int(gender_ok)) / 2.0
    hard = int(age_ok and gender_ok)
    reasons = []
    if not age_ok:
        reasons.append(f"age mismatch: pred={pred.get('age_group')!r} gold={gold.get('age_group')!r}")
    if not gender_ok:
        reasons.append(f"gender mismatch: pred={pred.get('gender')!r} gold={gold.get('gender')!r}")
    return pred, hard, soft, "; ".join(reasons)


def run_one(
    row: dict[str, Any],
    *,
    task: str,
    out_root: Path,
    skill_content: str,
    max_completion_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    item_id = safe_name(row.get("id"))
    pred_dir = out_root / "predictions" / item_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    system, user, gold = build_prompt(task, row, skill_content)
    started = time.time()
    response = ""
    usage: dict[str, Any] = {}
    ok = False
    error = ""
    try:
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
        error = f"{type(exc).__name__}: {exc}"
    duration = time.time() - started
    prediction, hard, soft, fail_reason = score(task, response, gold) if ok else ({}, 0, 0.0, error)

    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": response},
    ]
    result = {
        "id": item_id,
        "task": task,
        "hard": hard,
        "soft": soft,
        "agent_ok": ok,
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
        "gold": gold,
        "prediction": prediction,
        "fail_reason": fail_reason,
    }
    (pred_dir / "conversation.json").write_text(json.dumps(conversation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (pred_dir / "reward.json").write_text(json.dumps(reward, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (pred_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
                "gold_parsed": row["gold_parsed"],
                "predicted_parsed": row["predicted_parsed"],
                "llm_answer": clip(row["predicted_answer"], 600),
                "fail_reason": row["fail_reason"],
                "duration": row["duration"],
                "usage": row["usage"],
            }
            f.write(json.dumps(brief, ensure_ascii=False) + "\n")

    n = len(results)
    hard = sum(int(r["hard"]) for r in results)
    avg_soft = sum(float(r["soft"]) for r in results) / max(n, 1)
    agent_ok = sum(int(bool(r["agent_ok"])) for r in results)
    summary = {
        **config,
        "n": n,
        "hard": hard,
        "hard_acc": hard / max(n, 1),
        "avg_soft": avg_soft,
        "agent_ok": agent_ok,
        "out_root": str(out_root),
        "results_jsonl": str(results_path),
        "brief_result": str(brief_path),
    }
    task = config["task"]
    if task == "find_the_spy":
        spy_correct = sum(
            int(r["predicted_parsed"].get("spy_player") == r["gold_parsed"].get("spy_player"))
            for r in results
        )
        summary["spy_acc"] = spy_correct / max(n, 1)
    else:
        age_correct = sum(
            int(r["predicted_parsed"].get("age_group") == r["gold_parsed"].get("age_group"))
            for r in results
        )
        gender_correct = sum(
            int(r["predicted_parsed"].get("gender") == r["gold_parsed"].get("gender"))
            for r in results
        )
        summary["age_acc"] = age_correct / max(n, 1)
        summary["gender_acc"] = gender_correct / max(n, 1)

    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"brief_result={brief_path}", flush=True)
    if task == "find_the_spy":
        print(f"count={n} agent_ok={agent_ok} hard_acc={hard}/{n}={hard / max(n, 1):.4f} spy_acc={summary['spy_acc']:.4f}", flush=True)
    else:
        print(
            f"count={n} agent_ok={agent_ok} hard_acc={hard}/{n}={hard / max(n, 1):.4f} "
            f"avg_soft={avg_soft:.4f} age_acc={summary['age_acc']:.4f} gender_acc={summary['gender_acc']:.4f}",
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

    task = os.environ.get("TASK", "find_the_spy").strip()
    limit = int(os.environ.get("LIMIT", os.environ.get("NUM_TASKS", "10")))
    workers = int(os.environ.get("WORKERS", "10"))
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
        f"socialmaze_{task}_{provider}_{model.replace('/', '_')}_{run_kind}_{skill_name}_"
        f"{limit}_parallel{workers}_reasoning{reasoning_suffix}"
    )
    run_id = os.environ.get("RUN_ID", default_run_id)
    out_root = Path(os.environ.get("OUT_ROOT", str(PROJECT_ROOT / "results/cobras/socialmaze_hard" / run_id)))
    out_root.mkdir(parents=True, exist_ok=True)

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
        "[socialmaze-local] "
        f"task={task} provider={provider} model={target_model} limit={limit} workers={workers} "
        f"reasoning={reasoning_effort or 'off'} skill={skill_path or 'none'} out={out_root}",
        flush=True,
    )
    items = load_items(task, limit, seed)
    print(f"[socialmaze-local] loaded_items={len(items)}", flush=True)

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_one,
                row,
                task=task,
                out_root=out_root,
                skill_content=skill_content,
                max_completion_tokens=max_completion_tokens,
                timeout=timeout,
            ): row
            for row in items
        }
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                done = len(results)
                hard = sum(int(r["hard"]) for r in results)
                print(
                    f"[socialmaze-local] {done}/{len(items)} id={result['id']} hard={result['hard']} "
                    f"soft={result['soft']:.2f} acc={hard / max(done, 1):.3f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                item_id = safe_name(row.get("id"))
                failures.append(item_id)
                trace = traceback.format_exc()
                print(f"[socialmaze-local] failed id={item_id}: {exc}\n{trace}", flush=True)

    order = {safe_name(row.get("id")): idx for idx, row in enumerate(items)}
    results.sort(key=lambda r: order.get(r["id"], 10**9))
    config = {
        "dataset": "xzx34/SocialMaze local json",
        "task": task,
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
