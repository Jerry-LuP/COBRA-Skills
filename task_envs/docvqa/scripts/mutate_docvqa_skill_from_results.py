from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from cobras.cobras_core.model import (
    chat_optimizer,
    chat_optimizer_messages,
    configure_azure_openai,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_reasoning_effort,
)


ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = ROOT

META_MUTATION_SKILL = """You are an expert DocVQA skill editor.

You will receive:
1. An existing DocVQA SKILL.md.
2. A small random sample of rollout observations produced by a weaker target model using that skill.
3. The document images for the sampled observations, when available.

Your job is to make a conservative mutation of the existing skill: preserve its overall structure and intent, but refine wording, replace confusing instructions, delete low-value wording, and improve visual answer localization and exact-span output behavior.

Important constraints:
- Do not rewrite from scratch.
- Make at most 2 to 4 conceptual edits.
- Prefer replacing or deleting wording over adding wording.
- Add at most 2 new bullets total.
- Do not add new top-level sections unless the existing skill is missing a crucial section.
- Do not add concrete training examples, exact task IDs, exact gold answers, exact questions, document filenames, or memorized answers.
- Keep the skill generic and reusable across DocVQA.
- Prefer small, high-signal edits over prompt bloat.
- Keep the mutated skill at or below the original length when possible.

Use the observations and images to infer what should change; do not optimize for the sampled rows directly.

Output only the full mutated SKILL.md content. Do not wrap it in code fences. Do not explain your changes.
"""


def load_env_files() -> None:
    for env_path in [
        PROJECT_ROOT / ".env.docvqa.local",
        PROJECT_ROOT / ".env",
        ROOT / ".env.docvqa.local",
        ROOT / ".env",
    ]:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("export "):
                s = s[len("export "):].strip()
            if "=" not in s:
                continue
            key, value = s.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clip(text: object, limit: int) -> str:
    s = str(text or "").strip()
    return s if len(s) <= limit else s[:limit] + "\n...[truncated]"


def read_text(path: Path, limit: int) -> str:
    if not path.exists():
        return ""
    try:
        return clip(path.read_text(encoding="utf-8"), limit)
    except UnicodeDecodeError:
        return clip(path.read_text(errors="ignore"), limit)


def read_conversation(pred_dir: Path, max_chars: int, max_message_chars: int) -> str:
    conv_path = pred_dir / "conversation.json"
    if not conv_path.exists():
        return ""
    try:
        conv = json.loads(conv_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"<failed to read conversation.json: {exc}>"
    rendered = []
    for event in conv:
        role = event.get("role") or event.get("type") or "event"
        turn = event.get("turn")
        prefix = str(role)
        if turn is not None:
            prefix += f"[turn={turn}]"
        rendered.append(f"{prefix}: {clip(event.get('content', ''), max_message_chars)}")
    return clip("\n\n".join(rendered), max_chars)


def image_to_data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def sample_rows(
    rows: list[dict],
    *,
    sample_size: int,
    seed: int | None,
    failure_fraction: float,
    sample_policy: str,
) -> list[dict]:
    rng = random.Random(seed)
    sample_size = max(0, min(sample_size, len(rows)))
    if sample_policy == "random":
        selected = list(rows)
        rng.shuffle(selected)
        return selected[:sample_size]

    successes = [row for row in rows if int(row.get("hard") or 0)]
    failures = [row for row in rows if not int(row.get("hard") or 0)]
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def add_random(pool: list[dict]) -> bool:
        candidates = [row for row in pool if str(row.get("id")) not in selected_ids]
        if not candidates:
            return False
        row = rng.choice(candidates)
        selected.append(row)
        selected_ids.add(str(row.get("id")))
        return True

    if sample_size and successes and failures:
        add_random(successes)
        add_random(failures)
    elif sample_size:
        add_random(rows)

    target_failures = round(sample_size * max(0.0, min(1.0, failure_fraction)))
    while len(selected) < sample_size:
        n_fail = sum(1 for row in selected if not int(row.get("hard") or 0))
        if failures and n_fail < target_failures and add_random(failures):
            continue
        if successes and add_random(successes):
            continue
        remaining = [row for row in rows if str(row.get("id")) not in selected_ids]
        if not remaining:
            break
        row = rng.choice(remaining)
        selected.append(row)
        selected_ids.add(str(row.get("id")))
    rng.shuffle(selected)
    return selected


def build_observations(
    *,
    results_path: Path,
    predictions_dir: Path,
    sample_size: int,
    seed: int | None,
    failure_fraction: float,
    sample_policy: str,
    include_prompt: bool,
    include_conversation: bool,
    max_prompt_chars: int,
    max_response_chars: int,
    max_conversation_chars: int,
    max_message_chars: int,
) -> tuple[dict, list[dict]]:
    rows = load_jsonl(results_path)
    selected = sample_rows(
        rows,
        sample_size=sample_size,
        seed=seed,
        failure_fraction=failure_fraction,
        sample_policy=sample_policy,
    )
    samples = []
    for idx, row in enumerate(selected, start=1):
        pred_dir = predictions_dir / str(row.get("id"))
        item = {
            "sample_label": f"sample_{idx:02d}",
            "id": row.get("id"),
            "outcome": "success" if int(row.get("hard") or 0) else "failure",
            "task_type": row.get("task_type", "docvqa"),
            "question": row.get("question", ""),
            "gold_answers": row.get("gold_answer", row.get("gold_answers", [])),
            "predicted_answer": row.get("predicted_answer", ""),
            "hard": int(row.get("hard") or 0),
            "soft": float(row.get("soft") or 0.0),
            "agent_ok": bool(row.get("agent_ok")),
            "n_turns": row.get("n_turns"),
            "fail_reason": row.get("fail_reason", ""),
            "response_excerpt": clip(row.get("response", ""), max_response_chars),
        }
        image_paths = row.get("image_paths") or []
        item["has_image"] = bool(isinstance(image_paths, list) and image_paths and image_paths[0])
        if include_prompt:
            item["prompt_excerpt"] = read_text(pred_dir / "target_user_prompt.txt", max_prompt_chars)
        if include_conversation:
            item["conversation_excerpt"] = read_conversation(
                pred_dir,
                max_chars=max_conversation_chars,
                max_message_chars=max_message_chars,
            )
        samples.append(item)
    hard_correct = sum(int(row.get("hard") or 0) for row in rows)
    observations = {
        "source_results": str(results_path),
        "count": len(rows),
        "hard_correct": hard_correct,
        "hard_acc": hard_correct / max(len(rows), 1),
        "avg_soft": sum(float(row.get("soft") or 0.0) for row in rows) / max(len(rows), 1),
        "n_failures": sum(1 for row in rows if not int(row.get("hard") or 0)),
        "n_successes": sum(1 for row in rows if int(row.get("hard") or 0)),
        "sample_size": len(samples),
        "sample_seed": seed,
        "sample_policy": (
            "Uniform random sample from this skill's rollout results. "
            "Gold/evaluation feedback may appear only to infer generic improvements."
            if sample_policy == "random"
            else
            "Random sample from this skill's rollout results, biased toward failures. "
            "Gold/evaluation feedback may appear only to infer generic improvements."
        ),
        "sample_observations": samples,
    }
    return observations, selected


def build_image_parts(rows: list[dict], *, image_detail: str, max_images: int) -> tuple[list[dict], list[dict]]:
    parts: list[dict] = []
    manifest: list[dict] = []
    included = 0
    for idx, row in enumerate(rows, start=1):
        if included >= max_images:
            break
        image_paths = row.get("image_paths") or []
        if not isinstance(image_paths, list) or not image_paths:
            continue
        image_path = str(image_paths[0] or "")
        image_file = Path(image_path)
        if not image_path or not image_file.exists():
            continue
        image_url = {"url": image_to_data_uri(image_path)}
        if image_detail and image_detail != "auto":
            image_url["detail"] = image_detail
        parts.append(
            {
                "type": "text",
                "text": f"Document image for sampled observation {idx}. Use it only to infer generic DocVQA skill edits.",
            }
        )
        parts.append({"type": "image_url", "image_url": image_url})
        manifest.append(
            {
                "sample_observation_index": idx,
                "id": str(row.get("id")),
                "image_path": image_path,
                "bytes": image_file.stat().st_size,
                "detail": image_detail,
            }
        )
        included += 1
    return parts, manifest


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", text).strip()


def discover_skill_runs(skills_root: Path, eval_root: Path) -> list[tuple[str, Path, Path, Path]]:
    runs = []
    for skill_path in sorted(skills_root.glob("*/skill.md")):
        skill_name = skill_path.parent.name
        run_root = eval_root / skill_name
        results_path = run_root / "results.jsonl"
        predictions_dir = run_root / "predictions"
        if not results_path.exists():
            print(f"[docvqa-skill-mutate] skip {skill_name}: missing {results_path}", flush=True)
            continue
        runs.append((skill_name, skill_path, results_path, predictions_dir))
    if not runs:
        raise SystemExit(f"No skill/result pairs found under {skills_root} and {eval_root}")
    return runs


def mutate_one(
    *,
    skill_name: str,
    skill_path: Path,
    results_path: Path,
    predictions_dir: Path,
    out_dir: Path,
    model: str,
    base_url: str,
    api_key: str,
    reasoning_effort: str | None,
    sample_size: int,
    sample_seed: int | None,
        failure_fraction: float,
        sample_policy: str,
    include_images: bool,
    image_detail: str,
    max_images: int,
    max_length_ratio: float,
    max_abs_chars: int,
    reject_over_length: bool,
    max_completion_tokens: int,
    timeout: int,
) -> dict:
    original_skill = skill_path.read_text(encoding="utf-8")
    observations, selected_rows = build_observations(
        results_path=results_path,
        predictions_dir=predictions_dir,
        sample_size=sample_size,
        seed=sample_seed,
        failure_fraction=failure_fraction,
        sample_policy=sample_policy,
        include_prompt=env_bool("MUTATION_INCLUDE_PROMPT", True),
        include_conversation=env_bool("MUTATION_INCLUDE_CONVERSATION", True),
        max_prompt_chars=int(os.environ.get("MAX_PROMPT_CHARS", "1200")),
        max_response_chars=int(os.environ.get("MAX_RESPONSE_CHARS", "900")),
        max_conversation_chars=int(os.environ.get("MAX_CONVERSATION_CHARS", "2200")),
        max_message_chars=int(os.environ.get("MAX_MESSAGE_CHARS", "900")),
    )
    original_len = len(original_skill)
    length_limit = min(max_abs_chars, max(1200, int(original_len * max_length_ratio)))
    user = (
        META_MUTATION_SKILL
        + "\n\n## Mutation Budget\n"
        + f"- Original length: {original_len} characters.\n"
        + f"- Target maximum mutated length: {length_limit} characters.\n"
        + "- Prefer deleting, replacing, or tightening confusing wording before adding new rules.\n"
        + "\n\n## Existing SKILL.md\n"
        + original_skill.strip()
        + "\n\n## Rollout Observations\n"
        + json.dumps(observations, ensure_ascii=False, indent=2)
        + "\n\n## Required Output\n"
        + "Return the full mutated DocVQA SKILL.md. Keep it generic and do not include concrete sampled task IDs, gold answers, exact questions, or document filenames.\n"
    )
    image_parts: list[dict] = []
    image_manifest: list[dict] = []
    if include_images:
        image_parts, image_manifest = build_image_parts(
            selected_rows,
            image_detail=image_detail,
            max_images=max_images,
        )
        if image_parts:
            user += (
                "\nThe document images for sampled observations are attached after this text. "
                "Inspect them only to infer generic visual-reading and answer-localization edits."
            )

    set_optimizer_backend("openai_chat")
    set_optimizer_deployment(model)
    set_reasoning_effort(reasoning_effort)
    configure_azure_openai(
        endpoint=base_url,
        api_key=api_key,
        auth_mode="openai_compatible",
        optimizer_endpoint=base_url,
        optimizer_api_key=api_key,
        optimizer_auth_mode="openai_compatible",
    )
    print(
        f"[docvqa-skill-mutate] skill={skill_name} model={model} "
        f"hard={observations['hard_correct']}/{observations['count']} "
        f"samples={observations['sample_size']} images={len(image_manifest)} seed={sample_seed}",
        flush=True,
    )
    if image_parts:
        messages = [
            {"role": "system", "content": "You conservatively edit reusable SKILL.md instructions for multimodal DocVQA agents."},
            {"role": "user", "content": [{"type": "text", "text": user}, *image_parts]},
        ]
        response, usage = chat_optimizer_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=5,
            stage="docvqa_skill_mutation",
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
    else:
        response, usage = chat_optimizer(
            system="You conservatively edit reusable SKILL.md instructions for multimodal DocVQA agents.",
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=5,
            stage="docvqa_skill_mutation",
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    mutated = strip_code_fence(response)
    rejected = reject_over_length and len(mutated) > length_limit
    (out_dir / "original_skill.md").write_text(original_skill, encoding="utf-8")
    (out_dir / "mutation_prompt.txt").write_text(user, encoding="utf-8")
    (out_dir / "mutation_raw_response.txt").write_text(response, encoding="utf-8")
    (out_dir / "mutation_image_manifest.json").write_text(
        json.dumps(image_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "observations.json").write_text(
        json.dumps(observations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if rejected:
        (out_dir / "rejected_skill.md").write_text(mutated + "\n", encoding="utf-8")
        (out_dir / "skill.md").write_text(original_skill.rstrip() + "\n", encoding="utf-8")
    else:
        (out_dir / "skill.md").write_text(mutated + "\n", encoding="utf-8")
    summary = {
        "skill": skill_name,
        "source_skill": str(skill_path),
        "source_results": str(results_path),
        "out_dir": str(out_dir),
        "model": model,
        "sample_size": observations["sample_size"],
        "sample_seed": sample_seed,
        "n_attached_images": len(image_manifest),
        "hard_acc_before": observations["hard_acc"],
        "avg_soft_before": observations["avg_soft"],
        "original_chars": original_len,
        "mutated_chars": len(mutated),
        "target_max_chars": length_limit,
        "rejected": rejected,
        "usage": usage,
    }
    (out_dir / "mutation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    load_env_files()
    model = os.environ.get("MUTATION_MODEL", "openai/gpt-5.4")
    base_url = os.environ.get("TEACHER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.environ.get("TEACHER_API_KEY", "")
    reasoning_effort = os.environ.get("MUTATION_REASONING_EFFORT", "medium").strip() or None
    sample_size = int(os.environ.get("MUTATION_SAMPLE_SIZE", "8"))
    seed_raw = os.environ.get("MUTATION_SAMPLE_SEED", "").strip()
    sample_seed = int(seed_raw) if seed_raw else None
    skills_root = Path(
        os.environ.get(
            "SKILLS_ROOT",
            str(PROJECT_ROOT / "results/skills/docvqa/generated_docvqa_gpt54_medium_multimodal_sample10_seeds"),
        )
    )
    eval_root = Path(
        os.environ.get(
            "EVAL_ROOT",
            str(PROJECT_ROOT / "results/gpt-5.4-nano/docvqa/docvqa_openrouter_54nano_skill_sweep_generated_sample10_seed1to5_train50_parallel15_turn1_reasoningoff"),
        )
    )
    out_root = Path(
        os.environ.get(
            "OUT_ROOT",
            str(PROJECT_ROOT / "results/skills/docvqa/mutated_docvqa" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")),
        )
    )
    summaries = []
    for idx, (skill_name, skill_path, results_path, predictions_dir) in enumerate(discover_skill_runs(skills_root, eval_root)):
        seed = sample_seed + idx if sample_seed is not None else None
        summaries.append(
            mutate_one(
                skill_name=skill_name,
                skill_path=skill_path,
                results_path=results_path,
                predictions_dir=predictions_dir,
                out_dir=out_root / skill_name,
                model=model,
                base_url=base_url,
                api_key=api_key,
                reasoning_effort=reasoning_effort,
                sample_size=sample_size,
                sample_seed=seed,
                failure_fraction=float(os.environ.get("MUTATION_FAILURE_FRACTION", "0.65")),
                sample_policy=os.environ.get("MUTATION_SAMPLE_POLICY", "random").strip().lower() or "random",
                include_images=env_bool("MUTATION_INCLUDE_IMAGES", True),
                image_detail=os.environ.get("MUTATION_IMAGE_DETAIL", "auto").strip() or "auto",
                max_images=int(os.environ.get("MUTATION_MAX_IMAGES", str(sample_size))),
                max_length_ratio=float(os.environ.get("MUTATION_MAX_LENGTH_RATIO", "1.05")),
                max_abs_chars=int(os.environ.get("MUTATION_MAX_ABS_CHARS", "4200")),
                reject_over_length=env_bool("MUTATION_REJECT_OVER_LENGTH", False),
                max_completion_tokens=int(os.environ.get("MUTATION_MAX_COMPLETION_TOKENS", "6500")),
                timeout=int(os.environ.get("MUTATION_TIMEOUT", "300")),
            )
        )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "mutation_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[docvqa-skill-mutate] summary={out_root / 'mutation_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
