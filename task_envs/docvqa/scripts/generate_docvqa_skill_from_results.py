from __future__ import annotations

import json
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

META_SKILL = """You are an expert skill writer for DocVQA.

Your job is to write a reusable SKILL.md for a weaker DocVQA target model. The target model receives a document image and a question about the document, then must return the final answer inside <answer>...</answer>.

Use the observed trajectories and rewards to infer general behavioral rules. The observations may include gold/evaluation feedback. Use that feedback only to identify reusable patterns. Do not memorize, quote, or leak task IDs, exact gold answers, exact questions, document filenames, or concrete training examples.

The generated skill must:
- be generic across DocVQA tasks,
- be useful when injected into the target model's system prompt,
- output only one valid SKILL.md with YAML frontmatter.

Do not assume any predefined list of failure modes. Infer what matters from the observations and images. Prefer broadly applicable document-question-answering guidance over dataset-specific rules or examples.

If several reusable lessons seem plausible, do not write a catch-all skill that tries to cover every possible issue. Choose a coherent high-impact hypothesis about what would most improve the weaker model, and write the skill around that hypothesis. A useful skill can be narrow if the narrow guidance is general, actionable, and well supported by the observations.

Avoid defaulting to a generic "copy the shortest exact span" checklist unless the observations strongly indicate that this is the single most important reusable lesson. Look for a more specific operational strategy in the trajectories: for example, a way to inspect document layout, decide what evidence is sufficient, resolve nearby competing text, handle answer format, or recover from a pattern of mistakes. Do not list these as required topics; choose the one coherent strategy that the observations make most useful.

Output only the final SKILL.md content. Do not wrap it in code fences.
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


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def clip(text: object, limit: int) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...[truncated]"


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
        content = event.get("content", "")
        prefix = str(role)
        if turn is not None:
            prefix += f"[turn={turn}]"
        rendered.append(f"{prefix}: {clip(content, max_message_chars)}")
    return clip("\n\n".join(rendered), max_chars)


def image_to_data_uri(path: str) -> str:
    import base64
    import mimetypes

    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def select_rows(rows: list[dict], sample_size: int, sample_seed: int | None) -> list[dict]:
    rng = random.Random(sample_seed)
    successes = [row for row in rows if int(row.get("hard") or 0)]
    failures = [row for row in rows if not int(row.get("hard") or 0)]
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def add_random(pool: list[dict]) -> None:
        candidates = [row for row in pool if str(row.get("id")) not in selected_ids]
        if not candidates:
            return
        row = rng.choice(candidates)
        selected.append(row)
        selected_ids.add(str(row.get("id")))

    sample_size = max(0, min(sample_size, len(rows)))
    if sample_size and successes and failures:
        add_random(successes)
        add_random(failures)
    elif sample_size:
        add_random(rows)

    remaining = [row for row in rows if str(row.get("id")) not in selected_ids]
    rng.shuffle(remaining)
    for row in remaining[: max(0, sample_size - len(selected))]:
        selected.append(row)
        selected_ids.add(str(row.get("id")))
    rng.shuffle(selected)
    return selected


def compact_row(
    row: dict,
    *,
    predictions_dir: Path,
    include_prompt: bool,
    include_conversation: bool,
    max_prompt_chars: int,
    max_response_chars: int,
    max_conversation_chars: int,
    max_message_chars: int,
) -> dict:
    item_id = str(row.get("id"))
    pred_dir = predictions_dir / item_id
    item = {
        "id": item_id,
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
    if isinstance(image_paths, list) and image_paths:
        item["has_image"] = bool(image_paths[0])
    else:
        item["has_image"] = False
    if include_prompt:
        item["prompt_excerpt"] = read_text(pred_dir / "target_user_prompt.txt", max_prompt_chars)
    if include_conversation:
        item["conversation_excerpt"] = read_conversation(
            pred_dir,
            max_chars=max_conversation_chars,
            max_message_chars=max_message_chars,
        )
    return item


def build_observations(
    *,
    results_path: Path,
    predictions_dir: Path,
    sample_size: int,
    sample_seed: int | None,
    include_prompt: bool,
    include_conversation: bool,
    max_prompt_chars: int,
    max_response_chars: int,
    max_conversation_chars: int,
    max_message_chars: int,
) -> dict:
    rows = load_jsonl(results_path)
    selected = select_rows(rows, sample_size, sample_seed)
    failures = [row for row in rows if not int(row.get("hard") or 0)]
    successes = [row for row in rows if int(row.get("hard") or 0)]

    samples = []
    for row in selected:
        item = compact_row(
            row,
            predictions_dir=predictions_dir,
            include_prompt=include_prompt,
            include_conversation=include_conversation,
            max_prompt_chars=max_prompt_chars,
            max_response_chars=max_response_chars,
            max_conversation_chars=max_conversation_chars,
            max_message_chars=max_message_chars,
        )
        item["outcome"] = "success" if int(row.get("hard") or 0) else "failure"
        samples.append(item)

    return {
        "source_results": str(results_path),
        "predictions_dir": str(predictions_dir),
        "count": len(rows),
        "hard_correct": sum(int(row.get("hard") or 0) for row in rows),
        "avg_soft": sum(float(row.get("soft") or 0.0) for row in rows) / max(len(rows), 1),
        "n_failures": len(failures),
        "n_successes": len(successes),
        "sample_size": len(samples),
        "sample_seed": sample_seed,
        "sample_policy": (
            "Random sample from all rows. If both successful and failed rows exist, "
            "at least one of each is included. Gold answers and evaluation feedback may appear, "
            "but the generated skill must not copy or leak them."
        ),
        "sample_observations": samples,
    }


def build_image_parts(
    *,
    rows: list[dict],
    image_detail: str,
    max_images: int,
) -> tuple[list[dict], list[dict]]:
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
        if not image_path or not Path(image_path).exists():
            continue
        image_file = Path(image_path)
        image_url = {"url": image_to_data_uri(image_path)}
        if image_detail and image_detail != "auto":
            image_url["detail"] = image_detail
        parts.append(
            {
                "type": "text",
                "text": f"Document image for sampled observation {idx}. Use it only to infer generic DocVQA behavior.",
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


def main() -> None:
    load_env_files()
    model = os.environ.get("SKILL_GEN_MODEL", "openai/gpt-5.4")
    base_url = os.environ.get("TEACHER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.environ.get("TEACHER_API_KEY", "")
    reasoning_effort = os.environ.get("SKILL_GEN_REASONING_EFFORT", "medium").strip() or None
    max_completion_tokens = int(os.environ.get("SKILL_GEN_MAX_COMPLETION_TOKENS", "7000"))
    timeout = int(os.environ.get("SKILL_GEN_TIMEOUT", "300"))

    default_run_root = (
        PROJECT_ROOT
        / "results/gpt-5.4-nano/docvqa"
        / "docvqa_openrouter_54nano_baseline_train50_parallel15_turn1_reasoningoff"
    )
    run_root = Path(os.environ.get("RUN_ROOT", str(default_run_root)))
    results_path = Path(os.environ.get("RESULTS_PATH", str(run_root / "results.jsonl")))
    predictions_dir = Path(os.environ.get("PREDICTIONS_DIR", str(run_root / "predictions")))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        os.environ.get(
            "OUT_DIR",
            str(PROJECT_ROOT / "results/skills/docvqa" / f"generated_docvqa_{timestamp}"),
        )
    )
    sample_size = int(os.environ.get("SAMPLE_SIZE", "10"))
    sample_seed_raw = os.environ.get("SAMPLE_SEED", "").strip()
    sample_seed = int(sample_seed_raw) if sample_seed_raw else None
    include_prompt = os.environ.get("INCLUDE_PROMPT", "1").strip().lower() in {"1", "true", "yes"}
    include_conversation = os.environ.get("INCLUDE_CONVERSATION", "1").strip().lower() in {"1", "true", "yes"}
    include_images = os.environ.get("INCLUDE_IMAGES", "1").strip().lower() in {"1", "true", "yes"}
    teacher_image_detail = os.environ.get("TEACHER_IMAGE_DETAIL", "auto").strip() or "auto"
    max_images = int(os.environ.get("MAX_IMAGES", str(sample_size)))
    max_prompt_chars = int(os.environ.get("MAX_PROMPT_CHARS", "1200"))
    max_response_chars = int(os.environ.get("MAX_RESPONSE_CHARS", "900"))
    max_conversation_chars = int(os.environ.get("MAX_CONVERSATION_CHARS", "2200"))
    max_message_chars = int(os.environ.get("MAX_MESSAGE_CHARS", "900"))
    strategy_hint = os.environ.get("SKILL_STRATEGY_HINT", "").strip()
    reference_skill_path = Path(
        os.environ.get(
            "REFERENCE_SKILL_PATH",
            str(PROJECT_ROOT / "results/skills/docvqa/docvqa_manual_exact_v1/skill.md"),
        )
    )

    if not api_key:
        raise SystemExit("Missing OpenRouter API key")
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")
    if not predictions_dir.exists():
        raise SystemExit(f"Predictions dir not found: {predictions_dir}")

    rows = load_jsonl(results_path)
    selected_rows = select_rows(rows, sample_size, sample_seed)
    failures = [row for row in rows if not int(row.get("hard") or 0)]
    successes = [row for row in rows if int(row.get("hard") or 0)]

    observations = build_observations(
        results_path=results_path,
        predictions_dir=predictions_dir,
        sample_size=sample_size,
        sample_seed=sample_seed,
        include_prompt=include_prompt,
        include_conversation=include_conversation,
        max_prompt_chars=max_prompt_chars,
        max_response_chars=max_response_chars,
        max_conversation_chars=max_conversation_chars,
        max_message_chars=max_message_chars,
    )
    # Reuse the exact selected rows for image attachments by matching IDs in sampled order.
    selected_by_id = {str(row.get("id")): row for row in selected_rows}
    sampled_rows_for_images = [
        selected_by_id.get(str(item.get("id")))
        for item in observations["sample_observations"]
    ]
    sampled_rows_for_images = [row for row in sampled_rows_for_images if row is not None]

    reference_skill = ""
    if reference_skill_path.exists() and os.environ.get("INCLUDE_REFERENCE_SKILL", "0").strip().lower() in {"1", "true", "yes"}:
        reference_skill = reference_skill_path.read_text(encoding="utf-8")

    system = "You write high-quality, generalizable SKILL.md instructions for multimodal LLM agents."
    user = (
        META_SKILL
        + "\n\n## Observed Baseline DocVQA Results\n"
        + json.dumps(observations, ensure_ascii=False, indent=2)
    )
    if strategy_hint:
        user += (
            "\n\n## Strategy Angle For This Skill\n"
            "Use this as a lens for choosing which reusable lesson to emphasize. "
            "Do not force it if the observations contradict it, and do not leak task-specific examples.\n\n"
            + strategy_hint
        )
    if reference_skill:
        user += (
            "\n\n## Optional Reference Skill\n"
            "The following hand-written skill performed well in a prior experiment. You may borrow general ideas, "
            "but improve from the observed baseline trajectories and keep the final skill generic.\n\n"
            + reference_skill.strip()
        )
    user += (
        "\n\n## Required Output\n"
        "Write one DocVQA SKILL.md. It must be generic and must not include exact task IDs, exact gold answers, "
        "exact questions, document filenames, or concrete training examples from the observations.\n"
    )
    if include_images:
        user += (
            "\nThe document images for sampled observations are attached after this text. "
            "Inspect them only to infer generic visual-reading and answer-localization patterns."
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

    image_parts = []
    image_manifest = []
    if include_images:
        image_parts, image_manifest = build_image_parts(
            rows=sampled_rows_for_images,
            image_detail=teacher_image_detail,
            max_images=max_images,
        )

    print(
        "[docvqa-skill-gen] "
        f"model={model} results={results_path} rows={observations['count']} "
        f"failures={observations['n_failures']} successes={observations['n_successes']} "
        f"samples={observations['sample_size']} seed={observations['sample_seed']} "
        f"images={len(image_parts) // 2} out={out_dir}",
        flush=True,
    )

    if image_parts:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user}, *image_parts]},
        ]
        response, usage = chat_optimizer_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=5,
            stage="docvqa_skill_generation",
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
    else:
        response, usage = chat_optimizer(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=5,
            stage="docvqa_skill_generation",
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    skill_text = strip_code_fence(response) + "\n"
    (out_dir / "skill.md").write_text(skill_text, encoding="utf-8")
    (out_dir / "skill_generation_prompt.txt").write_text(user, encoding="utf-8")
    (out_dir / "skill_generation_raw_response.txt").write_text(response, encoding="utf-8")
    (out_dir / "skill_generation_image_manifest.json").write_text(
        json.dumps(image_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "observations.json").write_text(
        json.dumps(observations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "skill_generation_summary.json").write_text(
        json.dumps(
            {
                "model": model,
                "base_url": base_url,
                "results_path": str(results_path),
                "predictions_dir": str(predictions_dir),
                "out_dir": str(out_dir),
                "reference_skill_path": str(reference_skill_path) if reference_skill else "",
                "n_rows": observations["count"],
                "n_failures": observations["n_failures"],
                "n_successes": observations["n_successes"],
                "sample_size": observations["sample_size"],
                "sample_seed": observations["sample_seed"],
                "sample_policy": observations["sample_policy"],
                "strategy_hint": strategy_hint,
                "include_images": include_images,
                "n_attached_images": len(image_parts) // 2,
                "teacher_image_detail": teacher_image_detail,
                "hard_correct": observations["hard_correct"],
                "avg_soft": observations["avg_soft"],
                "usage": usage,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[docvqa-skill-gen] skill={out_dir / 'skill.md'}", flush=True)
    print(f"[docvqa-skill-gen] usage={usage}", flush=True)


if __name__ == "__main__":
    main()
