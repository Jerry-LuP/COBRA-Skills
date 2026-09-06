from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROVIDER = os.environ.get("LIVEMATH_PROVIDER", "yunwu").strip().lower() or "yunwu"
DEFAULT_SPLIT_ROOT = Path(os.environ.get("LIVEMATH_SPLIT_ROOT", str(PROJECT_ROOT / "data" / "livemath")))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_skill_tree(src_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)


def ensure_skill_md(skill_root: Path) -> None:
    lower = skill_root / "skill.md"
    upper = skill_root / "SKILL.md"
    if not lower.exists() and not upper.exists():
        raise FileNotFoundError(f"Expected generated skill file under {skill_root}")


def skill_entrypoint(skill_root: Path) -> Path:
    lower = skill_root / "skill.md"
    upper = skill_root / "SKILL.md"
    if lower.exists():
        return lower
    if upper.exists():
        return upper
    raise FileNotFoundError(f"Could not find skill.md or SKILL.md under {skill_root}")


def skill_file_manifest(root: Path) -> list[str]:
    return [str(path.relative_to(root)).replace(os.sep, "/") for path in sorted(root.glob("**/*")) if path.is_file()]


def split_items_path(split_root: Path, split: str) -> Path:
    return split_root / split / "items.json"


def load_split_items(split_root: Path, split: str, limit: int = 0) -> list[dict[str, Any]]:
    path = split_items_path(split_root, split)
    if not path.exists():
        raise FileNotFoundError(
            f"LiveMath data is not materialized at {path}. Run: "
            "cobra-materialize-data livemath --source-dir /path/to/monthly/files"
        )
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"Expected list in {path}")
    return items if limit <= 0 else items[:limit]


def normalize_choices(raw_choices: Any) -> list[dict[str, Any]]:
    if isinstance(raw_choices, list):
        choices: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_choices):
            if isinstance(item, dict):
                label = str(item.get("label") or chr(ord("A") + idx)).strip().upper()
                text = str(item.get("text") or item.get("content") or "").strip()
            else:
                label = chr(ord("A") + idx)
                text = str(item).strip()
            if text:
                choices.append({"label": label, "text": text})
        return choices
    if isinstance(raw_choices, dict):
        return [
            {"label": str(label).strip().upper(), "text": str(text).strip()}
            for label, text in raw_choices.items()
            if str(text).strip()
        ]
    return []


def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    mcq = item.get("mcq") if isinstance(item.get("mcq"), dict) else {}
    question = str(mcq.get("question") or item.get("question") or "").strip()
    choices = normalize_choices(mcq.get("choices") or item.get("choices") or [])
    correct = mcq.get("correct_choice") or item.get("correct_choice") or {}
    if isinstance(correct, dict):
        correct_label = str(correct.get("label") or "").strip().upper()
        correct_text = str(correct.get("text") or "").strip()
    else:
        correct_label = str(correct).strip().upper()
        correct_text = ""
    if correct_label and not correct_text:
        for choice in choices:
            if choice["label"] == correct_label:
                correct_text = choice["text"]
                break
    if correct_label and correct_text and not any(choice["label"] == correct_label for choice in choices):
        choices.append({"label": correct_label, "text": correct_text})
        choices.sort(key=lambda choice: choice["label"])
    theorem_type = item.get("theorem_type") or []
    if isinstance(theorem_type, str):
        theorem_type = [theorem_type]
    elif not isinstance(theorem_type, list):
        theorem_type = []
    return {
        "id": f"{item.get('month')}:{item.get('no')}",
        "month": item.get("month"),
        "no": item.get("no"),
        "paper_link": item.get("paper_link", ""),
        "question": question,
        "choices": choices,
        "correct_choice": {"label": correct_label, "text": correct_text},
        "theorem": str(item.get("theorem") or "").strip(),
        "sketch": str(item.get("sketch") or "").strip(),
        "theorem_type": theorem_type,
    }


def load_normalized_split_items(split_root: Path, split: str, limit: int = 0) -> list[dict[str, Any]]:
    return [normalize_item(item) for item in load_split_items(split_root, split, limit=limit)]
