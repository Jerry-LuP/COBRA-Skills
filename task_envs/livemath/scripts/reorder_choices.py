#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent

DEFAULT_INPUT_ROOT = _PROJECT_ROOT / "data" / "livemath"
DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "data" / "livemath_reordered"
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reorder LiveMath MCQ choices and rewrite correct labels.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    return parser.parse_args()


def _normalize_choice(choice: dict[str, Any], idx: int) -> dict[str, str]:
    label = str(choice.get("label") or chr(ord("A") + idx)).strip().upper()
    text = str(choice.get("text") or choice.get("content") or "").strip()
    return {"label": label, "text": text}


def _relabel_choices(choices: list[dict[str, str]], new_order: list[int]) -> tuple[list[dict[str, str]], str]:
    new_choices: list[dict[str, str]] = []
    correct_label = ""
    for new_idx, old_idx in enumerate(new_order):
        old_choice = choices[old_idx]
        new_label = chr(ord("A") + new_idx)
        new_choices.append({"label": new_label, "text": old_choice["text"]})
        if old_choice.get("is_correct"):
            correct_label = new_label
    if not correct_label:
        raise ValueError("Could not locate the correct choice in choices")
    return new_choices, correct_label


def _rewrite_item(item: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    mcq = item.get("mcq") if isinstance(item.get("mcq"), dict) else {}
    raw_choices = mcq.get("choices") or []
    choices = [_normalize_choice(choice, idx) for idx, choice in enumerate(raw_choices)]
    correct_choice = mcq.get("correct_choice") or {}
    correct_text = str(correct_choice.get("text") or "").strip()
    if not correct_text:
        raise ValueError(f"Missing correct_choice.text for item {item.get('month')}:{item.get('no')}")

    all_choices = list(choices) + [{"label": str(correct_choice.get("label") or "A").strip().upper(), "text": correct_text, "is_correct": True}]
    if len(all_choices) < 2:
        raise ValueError(f"Expected at least 2 choices, got {len(all_choices)} for item {item.get('month')}:{item.get('no')}")

    order = list(range(len(all_choices)))
    rng.shuffle(order)
    if order == list(range(len(all_choices))):
        order = order[1:] + order[:1]

    new_choices, correct_label = _relabel_choices(all_choices, order)
    correct_text = next(choice["text"] for choice in new_choices if choice["label"] == correct_label)

    new_item = dict(item)
    new_mcq = dict(mcq)
    new_mcq["choices"] = new_choices
    correct_choice = dict(mcq.get("correct_choice") or {})
    correct_choice["label"] = correct_label
    correct_choice["text"] = correct_text
    new_mcq["correct_choice"] = correct_choice
    new_item["mcq"] = new_mcq
    return new_item


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "seed": args.seed,
        "splits": args.splits,
        "description": "LiveMath split with reordered MCQ choices and corrected labels.",
    }
    (args.output_root / "reorder_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for split in args.splits:
        in_path = args.input_root / split / "items.json"
        if not in_path.exists():
            raise FileNotFoundError(in_path)
        items = json.loads(in_path.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise ValueError(f"Expected list in {in_path}")
        out_items = [_rewrite_item(item, rng) for item in items]
        out_dir = args.output_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "items.json").write_text(json.dumps(out_items, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
