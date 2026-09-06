from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(os.environ.get("OUT_DIR", str(ROOT / "data/socialmaze_hidden_role_balanced")))
ANSWER_RE = re.compile(
    r"Final Criminal Is Player\s*(\d+)\.\s*My Role Is\s*"
    r"(Investigator|Criminal|Rumormonger|Lunatic)",
    re.IGNORECASE,
)


TRAIN_QUOTAS = {
    "Rumormonger": 20,
    "Lunatic": 12,
    "Investigator": 9,
    "Criminal": 9,
}
TEST_QUOTAS = {
    "Rumormonger": 40,
    "Lunatic": 25,
    "Investigator": 18,
    "Criminal": 17,
}


def parse_answer(answer: str) -> tuple[int, str] | None:
    match = ANSWER_RE.search(answer or "")
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    role_counts = Counter(row["gold_role"] for row in rows)
    criminal_counts = Counter(str(row["gold_criminal_player"]) for row in rows)
    return {
        "n": len(rows),
        "role_counts": dict(sorted(role_counts.items())),
        "role_fraction": {
            role: role_counts[role] / max(len(rows), 1)
            for role in sorted(role_counts)
        },
        "criminal_player_counts": dict(sorted(criminal_counts.items())),
    }


def main() -> None:
    split = os.environ.get("SOURCE_SPLIT", "easy")
    seed = int(os.environ.get("SAMPLE_SEED", "20260713"))
    rng = random.Random(seed)
    quotas = {
        role: TRAIN_QUOTAS.get(role, 0) + TEST_QUOTAS.get(role, 0)
        for role in set(TRAIN_QUOTAS) | set(TEST_QUOTAS)
    }
    buckets: dict[str, list[dict[str, Any]]] = {role: [] for role in quotas}
    seen_source_ids: set[int] = set()

    scanned = 0
    for source_index, row in enumerate(load_dataset("MBZUAI/SocialMaze", split=split, streaming=True)):
        scanned += 1
        if scanned % 1000 == 0:
            have = {role: len(items) for role, items in buckets.items()}
            print(f"[socialmaze-split] split={split} scanned={scanned} have={have}", flush=True)
        parsed = parse_answer(row.get("answer", ""))
        if not parsed:
            continue
        criminal, role = parsed
        if role not in buckets or len(buckets[role]) >= quotas[role]:
            continue
        if source_index in seen_source_ids:
            continue
        seen_source_ids.add(source_index)
        item = dict(row)
        item["source_split"] = split
        item["source_index"] = source_index
        item["id"] = f"{split}_{source_index}"
        item["gold_criminal_player"] = criminal
        item["gold_role"] = role
        buckets[role].append(item)
        if all(len(buckets[role]) >= quotas[role] for role in quotas):
            break

    missing = {
        role: quotas[role] - len(items)
        for role, items in buckets.items()
        if len(items) < quotas[role]
    }
    if missing:
        raise SystemExit(f"Not enough samples after scanning {scanned}: {missing}")

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for role in sorted(quotas):
        items = list(buckets[role])
        rng.shuffle(items)
        train.extend(items[: TRAIN_QUOTAS[role]])
        test.extend(items[TRAIN_QUOTAS[role] : TRAIN_QUOTAS[role] + TEST_QUOTAS[role]])
    rng.shuffle(train)
    rng.shuffle(test)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "test.jsonl", test)
    summary = {
        "dataset": "MBZUAI/SocialMaze",
        "source_split": split,
        "seed": seed,
        "scanned": scanned,
        "train_quotas": TRAIN_QUOTAS,
        "test_quotas": TEST_QUOTAS,
        "train": summarize(train),
        "test": summarize(test),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
