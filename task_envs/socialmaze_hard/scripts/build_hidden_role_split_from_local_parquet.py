from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = Path(
    os.environ.get(
        "SOURCE_PATH",
        str(ROOT / "data/socialmaze_hard_shards/hard-00000-of-00005.parquet"),
    )
)
OUT_DIR = Path(
    os.environ.get(
        "OUT_DIR",
        str(ROOT / "data/socialmaze_hard_hidden_role_local150"),
    )
)
SOURCE_SPLIT = os.environ.get("SOURCE_SPLIT", "hard")
TRAIN_N = int(os.environ.get("TRAIN_N", "50"))
TEST_N = int(os.environ.get("TEST_N", "100"))
SEED = int(os.environ.get("SAMPLE_SEED", "20260713"))

ANSWER_RE = re.compile(
    r"Final Criminal Is Player\s*(\d+)\.\s*My Role Is\s*"
    r"(Investigator|Criminal|Rumormonger|Lunatic)",
    re.IGNORECASE,
)


def parse_answer(answer: str) -> tuple[int, str] | None:
    match = ANSWER_RE.search(answer or "")
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if not SOURCE_PATH.exists():
        raise SystemExit(f"SOURCE_PATH not found: {SOURCE_PATH}")

    total_needed = TRAIN_N + TEST_N
    rows: list[dict[str, Any]] = []
    scanned = 0
    parquet_file = pq.ParquetFile(SOURCE_PATH)
    source_index = 0
    skipped_row_groups: list[dict[str, Any]] = []
    for row_group in range(parquet_file.metadata.num_row_groups):
        try:
            table = parquet_file.read_row_group(row_group)
        except Exception as exc:  # noqa: BLE001
            skipped_row_groups.append(
                {
                    "row_group": row_group,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            source_index += parquet_file.metadata.row_group(row_group).num_rows
            continue
        for row in table.to_pylist():
            scanned += 1
            parsed = parse_answer(str(row.get("answer", "")))
            if not parsed:
                source_index += 1
                continue
            criminal, role = parsed
            item = dict(row)
            item["source_split"] = SOURCE_SPLIT
            item["source_path"] = str(SOURCE_PATH)
            item["source_row_group"] = row_group
            item["source_index"] = source_index
            item["id"] = f"{SOURCE_SPLIT}_{source_index}"
            item["gold_criminal_player"] = criminal
            item["gold_role"] = role
            rows.append(item)
            source_index += 1
            if len(rows) >= total_needed:
                break
        if len(rows) >= total_needed:
            break

    if len(rows) < total_needed:
        raise SystemExit(
            f"Not enough parsable samples: got={len(rows)} need={total_needed} scanned={scanned}"
        )

    rng = random.Random(SEED)
    rng.shuffle(rows)
    train = rows[:TRAIN_N]
    test = rows[TRAIN_N : TRAIN_N + TEST_N]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "test.jsonl", test)
    summary = {
        "dataset": "MBZUAI/SocialMaze",
        "source_split": SOURCE_SPLIT,
        "source_path": str(SOURCE_PATH),
        "seed": SEED,
        "scanned": scanned,
        "skipped_row_groups": skipped_row_groups,
        "train_n": TRAIN_N,
        "test_n": TEST_N,
        "train": summarize(train),
        "test": summarize(test),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
