from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[3]
REPO_ID = "MBZUAI/SocialMaze"
SOURCE_SPLIT = os.environ.get("SOURCE_SPLIT", "hard")
OUT_DIR = Path(os.environ.get("OUT_DIR", str(ROOT / f"data/socialmaze_{SOURCE_SPLIT}_hidden_role_balanced")))
HF_CACHE_DIR = Path(os.environ.get("HF_CACHE_DIR", str(ROOT / "data/hf_socialmaze_cache")))
SEED = int(os.environ.get("SAMPLE_SEED", "20260713"))
MAX_SHARDS = int(os.environ.get("MAX_SHARDS", "10"))

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


def shard_name(split: str, index: int, total: int) -> str:
    return f"data/{split}-{index:05d}-of-{total:05d}.parquet"


def find_and_download_shard(split: str, index: int) -> tuple[Path, int] | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    for total in range(1, MAX_SHARDS + 1):
        filename = shard_name(split, index, total)
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                cache_dir=str(HF_CACHE_DIR),
                token=token,
                resume_download=True,
            )
            return Path(path), total
        except Exception as exc:  # noqa: BLE001
            if total == MAX_SHARDS:
                print(f"[socialmaze-shard] shard_missing_or_failed index={index} last={filename} error={type(exc).__name__}: {exc}", flush=True)
    return None


def iter_parquet_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def main() -> None:
    rng = random.Random(SEED)
    quotas = {
        role: TRAIN_QUOTAS.get(role, 0) + TEST_QUOTAS.get(role, 0)
        for role in set(TRAIN_QUOTAS) | set(TEST_QUOTAS)
    }
    buckets: dict[str, list[dict[str, Any]]] = {role: [] for role in quotas}
    scanned = 0
    downloaded: list[str] = []
    discovered_total: int | None = None

    for shard_index in range(MAX_SHARDS):
        if discovered_total is not None and shard_index >= discovered_total:
            break
        result = find_and_download_shard(SOURCE_SPLIT, shard_index)
        if result is None:
            break
        path, total = result
        discovered_total = total
        downloaded.append(str(path))
        print(f"[socialmaze-shard] reading split={SOURCE_SPLIT} shard={shard_index + 1}/{total} path={path}", flush=True)
        for row_index, row in enumerate(iter_parquet_rows(path)):
            scanned += 1
            parsed = parse_answer(str(row.get("answer", "")))
            if not parsed:
                continue
            criminal, role = parsed
            if role not in buckets or len(buckets[role]) >= quotas[role]:
                continue
            item = dict(row)
            item["source_split"] = SOURCE_SPLIT
            item["source_shard"] = shard_index
            item["source_row_index"] = row_index
            item["id"] = f"{SOURCE_SPLIT}_{shard_index}_{row_index}"
            item["gold_criminal_player"] = criminal
            item["gold_role"] = role
            buckets[role].append(item)
            have = {r: len(items) for r, items in buckets.items()}
            if scanned % 500 == 0:
                print(f"[socialmaze-shard] scanned={scanned} have={have}", flush=True)
            if all(len(buckets[role]) >= quotas[role] for role in quotas):
                break
        have = {role: len(items) for role, items in buckets.items()}
        print(f"[socialmaze-shard] after_shard={shard_index} scanned={scanned} have={have}", flush=True)
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
        "dataset": REPO_ID,
        "source_split": SOURCE_SPLIT,
        "seed": SEED,
        "scanned": scanned,
        "downloaded_shards": downloaded,
        "train_quotas": TRAIN_QUOTAS,
        "test_quotas": TEST_QUOTAS,
        "train": summarize(train),
        "test": summarize(test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
