"""SearchQA task dataloader."""
from __future__ import annotations

import json

from cobras.task_envs.datasets.base import SplitDataLoader


# ── Raw data loading utilities (for preprocessing / standalone eval) ─────

def _load_items(path: str) -> list[dict]:
    """Load items from JSON or JSONL file."""
    with open(path) as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data") or list(data.values())
    except json.JSONDecodeError:
        pass

    items = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


# ── Dataloader ───────────────────────────────────────────────────────────

class SearchQADataLoader(SplitDataLoader):
    """SearchQA dataloader.

    Each split directory (train/, val/, test/) contains a .json file —
    a JSON array of question items.
    """

    def load_raw_items(self, data_path: str) -> list[dict]:
        return _load_items(data_path)

    def load_split_items(self, split_path: str) -> list[dict]:
        try:
            return super().load_split_items(split_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"SearchQA data is not materialized at {split_path}. "
                "Install the data extras and run: cobra-materialize-data searchqa"
            ) from exc
