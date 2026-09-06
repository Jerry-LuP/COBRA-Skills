#!/usr/bin/env python3
"""Materialize benchmark payloads from checked-in, ID-only split manifests."""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

from cobras.cobras_core.paths import COBRAS_ROOT
from cobras.task_envs.livemath.scripts.reorder_choices import _rewrite_item


DEFAULT_DATA_ROOT = COBRAS_ROOT / "data"
DEFAULT_MANIFEST_ROOT = DEFAULT_DATA_ROOT / "manifests"
DATASETS = ("searchqa", "docvqa", "livemath")


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing split manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _manifest(manifest_root: Path, dataset: str) -> dict[str, Any]:
    payload = _read_json(manifest_root / dataset / "manifest.json")
    if payload.get("dataset") != dataset:
        raise ValueError(f"Manifest dataset mismatch for {dataset!r}")
    return payload


def _split_ids(manifest_root: Path, dataset: str, split: str) -> list[str]:
    payload = _read_json(manifest_root / dataset / "splits" / f"{split}.json")
    ids = payload.get("ids")
    if not isinstance(ids, list) or not all(isinstance(item, str) and item for item in ids):
        raise ValueError(f"Expected a non-empty string ID list in {dataset}/{split}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {dataset}/{split}")
    return ids


def _load_dataset_function() -> Callable[..., Iterable[dict[str, Any]]]:
    try:
        module = importlib.import_module("datasets")
        loader = getattr(module, "load_dataset")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Hugging Face datasets is required for this command. "
            "Install the data extras with: python -m pip install -e '.[data]'"
        ) from exc
    return loader


def _load_hf_split(
    loader: Callable[..., Iterable[dict[str, Any]]],
    repo: str,
    config: str,
    split: str,
    cache_dir: Path | None,
    revision: str = "",
) -> Iterable[dict[str, Any]]:
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if revision:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if config:
        return loader(repo, config, **kwargs)
    return loader(repo, **kwargs)


def _clear(paths: Iterable[Path], force: bool) -> None:
    if not force:
        return
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def materialize_searchqa(
    *,
    data_root: Path,
    manifest_root: Path,
    cache_dir: Path | None = None,
    force: bool = False,
    load_dataset: Callable[..., Iterable[dict[str, Any]]] | None = None,
) -> None:
    """Download only the SearchQA rows selected by the release manifests."""
    manifest = _manifest(manifest_root, "searchqa")
    split_names = list(manifest["splits"])
    ids_by_split = {
        split: _split_ids(manifest_root, "searchqa", split)
        for split in split_names
    }
    wanted = {item_id for ids in ids_by_split.values() for item_id in ids}
    if len(wanted) != sum(len(ids) for ids in ids_by_split.values()):
        raise ValueError("SearchQA IDs overlap across release splits")

    output_root = data_root / "searchqa"
    _clear([output_root], force)
    loader = load_dataset or _load_dataset_function()
    upstream = manifest["upstream"]
    rows: dict[str, dict[str, Any]] = {}
    for source_split in upstream["splits"]:
        dataset = _load_hf_split(
            loader,
            str(upstream["repo"]),
            str(upstream.get("config") or ""),
            str(source_split),
            cache_dir,
            str(upstream.get("revision") or ""),
        )
        for row in dataset:
            item_id = str(row.get("key") or row.get("id") or "")
            if item_id not in wanted or item_id in rows:
                continue
            answers = row.get("answers") or []
            if isinstance(answers, str):
                answers = [answers]
            rows[item_id] = {
                "id": item_id,
                "question": str(row.get("question") or ""),
                "context": str(row.get("context") or ""),
                "answers": [str(answer) for answer in answers],
            }
            if len(rows) == len(wanted):
                break
        if len(rows) == len(wanted):
            break

    missing = sorted(wanted - rows.keys())
    if missing:
        raise RuntimeError(
            f"SearchQA source is missing {len(missing)} selected IDs; "
            f"first missing ID: {missing[0]}"
        )
    for split, ids in ids_by_split.items():
        _write_json(output_root / split / "items.json", [rows[item_id] for item_id in ids])
        print(f"[materialize-data] searchqa {split}={len(ids)}", flush=True)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _save_image(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image, "save"):
        image.save(path, format="PNG")
        return
    if isinstance(image, dict):
        if image.get("bytes"):
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError("Pillow is required to decode DocVQA images") from exc
            with Image.open(io.BytesIO(image["bytes"])) as decoded:
                decoded.save(path, format="PNG")
            return
        if image.get("path"):
            shutil.copy2(image["path"], path)
            return
    raise TypeError(f"Unsupported DocVQA image value: {type(image).__name__}")


DOCVQA_FIELDS = [
    "id",
    "questionId",
    "docId",
    "question",
    "answer",
    "answers",
    "ground_truth",
    "image_path",
    "image_paths",
    "topic",
    "question_types",
    "ucsf_document_id",
    "ucsf_document_page_no",
    "source_dataset",
    "source_config",
    "source_split",
    "data_split",
    "sample_seed",
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DOCVQA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def materialize_docvqa(
    *,
    data_root: Path,
    manifest_root: Path,
    cache_dir: Path | None = None,
    force: bool = False,
    load_dataset: Callable[..., Iterable[dict[str, Any]]] | None = None,
) -> None:
    """Download selected DocVQA rows and their document images."""
    manifest = _manifest(manifest_root, "docvqa")
    split_names = list(manifest["splits"])
    ids_by_split = {
        split: _split_ids(manifest_root, "docvqa", split)
        for split in split_names
    }
    owner = {
        item_id: split
        for split, ids in ids_by_split.items()
        for item_id in ids
    }
    if len(owner) != sum(len(ids) for ids in ids_by_split.values()):
        raise ValueError("DocVQA IDs overlap across release splits")

    output_root = data_root / "docvqa"
    image_root = data_root / "docvqa_images"
    _clear([output_root / "splits", image_root], force)
    loader = load_dataset or _load_dataset_function()
    upstream = manifest["upstream"]
    dataset = _load_hf_split(
        loader,
        str(upstream["repo"]),
        str(upstream.get("config") or ""),
        str(upstream["split"]),
        cache_dir,
        str(upstream.get("revision") or ""),
    )

    rows: dict[str, dict[str, Any]] = {}
    for row in dataset:
        question_id = str(row.get("questionId") or row.get("id") or "")
        if question_id not in owner or question_id in rows:
            continue
        doc_id = str(row.get("docId") or "")
        image_path = (image_root / f"q{question_id}_d{doc_id}.png").resolve()
        if force or not image_path.exists():
            _save_image(row.get("image"), image_path)
        answers = _string_list(row.get("answers"))
        question_types = _string_list(row.get("question_types"))
        rows[question_id] = {
            "id": question_id,
            "questionId": question_id,
            "docId": doc_id,
            "question": str(row.get("question") or ""),
            "answer": answers[0] if answers else "",
            "answers": json.dumps(answers, ensure_ascii=False),
            "ground_truth": json.dumps(answers, ensure_ascii=False),
            "image_path": str(image_path),
            "image_paths": json.dumps([str(image_path)], ensure_ascii=False),
            "topic": "|".join(question_types),
            "question_types": json.dumps(question_types, ensure_ascii=False),
            "ucsf_document_id": str(row.get("ucsf_document_id") or ""),
            "ucsf_document_page_no": str(row.get("ucsf_document_page_no") or ""),
            "source_dataset": str(upstream["repo"]),
            "source_config": str(upstream.get("config") or ""),
            "source_split": str(upstream["split"]),
            "data_split": str(upstream.get("data_split") or upstream["split"]),
            "sample_seed": str(manifest["selection"]["name"]),
        }
        if len(rows) == len(owner):
            break

    missing = sorted(owner.keys() - rows.keys())
    if missing:
        raise RuntimeError(
            f"DocVQA source is missing {len(missing)} selected IDs; "
            f"first missing ID: {missing[0]}"
        )
    for split, ids in ids_by_split.items():
        _write_csv(output_root / "splits" / split / "items.csv", [rows[item_id] for item_id in ids])
        print(f"[materialize-data] docvqa {split}={len(ids)}", flush=True)


def _livemath_source_index(source_dir: Path) -> dict[str, dict[str, Any]]:
    candidates = sorted(source_dir.rglob("qa_*_final.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No qa_*_final.json files found under LiveMath source directory: {source_dir}"
        )
    index: dict[str, dict[str, Any]] = {}
    for path in candidates:
        match = re.search(r"qa_(\d{6})_final\.json$", path.name)
        file_month = match.group(1) if match else ""
        payload = _read_json(path)
        if isinstance(payload, dict):
            payload = payload.get("data") or payload.get("items") or list(payload.values())
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list of LiveMath items in {path}")
        for raw_item in payload:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            month = str(item.get("month") or file_month)
            item["month"] = month
            item_id = f"{month}:{item.get('no')}"
            if item_id in index:
                raise ValueError(f"Duplicate LiveMath item {item_id} in {path}")
            index[item_id] = item
    return index


def materialize_livemath(
    *,
    data_root: Path,
    manifest_root: Path,
    source_dir: Path,
    force: bool = False,
) -> None:
    """Select the release split from user-obtained LiveMath monthly files."""
    manifest = _manifest(manifest_root, "livemath")
    split_specs = manifest["splits"]
    split_order = list(manifest["transforms"].get("split_order") or split_specs)
    output_splits = {
        split
        for split, spec in split_specs.items()
        if bool(spec.get("materialize", True))
    }
    missing_from_order = sorted(output_splits - set(split_order))
    if missing_from_order:
        raise ValueError(
            f"LiveMath transform order omits output splits: {missing_from_order}"
        )
    ids_by_split = {
        split: _split_ids(manifest_root, "livemath", split)
        for split in split_order
    }
    output_root = data_root / "livemath"
    _clear([output_root], force)
    source_index = _livemath_source_index(source_dir.resolve())
    wanted = {item_id for ids in ids_by_split.values() for item_id in ids}
    missing = sorted(wanted - source_index.keys())
    if missing:
        raise RuntimeError(
            f"LiveMath source is missing {len(missing)} selected IDs; "
            f"first missing ID: {missing[0]}"
        )

    rng = random.Random(int(manifest["transforms"]["choice_shuffle_seed"]))
    for split in split_order:
        ids = ids_by_split[split]
        selected = [_rewrite_item(source_index[item_id], rng) for item_id in ids]
        if split not in output_splits:
            continue
        _write_json(output_root / split / "items.json", selected)
        print(f"[materialize-data] livemath {split}={len(ids)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize benchmark payloads from COBRA-Skills ID-only manifests."
    )
    parser.add_argument("dataset", choices=(*DATASETS, "all"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Directory containing user-obtained LiveMath qa_*_final.json files.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing materialized files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    if "livemath" in selected and args.source_dir is None:
        raise SystemExit("livemath requires --source-dir /path/to/LiveMath/monthly/files")
    for dataset in selected:
        if dataset == "searchqa":
            materialize_searchqa(
                data_root=args.data_root,
                manifest_root=args.manifest_root,
                cache_dir=args.cache_dir,
                force=args.force,
            )
        elif dataset == "docvqa":
            materialize_docvqa(
                data_root=args.data_root,
                manifest_root=args.manifest_root,
                cache_dir=args.cache_dir,
                force=args.force,
            )
        else:
            materialize_livemath(
                data_root=args.data_root,
                manifest_root=args.manifest_root,
                source_dir=args.source_dir,
                force=args.force,
            )


if __name__ == "__main__":
    main()
