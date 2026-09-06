from __future__ import annotations

import csv
import json
from pathlib import Path

from cobras.scripts.materialize_data import (
    materialize_docvqa,
    materialize_livemath,
    materialize_searchqa,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_manifest(root: Path, dataset: str, splits: dict[str, list[str]], **extra: object) -> None:
    manifest = {"dataset": dataset, "splits": {name: {"count": len(ids)} for name, ids in splits.items()}, **extra}
    _write_json(root / dataset / "manifest.json", manifest)
    for name, ids in splits.items():
        _write_json(root / dataset / "splits" / f"{name}.json", {"ids": ids})


def test_materialize_searchqa_preserves_manifest_order(tmp_path: Path) -> None:
    manifest_root = tmp_path / "manifests"
    data_root = tmp_path / "data"
    _write_manifest(
        manifest_root,
        "searchqa",
        {"train": ["b", "a"], "test": ["c"]},
        upstream={
            "repo": "example/searchqa",
            "splits": ["train", "validation"],
            "revision": "search-revision",
        },
    )
    source = {
        "train": [
            {"key": "a", "question": "qa", "context": "ca", "answers": ["aa"]},
            {"key": "ignored", "question": "x", "context": "x", "answers": ["x"]},
        ],
        "validation": [
            {"key": "c", "question": "qc", "context": "cc", "answers": ["ac"]},
            {"key": "b", "question": "qb", "context": "cb", "answers": ["ab"]},
        ],
    }

    calls: list[dict[str, object]] = []

    def fake_load_dataset(_repo: str, *, split: str, **kwargs: object):
        calls.append({"split": split, **kwargs})
        return source[split]

    stale = data_root / "searchqa/val/stale.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale", encoding="utf-8")
    materialize_searchqa(
        data_root=data_root,
        manifest_root=manifest_root,
        force=True,
        load_dataset=fake_load_dataset,
    )

    train = json.loads((data_root / "searchqa/train/items.json").read_text())
    test = json.loads((data_root / "searchqa/test/items.json").read_text())
    assert [row["id"] for row in train] == ["b", "a"]
    assert [row["id"] for row in test] == ["c"]
    assert train[0]["answers"] == ["ab"]
    assert not stale.exists()
    assert calls == [
        {"split": "train", "streaming": True, "revision": "search-revision"},
        {"split": "validation", "streaming": True, "revision": "search-revision"},
    ]


class _FakeImage:
    def save(self, path: Path, *, format: str) -> None:
        assert format == "PNG"
        Path(path).write_bytes(b"test-image")


def test_materialize_docvqa_writes_selected_csv_and_images(tmp_path: Path) -> None:
    manifest_root = tmp_path / "manifests"
    data_root = tmp_path / "data"
    _write_manifest(
        manifest_root,
        "docvqa",
        {"train": ["20"], "val": ["10"]},
        upstream={
            "repo": "example/docvqa",
            "config": "DocVQA",
            "split": "validation",
            "revision": "docvqa-revision",
            "data_split": "val",
        },
        selection={"name": "fixed"},
    )
    source = [
        {
            "questionId": "10",
            "docId": 7,
            "question": "Question 10?",
            "question_types": ["table/list"],
            "image": _FakeImage(),
            "answers": ["answer 10", "alt"],
            "ucsf_document_id": "doc",
            "ucsf_document_page_no": "1",
        },
        {
            "questionId": "20",
            "docId": 8,
            "question": "Question 20?",
            "question_types": ["form"],
            "image": _FakeImage(),
            "answers": ["answer 20"],
        },
    ]

    calls: list[dict[str, object]] = []

    def fake_load_dataset(_repo: str, _config: str, **kwargs: object):
        calls.append(kwargs)
        return source

    materialize_docvqa(
        data_root=data_root,
        manifest_root=manifest_root,
        load_dataset=fake_load_dataset,
    )

    with (data_root / "docvqa/splits/val/items.csv").open(newline="") as handle:
        val_rows = list(csv.DictReader(handle))
    assert [row["questionId"] for row in val_rows] == ["10"]
    assert json.loads(val_rows[0]["answers"]) == ["answer 10", "alt"]
    assert val_rows[0]["data_split"] == "val"
    assert Path(val_rows[0]["image_path"]).read_bytes() == b"test-image"
    assert calls == [
        {
            "split": "validation",
            "streaming": True,
            "revision": "docvqa-revision",
        }
    ]


def test_materialize_livemath_selects_and_relabels_choices(tmp_path: Path) -> None:
    manifest_root = tmp_path / "manifests"
    data_root = tmp_path / "data"
    source_root = tmp_path / "source"
    split_ids = {"train": ["202601:2"], "dev": ["202601:1"], "test": ["202601:3"]}
    _write_manifest(
        manifest_root,
        "livemath",
        split_ids,
        upstream={"repo": "example/livemath"},
        transforms={
            "choice_shuffle_seed": 42,
            "split_order": ["train", "dev", "test"],
        },
    )
    manifest_path = manifest_root / "livemath/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["dev"]["materialize"] = False
    _write_json(manifest_path, manifest)
    rows = []
    for number in (1, 2, 3):
        rows.append(
            {
                "month": "202601",
                "no": number,
                "mcq": {
                    "question": f"Question {number}?",
                    "choices": [
                        {"label": "A", "text": f"wrong-{number}-1"},
                        {"label": "B", "text": f"wrong-{number}-2"},
                    ],
                    "correct_choice": {"label": "C", "text": f"right-{number}"},
                },
            }
        )
    _write_json(source_root / "data/202601/qa_202601_final.json", rows)

    materialize_livemath(
        data_root=data_root,
        manifest_root=manifest_root,
        source_dir=source_root,
    )

    train = json.loads((data_root / "livemath/train/items.json").read_text())
    assert [f"{row['month']}:{row['no']}" for row in train] == ["202601:2"]
    test = json.loads((data_root / "livemath/test/items.json").read_text())
    assert len(test) == 1
    assert not (data_root / "livemath/dev").exists()
    assert [choice["text"] for choice in test[0]["mcq"]["choices"]] == [
        "wrong-3-2",
        "right-3",
        "wrong-3-1",
    ]
    assert test[0]["mcq"]["correct_choice"] == {"label": "B", "text": "right-3"}
    choices = train[0]["mcq"]["choices"]
    correct = train[0]["mcq"]["correct_choice"]
    assert len(choices) == 3
    assert next(choice["text"] for choice in choices if choice["label"] == correct["label"]) == "right-2"


def test_checked_in_manifests_match_default_release_scope() -> None:
    manifest_root = Path(__file__).resolve().parents[1] / "data/manifests"
    expected_outputs = {"train", "test"}

    for dataset in ("searchqa", "docvqa", "livemath"):
        manifest = json.loads((manifest_root / dataset / "manifest.json").read_text())
        outputs = {
            name
            for name, spec in manifest["splits"].items()
            if bool(spec.get("materialize", True))
        }
        assert outputs == expected_outputs

        for split, spec in manifest["splits"].items():
            payload = json.loads((manifest_root / dataset / "splits" / f"{split}.json").read_text())
            ids = payload["ids"]
            assert len(ids) == int(spec["count"])
            assert len(ids) == len(set(ids))
            digest = __import__("hashlib").sha256(("\n".join(ids) + "\n").encode()).hexdigest()
            assert digest == spec["id_order_sha256"]

    livemath = json.loads((manifest_root / "livemath/manifest.json").read_text())
    assert livemath["transforms"]["split_order"] == ["train", "dev", "test"]
    assert livemath["splits"]["dev"]["materialize"] is False
