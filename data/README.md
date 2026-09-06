# Benchmark Data

COBRA-Skills preserves fixed experiment splits without applying the repository's
Apache-2.0 license to third-party benchmark content.

| Benchmark | What is checked in | Upstream terms |
| --- | --- | --- |
| SearchQA | Ordered IDs only (`data/manifests/searchqa/`) | The selected Hugging Face mirror does not declare a dataset license; payload excluded |
| DocVQA | Ordered question IDs only (`data/manifests/docvqa/`) | Mirror declares Apache-2.0; official document data is portal-distributed; payload excluded |
| LiveMath | Ordered month/item IDs only (`data/manifests/livemath/`) | No clear monthly-payload license located; payload excluded |
| SocialMaze-Hard | Selected task records and split metadata | CC BY 4.0 |
| SpreadsheetBench | Verified spreadsheet subset and split metadata | CC BY-SA 4.0 |
| ALFWorld | Selected game assets and split metadata | MIT |

See `THIRD_PARTY_NOTICES.md` for sources, attribution, modification notices,
and precise file scope.

## Materialization

Install the optional dependencies from the repository root:

```bash
python -m pip install -e '.[data]'
```

Then materialize the excluded payloads as needed:

```bash
cobra-materialize-data searchqa
cobra-materialize-data docvqa
cobra-materialize-data livemath --source-dir /path/to/livemath/monthly/files
```

SearchQA and DocVQA are streamed from the upstream Hugging Face mirrors and only
selected IDs are retained. Each command writes exactly the default COBRA-Skills
evaluation subset: 50 ordered training records and 100 ordered test records.

LiveMath must be supplied as upstream monthly `qa_*_final.json` files. Its
materializer selects the fixed IDs and repeats the release's deterministic
choice shuffle with seed 42. The 27 historical dev IDs are used only to advance
the shuffle RNG between train and test; no dev payload is written.

Generated payloads are ignored by Git. `--force` deletes and rebuilds only the
materialized directories for the selected benchmark. `--data-root` and
`--manifest-root` can be used for non-default layouts.

Materialization is an action taken by the user. Review and comply with the
upstream terms before downloading or using any benchmark.
