from __future__ import annotations

from pathlib import Path

import pytest

from cobras.scripts.experiment import Pipeline, build_parser, main
from cobras.scripts.release_config import resolve_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"
ENV_FILE = ROOT / ".env.example"


def test_default_release_config_has_stable_output_layout(tmp_path: Path) -> None:
    config = resolve_config(
        CONFIG,
        ENV_FILE,
        overrides={
            "experiment.datasets": ["searchqa"],
            "experiment.trials": [2],
            "experiment.output_root": str(tmp_path),
        },
    )

    assert config.harness.backend == "native"
    assert config.target.reasoning_effort == "none"
    assert config.teacher.reasoning_effort == "medium"
    assert config.target.max_completion_tokens == 16384
    assert config.teacher.max_completion_tokens == 16384
    assert config.baseline_root("searchqa") == (
        tmp_path
        / "baselines"
        / "searchqa"
        / f"target_{config.target_tag}"
        / "harness_native"
        / "train_50"
    )
    assert config.init_root("searchqa", 2).name == "trial_002"
    assert config.run_root("searchqa", 2).name == "trial_002"


def test_endpoint_changes_do_not_change_experiment_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TARGET_BASE_URL", "http://127.0.0.1:8001/v1")
    config = resolve_config(
        CONFIG,
        ENV_FILE,
        overrides={
            "experiment.datasets": ["alfworld"],
            "experiment.output_root": str(tmp_path),
        },
    )

    identity = config.identity("alfworld", 1)
    assert "base_url" not in str(identity)
    assert identity["task"]["max_turns"] == 30


def test_removed_harness_is_rejected() -> None:
    with pytest.raises(ValueError, match="harness.backend"):
        resolve_config(
            CONFIG,
            ENV_FILE,
            overrides={"harness.backend": "qwen_code"},
        )


def test_codex_harness_uses_chroot_wrapper_and_harness_timeout(tmp_path: Path) -> None:
    config = resolve_config(
        CONFIG,
        ENV_FILE,
        overrides={
            "experiment.datasets": ["searchqa"],
            "experiment.output_root": str(tmp_path),
            "harness.backend": "codex",
            "harness.timeout_seconds": 321,
        },
    )
    pipeline = Pipeline(config, dry_run=True)
    args = pipeline._eval_args(
        "searchqa",
        split="train",
        limit=1,
        out_root=tmp_path / "baseline",
        seed=42,
    )

    assert pipeline.env["CODEX_EXEC_PATH"].endswith(
        "scripts/harness_scripts/codex_chroot_exec.py"
    )
    assert pipeline.env["COBRAS_CODEX_ALLOWED_ROOT"] == str(tmp_path)
    assert args[args.index("--exec-timeout") + 1] == "321"


def test_cli_accepts_configured_datasets_when_no_positional_is_given() -> None:
    args = build_parser().parse_args(["run", "--dry-run"])

    assert args.dataset == []


def test_cli_token_overrides_apply_to_target_tasks_and_teacher(tmp_path: Path) -> None:
    config = resolve_config(
        CONFIG,
        ENV_FILE,
        overrides={
            "experiment.datasets": ["searchqa"],
            "experiment.output_root": str(tmp_path),
            "target.max_completion_tokens": 2048,
            "teacher.max_completion_tokens": 4096,
            "datasets.searchqa.max_completion_tokens": 2048,
        },
    )

    assert config.target.max_completion_tokens == 2048
    assert config.teacher.max_completion_tokens == 4096
    assert config.dataset("searchqa").max_completion_tokens == 2048


def test_end_to_end_dry_run_builds_all_stages_without_secrets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "run",
            "searchqa",
            "--config",
            str(CONFIG),
            "--env-file",
            str(ENV_FILE),
            "--output-root",
            str(tmp_path),
            "--trials",
            "1",
            "--workers",
            "1",
            "--train-size",
            "2",
            "--test-size",
            "2",
            "--rounds",
            "1",
            "--pool-size",
            "1",
            "--prune-count",
            "0",
            "--init-count",
            "1",
            "--init-sample-size",
            "2",
            "--init-parallel",
            "1",
            "--embedding-backend",
            "hash",
            "--initial-eval",
            "none",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert "baseline dataset=searchqa" in output
    assert "init dataset=searchqa trial=trial_001" in output
    assert "trial_001/seed_01" in output
    assert "optimize dataset=searchqa trial=trial_001" in output
    assert "kind=baseline" in output
    assert "kind=optimized" in output
    assert "--api-key-env TARGET_API_KEY" in output
    assert "TARGET_API_KEY=" not in output
