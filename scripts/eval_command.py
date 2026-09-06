from __future__ import annotations

import sys
from pathlib import Path


def build_eval_command(
    *,
    dataset: str,
    split: str,
    limit: int,
    workers: int,
    out_root: Path,
    skill_path: Path | None = None,
    provider: str = "openrouter",
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    reasoning_effort: str = "",
    mode: str = "multi",
    max_turns: int = 0,
    max_completion_tokens: int = 0,
    task_timeout: int = 0,
    use_eval_feedback: bool = False,
    data_path: Path | None = None,
    sample_seed: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "cobras.scripts.run_baseline",
        "--dataset",
        dataset,
        "--split",
        split,
        "--limit",
        str(limit),
        "--workers",
        str(workers),
        "--provider",
        provider,
        "--out-root",
        str(out_root),
        "--mode",
        mode,
    ]
    optional_values = [
        ("--model", model),
        ("--base-url", base_url),
        ("--api-key", api_key),
        ("--reasoning-effort", reasoning_effort),
    ]
    for flag, value in optional_values:
        if value:
            command.extend([flag, value])
    if skill_path is not None:
        command.extend(["--skill-path", str(skill_path)])
    if max_turns > 0:
        command.extend(["--max-turns", str(max_turns)])
    if max_completion_tokens > 0:
        command.extend(["--max-completion-tokens", str(max_completion_tokens)])
    if task_timeout > 0:
        command.extend(["--task-timeout", str(task_timeout)])
    if use_eval_feedback:
        command.append("--use-eval-feedback")
    if data_path is not None:
        command.extend(["--data-path", str(data_path)])
    if sample_seed is not None:
        command.extend(["--sample-seed", str(sample_seed)])
    return command
