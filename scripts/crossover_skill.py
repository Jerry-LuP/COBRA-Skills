from __future__ import annotations

import argparse
import subprocess
import sys

from cobras.cobras_core.config import DATASETS, get_dataset
from cobras.cobras_core.paths import COBRAS_ROOT, ensure_project_paths


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run dataset crossover skill generation.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    return parser.parse_known_args()


def main() -> None:
    ensure_project_paths()
    args, forwarded_args = parse_args()
    module = get_dataset(args.dataset).crossover_module
    if not module:
        raise SystemExit(
            f"Dataset {args.dataset} uses crossover inside its COBRAS runner. "
            "Use run_cobras.py, or add a dedicated crossover adapter for this dataset."
        )
    completed = subprocess.run(
        [sys.executable, "-m", module, *forwarded_args],
        cwd=COBRAS_ROOT,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
