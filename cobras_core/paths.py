from __future__ import annotations

import os
import re
import sys
from pathlib import Path


COBRAS_ROOT = Path(__file__).resolve().parents[1]


def ensure_project_paths() -> None:
    """Prefer the standalone project packages over any old workspace packages."""
    for path in [
        COBRAS_ROOT,
    ]:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def model_slug(model: str) -> str:
    value = (model or "model").strip().split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "model"


def default_results_root() -> Path:
    return COBRAS_ROOT / "results" / "cobras"


def read_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_default_env_files(dataset: str = "") -> None:
    """Load the release's single env file; ``dataset`` is kept for API compatibility."""
    del dataset
    read_env_file(COBRAS_ROOT / ".env")
