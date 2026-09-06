from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _parse_env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_files(paths: Iterable[Path]) -> list[Path]:
    loaded: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :].strip()
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if not key:
                continue
            os.environ.setdefault(key, _parse_env_value(value))
        loaded.append(path)
    return loaded


def _first_env(*keys: str) -> str:
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def resolve_chat_provider(provider: str | None) -> dict[str, str]:
    normalized = str(provider or os.environ.get("COBRAS_PROVIDER", "openrouter")).strip().lower() or "openrouter"
    if normalized == "openrouter":
        return {
            "provider": "openrouter",
            "model": _first_env("OPENROUTER_COBRAS_MODEL", "COBRAS_MODEL", "MODEL"),
            "base_url": _first_env("OPENROUTER_BASE_URL", "COBRAS_BASE_URL", "OPENAI_BASE_URL"),
            "api_key": _first_env("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        }
    if normalized == "yunwu":
        return {
            "provider": "yunwu",
            "model": _first_env("YUNWU_COBRAS_MODEL", "COBRAS_MODEL", "MODEL"),
            "base_url": _first_env("YUNWU_BASE_URL", "YUNWU_OPENAI_BASE_URL", "COBRAS_BASE_URL", "OPENAI_BASE_URL"),
            "api_key": _first_env("YUNWU_API_KEY", "YUNWU_OPENAI_API_KEY", "OPENAI_API_KEY"),
        }
    if normalized in {"custom", "local"}:
        prefix = "TARGET" if normalized == "local" else "TEACHER"
        return {
            "provider": normalized,
            "model": _first_env(f"{prefix}_MODEL", "COBRAS_MODEL", "MODEL"),
            "base_url": _first_env(f"{prefix}_BASE_URL", "COBRAS_BASE_URL", "OPENAI_BASE_URL"),
            "api_key": _first_env(f"{prefix}_API_KEY", "OPENAI_API_KEY"),
        }
    raise ValueError(
        f"Unsupported provider={provider!r}. Use openrouter, yunwu, custom, or local."
    )
