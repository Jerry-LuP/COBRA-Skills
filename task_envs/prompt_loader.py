"""Prompt loading utilities for COBRAS task environments.

Prompts are stored under each task environment:
``task_envs/<env>/prompts/<name>.md``.
"""
from __future__ import annotations

import os

_TASK_ENVS_DIR = os.path.dirname(os.path.abspath(__file__))

_cache: dict[str, str] = {}


def _read_file(path: str) -> str | None:
    if path in _cache:
        return _cache[path]
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        content = f.read()
    _cache[path] = content
    return content


def load_prompt(name: str, env: str | None = None) -> str:
    """Load an environment-specific prompt by name.

    Lookup order:
      1. ``task_envs/{env}/prompts/{name}.md``  (if *env* given)

    Raises ``FileNotFoundError`` if the env prompt does not exist.
    """
    if env is None:
        raise FileNotFoundError(f"Prompt '{name}' requires an env name.")
    env_path = os.path.join(_TASK_ENVS_DIR, env, "prompts", f"{name}.md")
    content = _read_file(env_path)
    if content is not None:
        return content

    searched = [os.path.join("task_envs", env, "prompts", f"{name}.md")]
    raise FileNotFoundError(
        f"Prompt '{name}' not found. Searched: {', '.join(searched)}"
    )


def clear_cache() -> None:
    """Clear the prompt file cache (useful for testing)."""
    _cache.clear()
