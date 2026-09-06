from __future__ import annotations

import os
import re
from pathlib import Path

from cobras.cobras_core.model import (
    configure_openai,
    set_reasoning_effort,
    set_target_backend,
    set_target_deployment,
)
from cobras.task_envs.eval_types import EvalConfig


def api_key_for(config: EvalConfig) -> str:
    return config.api_key or os.environ.get(config.provider.api_key_env, "")


def configure_chat_target(
    config: EvalConfig,
    *,
    default_reasoning_effort: str = "",
    default_max_completion_tokens: int,
    default_timeout: int,
) -> tuple[str, str | None, int, int]:
    """Configure the existing model router while retaining task-level defaults."""

    reasoning = config.reasoning_effort.strip() or default_reasoning_effort.strip() or None
    max_tokens = config.max_completion_tokens or default_max_completion_tokens
    timeout = config.task_timeout or default_timeout
    timeout_override = os.environ.get("COBRAS_TARGET_TASK_TIMEOUT_SECONDS", "").strip()
    if timeout_override:
        try:
            timeout = max(1, int(float(timeout_override)))
        except ValueError as exc:
            raise ValueError(
                "COBRAS_TARGET_TASK_TIMEOUT_SECONDS must be a positive number, "
                f"got {timeout_override!r}"
            ) from exc
    set_reasoning_effort(reasoning)
    set_target_deployment(config.provider.model)

    requested_backend = os.environ.get("TARGET_BACKEND", "").strip().lower()
    if requested_backend in {"codex", "codex_exec", "claude_code_exec"}:
        backend = "codex_exec" if requested_backend == "codex" else requested_backend
        set_target_backend(backend)
        return backend, reasoning, max_tokens, timeout

    api_key = api_key_for(config)
    if not api_key and config.provider.provider != "local":
        raise SystemExit(f"Missing API key: {config.provider.api_key_env}")

    set_target_backend("openai_chat")
    configure_openai(
        endpoint=config.provider.base_url,
        api_key=api_key,
        auth_mode="openai_compatible",
        target_endpoint=config.provider.base_url,
        target_api_key=api_key,
        target_auth_mode="openai_compatible",
    )
    return "openai_chat", reasoning, max_tokens, timeout


def load_skill(path: Path | None) -> tuple[str, str]:
    if path is None:
        return "", ""
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "skill.md"
    if not resolved.exists():
        raise FileNotFoundError(f"Skill file not found: {resolved}")
    return str(resolved), resolved.read_text(encoding="utf-8")


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"
