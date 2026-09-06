"""Runtime backend configuration for optimizer/target model calls."""
from __future__ import annotations

import os

from cobras.cobras_core.model.common import default_model_for_backend, normalize_backend_name


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


OPTIMIZER_BACKEND = normalize_backend_name(os.environ.get("OPTIMIZER_BACKEND", "openai_chat"))
TARGET_BACKEND = normalize_backend_name(os.environ.get("TARGET_BACKEND", "openai_chat"))

CODEX_EXEC_PATH = os.environ.get("CODEX_EXEC_PATH", "codex")
CODEX_EXEC_SANDBOX = os.environ.get("CODEX_EXEC_SANDBOX", "read-only")
CODEX_EXEC_PROFILE = os.environ.get("CODEX_EXEC_PROFILE", "")
CODEX_EXEC_REASONING_EFFORT = os.environ.get("CODEX_EXEC_REASONING_EFFORT", "none")
CODEX_EXEC_USE_SDK = os.environ.get("CODEX_EXEC_USE_SDK", "cli")
CODEX_EXEC_NETWORK_ACCESS = _parse_bool(os.environ.get("CODEX_EXEC_NETWORK_ACCESS"), False)
CODEX_EXEC_WEB_SEARCH = _parse_bool(os.environ.get("CODEX_EXEC_WEB_SEARCH"), False)
CODEX_EXEC_APPROVAL_POLICY = os.environ.get("CODEX_EXEC_APPROVAL_POLICY", "never")
CODEX_EXEC_CONTEXT_WINDOW = os.environ.get("CODEX_EXEC_CONTEXT_WINDOW", "131072")
CODEX_EXEC_REQUEST_MAX_RETRIES = os.environ.get("CODEX_EXEC_REQUEST_MAX_RETRIES", "3")
CODEX_EXEC_STREAM_MAX_RETRIES = os.environ.get("CODEX_EXEC_STREAM_MAX_RETRIES", "3")
CODEX_EXEC_STREAM_IDLE_TIMEOUT_MS = os.environ.get("CODEX_EXEC_STREAM_IDLE_TIMEOUT_MS", "300000")

CLAUDE_CODE_EXEC_PATH = os.environ.get("CLAUDE_CODE_EXEC_PATH", "claude")
CLAUDE_CODE_EXEC_PROFILE = os.environ.get("CLAUDE_CODE_EXEC_PROFILE", "")
CLAUDE_CODE_EXEC_USE_SDK = os.environ.get("CLAUDE_CODE_EXEC_USE_SDK", "auto")
CLAUDE_CODE_EXEC_EFFORT = os.environ.get("CLAUDE_CODE_EXEC_EFFORT", "medium")
CLAUDE_CODE_EXEC_CONTEXT_WINDOW = os.environ.get("CLAUDE_CODE_EXEC_CONTEXT_WINDOW", "131072")
CLAUDE_CODE_EXEC_MAX_TOKENS = os.environ.get("CLAUDE_CODE_EXEC_MAX_TOKENS", "16384")


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        return default


EXEC_EMPTY_RESPONSE_RETRIES = max(0, _parse_int(os.environ.get("EXEC_EMPTY_RESPONSE_RETRIES"), 1))
CLAUDE_CODE_EXEC_MAX_THINKING_TOKENS = max(
    0,
    _parse_int(os.environ.get("CLAUDE_CODE_EXEC_MAX_THINKING_TOKENS"), 16384),
)


def set_optimizer_backend(backend: str) -> None:
    global OPTIMIZER_BACKEND
    OPTIMIZER_BACKEND = normalize_backend_name(backend or "openai_chat")
    if OPTIMIZER_BACKEND != "openai_chat":
        raise ValueError(
            f"Unsupported optimizer backend: {OPTIMIZER_BACKEND!r}. "
            "The supported value is 'openai_chat'."
        )
    os.environ["OPTIMIZER_BACKEND"] = OPTIMIZER_BACKEND


def get_optimizer_backend() -> str:
    return OPTIMIZER_BACKEND


def set_target_backend(backend: str) -> None:
    global TARGET_BACKEND
    TARGET_BACKEND = normalize_backend_name(backend or "openai_chat")
    if TARGET_BACKEND not in {"openai_chat", "codex_exec", "claude_code_exec"}:
        raise ValueError(
            f"Unsupported target backend: {TARGET_BACKEND!r}. "
            "Supported values are 'openai_chat', 'codex_exec', and 'claude_code_exec'."
        )
    os.environ["TARGET_BACKEND"] = TARGET_BACKEND


def get_target_backend() -> str:
    return TARGET_BACKEND


def is_target_exec_backend() -> bool:
    return TARGET_BACKEND in {"codex_exec", "claude_code_exec"}


def is_optimizer_chat_backend() -> bool:
    return OPTIMIZER_BACKEND == "openai_chat"


def is_target_chat_backend() -> bool:
    return TARGET_BACKEND == "openai_chat"


def configure_codex_exec(
    *,
    path: str | None = None,
    sandbox: str | None = None,
    profile: str | None = None,
    reasoning_effort: str | None = None,
    use_sdk: str | None = None,
    network_access: bool | str | None = None,
    web_search: bool | str | None = None,
    approval_policy: str | None = None,
    context_window: int | str | None = None,
) -> None:
    global CODEX_EXEC_PATH, CODEX_EXEC_SANDBOX, CODEX_EXEC_PROFILE, CODEX_EXEC_REASONING_EFFORT
    global CODEX_EXEC_USE_SDK, CODEX_EXEC_NETWORK_ACCESS, CODEX_EXEC_WEB_SEARCH
    global CODEX_EXEC_APPROVAL_POLICY, CODEX_EXEC_CONTEXT_WINDOW

    if path is not None:
        CODEX_EXEC_PATH = str(path).strip() or "codex"
        os.environ["CODEX_EXEC_PATH"] = CODEX_EXEC_PATH
    if sandbox is not None:
        value = str(sandbox).strip() or "read-only"
        if value not in {"read-only", "workspace-write", "danger-full-access"}:
            raise ValueError(f"Unsupported Codex sandbox mode: {value!r}")
        CODEX_EXEC_SANDBOX = value
        os.environ["CODEX_EXEC_SANDBOX"] = value
    if profile is not None:
        CODEX_EXEC_PROFILE = str(profile).strip()
        os.environ["CODEX_EXEC_PROFILE"] = CODEX_EXEC_PROFILE
    if reasoning_effort is not None:
        CODEX_EXEC_REASONING_EFFORT = str(reasoning_effort).strip() or "none"
        os.environ["CODEX_EXEC_REASONING_EFFORT"] = CODEX_EXEC_REASONING_EFFORT
    if use_sdk is not None:
        CODEX_EXEC_USE_SDK = str(use_sdk).strip().lower() or "cli"
        os.environ["CODEX_EXEC_USE_SDK"] = CODEX_EXEC_USE_SDK
    if network_access is not None:
        CODEX_EXEC_NETWORK_ACCESS = (
            _parse_bool(network_access, False)
            if isinstance(network_access, str)
            else bool(network_access)
        )
        os.environ["CODEX_EXEC_NETWORK_ACCESS"] = str(CODEX_EXEC_NETWORK_ACCESS).lower()
    if web_search is not None:
        CODEX_EXEC_WEB_SEARCH = (
            _parse_bool(web_search, False)
            if isinstance(web_search, str)
            else bool(web_search)
        )
        os.environ["CODEX_EXEC_WEB_SEARCH"] = str(CODEX_EXEC_WEB_SEARCH).lower()
    if approval_policy is not None:
        CODEX_EXEC_APPROVAL_POLICY = str(approval_policy).strip() or "never"
        os.environ["CODEX_EXEC_APPROVAL_POLICY"] = CODEX_EXEC_APPROVAL_POLICY
    if context_window is not None:
        CODEX_EXEC_CONTEXT_WINDOW = str(max(1, _parse_int(str(context_window), 131072)))
        os.environ["CODEX_EXEC_CONTEXT_WINDOW"] = CODEX_EXEC_CONTEXT_WINDOW


def get_codex_exec_config() -> dict[str, str | bool | int]:
    return {
        "path": CODEX_EXEC_PATH,
        "sandbox": CODEX_EXEC_SANDBOX,
        "profile": CODEX_EXEC_PROFILE,
        "reasoning_effort": CODEX_EXEC_REASONING_EFFORT,
        "use_sdk": CODEX_EXEC_USE_SDK,
        "network_access": CODEX_EXEC_NETWORK_ACCESS,
        "web_search": CODEX_EXEC_WEB_SEARCH,
        "approval_policy": CODEX_EXEC_APPROVAL_POLICY,
        "context_window": max(1, _parse_int(CODEX_EXEC_CONTEXT_WINDOW, 131072)),
        "request_max_retries": max(0, _parse_int(CODEX_EXEC_REQUEST_MAX_RETRIES, 3)),
        "stream_max_retries": max(0, _parse_int(CODEX_EXEC_STREAM_MAX_RETRIES, 3)),
        "stream_idle_timeout_ms": max(1000, _parse_int(CODEX_EXEC_STREAM_IDLE_TIMEOUT_MS, 300000)),
        "empty_response_retries": EXEC_EMPTY_RESPONSE_RETRIES,
    }


def configure_claude_code_exec(
    *,
    path: str | None = None,
    profile: str | None = None,
    use_sdk: str | None = None,
    effort: str | None = None,
    max_thinking_tokens: int | str | None = None,
    context_window: int | str | None = None,
    max_tokens: int | str | None = None,
) -> None:
    global CLAUDE_CODE_EXEC_PATH, CLAUDE_CODE_EXEC_PROFILE, CLAUDE_CODE_EXEC_USE_SDK
    global CLAUDE_CODE_EXEC_EFFORT, CLAUDE_CODE_EXEC_MAX_THINKING_TOKENS
    global CLAUDE_CODE_EXEC_CONTEXT_WINDOW, CLAUDE_CODE_EXEC_MAX_TOKENS
    if path is not None:
        CLAUDE_CODE_EXEC_PATH = str(path).strip() or "claude"
        os.environ["CLAUDE_CODE_EXEC_PATH"] = CLAUDE_CODE_EXEC_PATH
    if profile is not None:
        CLAUDE_CODE_EXEC_PROFILE = str(profile).strip()
        os.environ["CLAUDE_CODE_EXEC_PROFILE"] = CLAUDE_CODE_EXEC_PROFILE
    if use_sdk is not None:
        CLAUDE_CODE_EXEC_USE_SDK = str(use_sdk).strip().lower() or "auto"
        os.environ["CLAUDE_CODE_EXEC_USE_SDK"] = CLAUDE_CODE_EXEC_USE_SDK
    if effort is not None:
        CLAUDE_CODE_EXEC_EFFORT = str(effort).strip().lower() or "medium"
        os.environ["CLAUDE_CODE_EXEC_EFFORT"] = CLAUDE_CODE_EXEC_EFFORT
    if max_thinking_tokens is not None:
        CLAUDE_CODE_EXEC_MAX_THINKING_TOKENS = max(
            0,
            _parse_int(str(max_thinking_tokens), 16384),
        )
        os.environ["CLAUDE_CODE_EXEC_MAX_THINKING_TOKENS"] = str(CLAUDE_CODE_EXEC_MAX_THINKING_TOKENS)
    if context_window is not None:
        CLAUDE_CODE_EXEC_CONTEXT_WINDOW = str(
            max(1, _parse_int(str(context_window), 131072))
        )
        os.environ["CLAUDE_CODE_EXEC_CONTEXT_WINDOW"] = CLAUDE_CODE_EXEC_CONTEXT_WINDOW
    if max_tokens is not None:
        CLAUDE_CODE_EXEC_MAX_TOKENS = str(
            max(1, _parse_int(str(max_tokens), 16384))
        )
        os.environ["CLAUDE_CODE_EXEC_MAX_TOKENS"] = CLAUDE_CODE_EXEC_MAX_TOKENS


def get_claude_code_exec_config() -> dict[str, str | int]:
    return {
        "path": CLAUDE_CODE_EXEC_PATH,
        "profile": CLAUDE_CODE_EXEC_PROFILE,
        "use_sdk": CLAUDE_CODE_EXEC_USE_SDK,
        "effort": CLAUDE_CODE_EXEC_EFFORT,
        "max_thinking_tokens": CLAUDE_CODE_EXEC_MAX_THINKING_TOKENS,
        "context_window": max(1, _parse_int(CLAUDE_CODE_EXEC_CONTEXT_WINDOW, 131072)),
        "max_tokens": max(1, _parse_int(CLAUDE_CODE_EXEC_MAX_TOKENS, 16384)),
        "empty_response_retries": EXEC_EMPTY_RESPONSE_RETRIES,
    }
